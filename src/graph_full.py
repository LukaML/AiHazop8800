# src/graph_full.py
"""LangGraph state machine defining the full HAZOP pipeline with repair loops.

The pipeline flows:  START → RAG_PREP → L1 → L2 → L3 → Holistic Review → END

Each stage (L1, L2, L3) follows the same cycle:
  INIT → VALIDATE → (if issues) REVIEW → REGEN → VALIDATE → …
with a hard cap of MAX_STAGE_REPAIR rounds before proceeding.

After L3 passes, a holistic cross-stage review checks consistency across all
rows and may cascade regeneration back through earlier stages (L1→L2→L3).
"""
from __future__ import annotations
import logging
import re
from typing import TypedDict, List, Dict, Any, Literal
from copy import deepcopy
from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

# === Shared row utilities ====================================================
from .row_utils import (
    _ensure_rows_list,
    _row_get,
    _row_to_dict,
    _ensure_row_ids,
    _suffix,
    _pydantic_to_dicts,
    _merge_rag_notes,
    _rows_equal_by_suffix,
    _merge_full_by_key,
    _merge_subset_by_suffix,
    _pick_rows_by_suffix,
)

# === Chains / reviewers (existing API) =======================================
from .chains import (
    # L1
    l1_init_full_coverage,
    l1_to_reviewer,
    reviewer_to_l1,
    reviewer_patch_l1,
    # L2
    l2_init_from_l1,
    l2_to_reviewer,
    reviewer_patch_l2,
    # L3
    l3_init_from_l2,
    l3_to_reviewer,
    reviewer_patch_l3,
    # Holistic
    l3_pass_to_reviewer_holistic,
)

# === Validators & helpers (existing API) =====================================
from .validators import (
    validate_l1_payload,
    validate_l2_payload,
    validate_l3_payload,
    build_validator_report,
)

# === RAG utils (optional) ===================================================
try:
    from .rag_utils import load_documents, build_index, search  # type: ignore
except Exception:
    load_documents = build_index = search = None  # type: ignore



# ============================================================================
#                                STATE
# The graph state is a TypedDict carrying all data between nodes: input
# parameters, RAG artefacts, per-stage row lists, validation flags, repair
# counters, and holistic review state.
# ============================================================================
class HazopGraphState(TypedDict, total=False):
    functions: List[str]
    guideword_map: Dict[str, Any]
    max_devs_per_gw: int

    # --- RAG (optional) ---
    rag_enabled: bool
    rag_paths: List[str]
    rag_embedder: str  # 'local' | 'openai'
    rag_vec: Any
    rag_mat: Any
    rag_docs: List[str]
    rag_notes_l1: str
    rag_notes_l2: str
    rag_notes_l3: str
    rag_min_sim: float

    notes: str

    rows_l1: List[Dict[str, Any]]
    l1_ok: bool
    l1_issues: List[str]
    l1_suggestion: str
    l1_repair_round: int

    rows_l2: List[Dict[str, Any]]
    l2_ok: bool
    l2_issues: List[str]
    l2_suggestion: str
    l2_repair_round: int

    rows_l3: List[Dict[str, Any]]
    l3_ok: bool
    l3_issues: List[str]
    l3_suggestion: str
    l3_repair_round: int

    holistic_ok: bool
    holistic_issues: List[str]
    holistic_round: int
    holistic_suggestion: str
    holistic_scope: str
    holistic_target_ids: List[str]

# ============================================================================
#                            GENERIC HELPERS
# ============================================================================


def _extract_row_ids_from_issues(issues: List[str]) -> List[str]:
    """
    Extract a list of unique row_id strings (e.g., "L1-3") from a list
    of arbitrary issue strings.  The holistic reviewer may return
    messages that embed row identifiers in various contexts (e.g.,
    "Inconsistent cause for L1-3: …" or "row[0] (row_id=L1-3)").  A
    simple regex without word-boundary anchors is used to match any
    occurrences of the pattern `L<stage>-<number>`, regardless of
    trailing punctuation.  Duplicate identifiers are de-duplicated
    while preserving order.

    Args:
        issues: A list of arbitrary strings from the holistic reviewer.

    Returns:
        A list of unique row_ids in the order they first appear.
    """
    row_ids: List[str] = []
    if not issues:
        return []
    # Match any substring like "L1-3", "L2-10" or "L3-25".  Do not
    # anchor with word boundaries because row identifiers may be
    # adjacent to punctuation (e.g., "L1-3:" or "row_id=L1-3").
    pat = re.compile(r'(L[123]-\d+)')
    for s in issues:
        if s is None:
            continue
        row_ids.extend(pat.findall(str(s)))
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: List[str] = []
    for rid in row_ids:
        if rid not in seen:
            out.append(rid)
            seen.add(rid)
    return out

def _rag_notes(state: HazopGraphState, stage_key: str) -> str:
    """Merge base notes with RAG notes for a given stage."""
    return _merge_rag_notes(state.get("notes") or "", state.get(stage_key) or "")

def _suffixes_from_row_ids(row_ids: List[str]) -> List[str]:
    return sorted({ _suffix(rid) for rid in (row_ids or []) if isinstance(rid, str) and rid })

def _scope_to_highest_stage(scope: str) -> Literal["L1","L2","L3"]:
    s = (scope or "ALL").upper()
    if s == "L1": return "L1"
    if s == "L2": return "L2"
    if s == "L3": return "L3"
    return "L1"  # default to earliest/highest

# ============================================================================
#                          REGEN NODE FACTORY
# All three regen nodes (L1, L2, L3) share the same pattern: extract hints
# from the reviewer, attempt a targeted patch on the affected rows, and
# fall back to full regeneration if the patch didn't change anything.
# The factory below creates stage-specific nodes from a shared template.
# ============================================================================
# Configuration for stage-specific regen behavior. Each stage has:
# - rows_key: state key for the stage's rows
# - suggestion_key: state key for reviewer suggestion
# - issues_key: state key for validator issues
# - repair_round_key: state key for repair round counter
# - field: the field to patch (deviation, cause, effect)
# - prefix: row_id prefix (L1, L2, L3)
# - patcher: function to call for targeted patch
# - fallback_uses_full_key_merge: True for L1, False for L2/L3
from typing import Callable, Any as TypingAny

REGEN_CONFIG = {
    "L1": {
        "rows_key": "rows_l1",
        "suggestion_key": "l1_suggestion",
        "issues_key": "l1_issues",
        "repair_round_key": "l1_repair_round",
        "field": "deviation",
        "prefix": "L1",
        "patcher": lambda subset, hints, notes="": reviewer_patch_l1(subset, hints, notes=notes),
        "fallback_uses_full_key_merge": True,
    },
    "L2": {
        "rows_key": "rows_l2",
        "suggestion_key": "l2_suggestion",
        "issues_key": "l2_issues",
        "repair_round_key": "l2_repair_round",
        "field": "cause",
        "prefix": "L2",
        "patcher": lambda subset, hints, notes="": reviewer_patch_l2(subset, hints, notes=notes),
        "fallback_uses_full_key_merge": False,
    },
    "L3": {
        "rows_key": "rows_l3",
        "suggestion_key": "l3_suggestion",
        "issues_key": "l3_issues",
        "repair_round_key": "l3_repair_round",
        "field": "effect",
        "prefix": "L3",
        "patcher": lambda subset, hints, notes="": reviewer_patch_l3(subset, hints, notes=notes),
        "fallback_uses_full_key_merge": False,
    },
}


def _l1_fallback_regen(state: HazopGraphState, target_suffixes: List[str], sug: str) -> List[Dict[str, Any]]:
    """L1-specific fallback: regenerate via reviewer_to_l1 and merge by key."""
    notes = _rag_notes(state, "rag_notes_l1")
    regenerated = reviewer_to_l1(
        state["functions"],
        state["guideword_map"],
        [sug] if sug else [],
        state.get("max_devs_per_gw", 2),
        notes=notes,
    ) or []
    regenerated = _ensure_rows_list(regenerated)
    _ensure_row_ids("L1", regenerated)
    current = [_row_to_dict(r) for r in (state["rows_l1"] or [])]
    return _merge_full_by_key(
        current, regenerated, target_suffixes,
        key_fields=["function", "guideword"]
    )


def _l2_fallback_regen(state: HazopGraphState, target_suffixes: List[str], current: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """L2-specific fallback: regenerate via l2_init_from_l1 and merge by suffix."""
    notes = _rag_notes(state, "rag_notes_l2")
    l2_full = l2_init_from_l1(state["rows_l1"], notes=notes) or []
    l2_full = _ensure_rows_list(l2_full)
    _ensure_row_ids("L2", l2_full)
    l2_subset = _pick_rows_by_suffix(l2_full, target_suffixes)
    return _merge_subset_by_suffix(current, l2_subset)


def _l3_fallback_regen(state: HazopGraphState, target_suffixes: List[str], current: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """L3-specific fallback: regenerate via l3_init_from_l2 and merge by suffix."""
    notes = _rag_notes(state, "rag_notes_l3")
    l3_full = l3_init_from_l2(state["rows_l2"], notes=notes) or []
    l3_full = _ensure_rows_list(l3_full)
    _ensure_row_ids("L3", l3_full)
    l3_subset = _pick_rows_by_suffix(l3_full, target_suffixes)
    return _merge_subset_by_suffix(current, l3_subset)


def _create_regen_node(stage: str) -> Callable[[HazopGraphState], Dict[str, TypingAny]]:
    """Factory to create stage-specific regen nodes.

    All three regen nodes (L1, L2, L3) follow the same pattern:
    1. Extract suggestion and target row IDs from state
    2. If suggestion exists, attempt targeted patch
    3. Merge patched subset back into current rows
    4. If patch failed or no change, fall back to full regeneration
    5. Return updated rows and increment repair round

    Stage-specific differences are handled via REGEN_CONFIG.
    """
    cfg = REGEN_CONFIG[stage]

    def regen_node(state: HazopGraphState) -> Dict[str, TypingAny]:
        if stage == "L3":
            logger.debug("L3_REGEN IN: rows_l3=%d", len(state['rows_l3']))

        sug = (state.get(cfg["suggestion_key"], "") or "").strip()
        target_ids = _extract_row_ids_from_issues(state.get(cfg["issues_key"], []))
        if not target_ids:
            return {cfg["repair_round_key"]: state.get(cfg["repair_round_key"], 0) + 1}
        target_suffixes = _suffixes_from_row_ids(target_ids)

        current = [_row_to_dict(r) for r in (state[cfg["rows_key"]] or [])]
        patched_subset: List[Dict[str, TypingAny]] = []

        if sug:
            subset = _pick_rows_by_suffix(current, target_suffixes)
            if stage == "L3":
                logger.debug("L3_REGEN: target_suffixes=%s subset=%d", target_suffixes, len(subset))
            row_hints = [
                {"row_id": rid, "fields": [cfg["field"]], "suggestion": sug}
                for rid in target_ids
            ]
            notes = _rag_notes(state, f"rag_notes_{stage.lower()}")
            patched_subset = cfg["patcher"](subset, row_hints, notes=notes) or []
            patched_subset = _ensure_rows_list(patched_subset)
            _ensure_row_ids(cfg["prefix"], patched_subset)

        merged = _merge_subset_by_suffix(current, patched_subset) if patched_subset else current

        # Fallback if patch failed or no change
        if (not patched_subset) or _rows_equal_by_suffix(current, merged):
            if stage == "L1":
                merged = _l1_fallback_regen(state, target_suffixes, sug)
            elif stage == "L2":
                merged = _l2_fallback_regen(state, target_suffixes, current)
            else:  # L3
                merged = _l3_fallback_regen(state, target_suffixes, current)

        if stage == "L3":
            logger.debug("L3_REGEN OUT: merged=%d", len(merged))

        return {
            cfg["rows_key"]: merged,
            cfg["repair_round_key"]: state.get(cfg["repair_round_key"], 0) + 1,
        }

    return regen_node


# Create regen nodes via factory
l1_regen_node = _create_regen_node("L1")
l2_regen_node = _create_regen_node("L2")
l3_regen_node = _create_regen_node("L3")


# ============================================================================
#                             PIPELINE CONSTANTS
# ============================================================================
MAX_STAGE_REPAIR = 2  # hard stop per stage — prevents infinite repair loops


# ============================================================================
#                                RAG NODES
# Retrieval-Augmented Generation: load documents once, then build per-stage
# context by searching the index with stage-appropriate queries.
# ============================================================================

def rag_prep_node(state: HazopGraphState):
    """Load documents and build retrieval index once per run.

    This keeps RAG LangGraph-native: the index is stored in the graph state
    and reused by subsequent stage context nodes.
    """
    if not state.get("rag_enabled", False):
        return {"rag_docs": [], "rag_vec": None, "rag_mat": None,
                "rag_notes_l1": "", "rag_notes_l2": "", "rag_notes_l3": ""}

    if not (load_documents and build_index and search):
        # RAG requested but utils unavailable; keep pipeline running.
        return {"rag_docs": [], "rag_vec": None, "rag_mat": None,
                "rag_notes_l1": "", "rag_notes_l2": "", "rag_notes_l3": ""}

    paths = [p for p in (state.get("rag_paths") or []) if str(p).strip()]
    if not paths:
        return {"rag_docs": [], "rag_vec": None, "rag_mat": None,
                "rag_notes_l1": "", "rag_notes_l2": "", "rag_notes_l3": ""}

    docs = load_documents(paths)
    if not docs:
        logger.warning("RAG enabled but no document chunks loaded from paths: %s", paths)
    vec, mat = build_index(docs, embedder=state.get("rag_embedder", "local"))
    return {"rag_docs": docs, "rag_vec": vec, "rag_mat": mat,
            "rag_notes_l1": "", "rag_notes_l2": "", "rag_notes_l3": ""}


def _rag_search_lines(state: HazopGraphState, queries: List[str], *, top_k: int = 2, max_chars: int = 12000) -> str:
    if not state.get("rag_enabled", False):
        return ""
    if not (search and state.get("rag_docs") and state.get("rag_vec") is not None and state.get("rag_mat") is not None):
        return ""

    embedder = (state.get("rag_embedder") or "local").lower()
    default_sim = 0.1 if embedder == "local" else 0.25
    min_sim = state.get("rag_min_sim") or default_sim

    lines: List[str] = []
    used = 0
    seen: set[str] = set()
    for qi, q in enumerate(queries):
        q = (q or "").strip()
        if not q or q in seen:
            continue
        seen.add(q)
        hits = search(q, state["rag_vec"], state["rag_mat"], state["rag_docs"], top_k=top_k, min_similarity=min_sim) or []
        if not hits:
            continue
        for i, hit in enumerate(hits):
            logger.debug("  RAG hit %d (query='%.50s'): %.120s", i + 1, q, hit)
        # Cap each hit to keep per-query lines manageable
        max_hit_chars = 400
        trimmed = [str(h)[:max_hit_chars] for h in hits]
        joined = " / ".join(trimmed)
        line = f"Context for '{q}': {joined}"
        if used + len(line) > max_chars:
            remaining = sum(1 for qq in queries[qi + 1:]
                           if (qq or "").strip() and (qq or "").strip() not in seen)
            if remaining:
                logger.info("RAG context truncated at %d/%d chars; ~%d queries skipped",
                            used, max_chars, remaining)
            break
        lines.append(line)
        used += len(line)
    if lines:
        logger.info("RAG context: %d/%d queries matched, %d chunks, %d chars",
                    len(lines), len(seen), sum(l.count(" / ") + 1 for l in lines), used)
    return "\n".join(lines).strip()


def rag_ctx_l1_node(state: HazopGraphState):
    # Query by function name (best signal before any rows exist)
    queries = list(state.get("functions") or [])
    return {"rag_notes_l1": _rag_search_lines(state, queries, top_k=3)}


def rag_ctx_l2_node(state: HazopGraphState):
    # Query by one deviation per function (avoids bias toward first function)
    rows = state.get("rows_l1") or []
    seen_fns: set[str] = set()
    queries: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        fn = r.get("function", "")
        dev = str(r.get("deviation") or "").strip()
        if fn not in seen_fns and dev:
            queries.append(dev)
            seen_fns.add(fn)
    if not queries:
        queries = list(state.get("functions") or [])
    return {"rag_notes_l2": _rag_search_lines(state, queries, top_k=2)}


def rag_ctx_l3_node(state: HazopGraphState):
    # Query by one cause per function (fallback to deviation)
    rows = state.get("rows_l2") or []
    seen_fns: set[str] = set()
    queries: List[str] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        fn = r.get("function", "")
        text = str(r.get("cause") or r.get("deviation") or "").strip()
        if fn not in seen_fns and text:
            queries.append(text)
            seen_fns.add(fn)
    if not queries:
        queries = list(state.get("functions") or [])
    return {"rag_notes_l3": _rag_search_lines(state, queries, top_k=2)}

# ============================================================================
#                        L1 NODES — deviation generation
# Init generates rows, Validate checks Pydantic + coverage, Review asks LLM
# for quality feedback, Regen patches or regenerates faulty rows.
# ============================================================================

def l1_init_node(state: HazopGraphState):
    logger.info("=== L1_INIT: Generating deviations for %d functions ===", len(state.get("functions") or []))
    # Pass optional notes (e.g., RAG context) to L1 generator.
    merged_notes = _merge_rag_notes(
        state.get("notes") or "",
        state.get("rag_notes_l1") or ""
    )
    rows = l1_init_full_coverage(
        state["functions"],
        state["guideword_map"],
        state.get("max_devs_per_gw", 2),
        merged_notes,
    ) or []
    rows = _ensure_rows_list(rows)
    _ensure_row_ids("L1", rows)

    logger.info("L1_INIT: Generated %d rows", len(rows))
    return {"rows_l1": rows, "l1_repair_round": 0}

def l1_validate_node(state: HazopGraphState):
    # Validate L1 rows using Pydantic models.  Convert the validated rows back
    # into plain dictionaries so the rest of the pipeline operates on a
    # consistent structure (dicts rather than Pydantic objects).  Without
    # conversion later nodes would receive a mix of model instances and dicts,
    # leading to subtle bugs when merging or serialising rows.
    rows = state["rows_l1"]
    norm_rows, ok, issues = validate_l1_payload(rows, state["functions"])
    norm_rows_dicts = _pydantic_to_dicts(norm_rows)
    logger.info("L1_VALIDATE: ok=%s issues=%d", ok, len(issues))
    return {"rows_l1": norm_rows_dicts, "l1_ok": ok, "l1_issues": issues}

def l1_reviewer_node(state: HazopGraphState):
    report = build_validator_report("L1", state.get("l1_issues", []),
                                    "Re-generate ONLY targeted L1 rows using the single reviewer suggestion.")
    notes = _rag_notes(state, "rag_notes_l1")
    decision = l1_to_reviewer(report, state["rows_l1"], notes=notes) or {}
    return {
        "l1_suggestion": (decision.get("suggestion") or "").strip(),
    }

# ============================================================================
#                        L2 NODES — cause generation
# Same init → validate → review → regen cycle as L1.
# ============================================================================
def l2_init_node(state: HazopGraphState):
    logger.info("=== L2_INIT: Adding causes to %d L1 rows ===", len(state.get("rows_l1") or []))
    merged_notes = _merge_rag_notes(
        state.get("notes") or "",
        state.get("rag_notes_l2") or ""
    )

    rows = l2_init_from_l1(state["rows_l1"], notes=merged_notes) or []
    rows = _ensure_rows_list(rows)
    _ensure_row_ids("L2", rows)

    expected = len(state.get("rows_l1") or [])
    logger.info("L2_INIT: Generated %d rows (expected %d)", len(rows), expected)
    if len(rows) != expected:
        logger.warning(
            "L2_INIT: Row count mismatch! Got %d L2 rows from %d L1 rows — some rows may be missing causes.",
            len(rows), expected,
        )
    return {"rows_l2": rows, "l2_repair_round": 0}

def l2_validate_node(state: HazopGraphState):
    # Validate L2 rows and normalise them to plain dicts.  See l1_validate_node
    # for rationale.
    rows = state["rows_l2"]
    norm_rows, ok, issues = validate_l2_payload(rows)
    norm_rows_dicts = _pydantic_to_dicts(norm_rows)
    logger.info("L2_VALIDATE: ok=%s issues=%d", ok, len(issues))
    return {"rows_l2": norm_rows_dicts, "l2_ok": ok, "l2_issues": issues}

def l2_reviewer_node(state: HazopGraphState):
    report = build_validator_report("L2", state.get("l2_issues", []),
                                    "Re-generate ONLY targeted L2 rows using the single reviewer suggestion.")
    notes = _rag_notes(state, "rag_notes_l2")
    decision = l2_to_reviewer(report, state["rows_l2"], notes=notes) or {}
    return {
        "l2_suggestion": (decision.get("suggestion") or "").strip(),
    }

# ============================================================================
#                        L3 NODES — effect + safety triage generation
# Same init → validate → review → regen cycle as L1/L2.
# ============================================================================
def l3_init_node(state: HazopGraphState):
    logger.info("=== L3_INIT: Adding effects to %d L2 rows ===", len(state.get("rows_l2") or []))
    merged_notes = _merge_rag_notes(
        state.get("notes") or "",
        state.get("rag_notes_l3") or ""
    )

    rows = l3_init_from_l2(state["rows_l2"], notes=merged_notes) or []
    rows = _ensure_rows_list(rows)
    _ensure_row_ids("L3", rows)

    expected = len(state.get("rows_l2") or [])
    logger.info("L3_INIT: Generated %d rows (expected %d)", len(rows), expected)
    if len(rows) != expected:
        logger.warning(
            "L3_INIT: Row count mismatch! Got %d L3 rows from %d L2 rows — some rows may be missing effects.",
            len(rows), expected,
        )
    missing_effect = sum(1 for r in rows if not str(r.get("effect", "")).strip())
    logger.debug("L3_INIT: Missing effect fields: %d", missing_effect)
    for i, r in enumerate(rows[:3]):
        logger.debug("L3_INIT: Row %d preview: %s", i, r)

    # Do not assign triage here; L3 models should supply the 'potentially_dangerous' field directly.
    return {"rows_l3": rows, "l3_repair_round": 0}

def l3_validate_node(state: HazopGraphState):
    rows = state["rows_l3"]

    # Count missing effects before validation.  Use _row_get to handle both
    # dict and Pydantic rows transparently.
    missing_pre = sum(1 for r in rows if not str(_row_get(r, "effect") or "").strip())
    logger.debug("L3_VALIDATE:IN rows=%d missing_effect=%d", len(rows), missing_pre)

    norm_rows, ok, issues = validate_l3_payload(rows)

    # Convert validated rows back into plain dicts.  This ensures state
    # consistency and prevents downstream code from seeing Pydantic models.
    norm_rows_dicts = _pydantic_to_dicts(norm_rows)

    missing_post = sum(1 for r in norm_rows_dicts if not str(r.get("effect", "")).strip())
    logger.debug("L3_VALIDATE:OUT rows=%d ok=%s missing_effect=%d", len(norm_rows_dicts), ok, missing_post)
    logger.info("L3_VALIDATE: ok=%s issues=%d", ok, len(issues))
    return {"rows_l3": norm_rows_dicts, "l3_ok": ok, "l3_issues": issues}

def l3_reviewer_node(state: HazopGraphState):
    logger.debug("L3_REVIEW IN: rows_l3=%d", len(state['rows_l3']))
    report = build_validator_report("L3", state.get("l3_issues", []),
                                    "Re-generate ONLY targeted L3 rows using the single reviewer suggestion.")
    notes = _rag_notes(state, "rag_notes_l3")
    decision = l3_to_reviewer(report, state["rows_l3"], notes=notes) or {}
    logger.debug("L3_REVIEW OUT: decision=%s", decision.get('decision'))
    return {
        "l3_suggestion": (decision.get("suggestion") or "").strip(),
    }

# ============================================================================
#                         HOLISTIC REVIEW & CASCADE REGENERATION
# After all three stages pass, a holistic reviewer checks cross-row
# consistency.  If issues are found, cascade functions regenerate the
# affected rows starting at the scope determined by the reviewer (L1, L2,
# or L3) and propagate changes downward (e.g. L1→L2→L3).
# ============================================================================
# Allow up to two rounds of holistic repairs in addition to the initial review.
# This accommodates scenarios where multiple groups of inconsistent rows need
# separate regeneration cycles (e.g., L2 repairs on different row ranges).
# In this implementation we do not perform an automatic fallback to L1.
MAX_HOLISTIC_ROUNDS = 3

def holistic_review_node(state: HazopGraphState):
    """Perform holistic consistency review on the current set of rows.

    On the initial holistic round (round 0) we review all rows.  On subsequent
    rounds we only review rows that are still considered problematic (i.e.,
    those in ``state["holistic_target_ids"]``).  This prevents the reviewer
    from finding new issues in previously accepted rows.  The set of unresolved
    row IDs is persisted across rounds to control which rows are regenerated
    in future cycles.
    """
    logger.info("=== HOL_REVIEW: Reviewing %d L3 rows (round %d) ===",
                len(state.get('rows_l3') or []), state.get("holistic_round", 0))
    logger.debug("HOL_REVIEW IN: rows_l3=%d", len(state['rows_l3']))
    # Determine which rows to review: on round 0, review all rows; otherwise
    # review only those rows whose row_id appears in the previously stored
    # holistic_target_ids.  This freezes rows that have already passed the
    # holistic check.
    all_rows: List[Dict[str, Any]] = deepcopy(state.get("rows_l3") or [])
    prev_targets = state.get("holistic_target_ids") or []
    round_no = state.get("holistic_round", 0)
    if round_no == 0 or not prev_targets:
        rows_for_review = all_rows
    else:
        unresolved_set = set(prev_targets)
        rows_for_review = [r for r in all_rows if str(r.get("row_id")) in unresolved_set]

    # Merge RAG context from all three stages, deduplicating identical lines
    all_rag_lines: List[str] = []
    for stage_key in ("rag_notes_l1", "rag_notes_l2", "rag_notes_l3"):
        text = (state.get(stage_key) or "").strip()
        if text:
            all_rag_lines.extend(text.splitlines())
    seen_rag: set[str] = set()
    unique_lines: List[str] = []
    for line in all_rag_lines:
        stripped = line.strip()
        if stripped and stripped not in seen_rag:
            unique_lines.append(stripped)
            seen_rag.add(stripped)
    merged_rag = "\n".join(unique_lines)
    notes = _merge_rag_notes(state.get("notes") or "", merged_rag)
    decision = {}
    try:
        decision = l3_pass_to_reviewer_holistic(rows_for_review, [], notes=notes) or {}
    except TypeError:
        decision = l3_pass_to_reviewer_holistic(rows_for_review, notes=notes) or {}

    logger.debug("HOL_REVIEW OUT: decision=%s", decision.get('decision'))

    last_decision = (decision.get("decision") or "").upper()
    holistic_ok = (last_decision == "OK")
    logger.info("HOL_REVIEW: decision=%s ok=%s", last_decision, holistic_ok)
    suggestion = (decision.get("suggestion") or "").strip()
    scope = (decision.get("scope") or "ALL").upper()
    issues = decision.get("issues") or []
    new_target_ids = _extract_row_ids_from_issues(issues)

    # Determine target ids for the next round.  On the first round we simply
    # use all row ids reported in ``new_target_ids``.  On subsequent rounds
    # we **do not add new IDs**: only the previously unresolved row IDs are
    # retained if they still appear in the current ``new_target_ids``.  Any
    # additional issues discovered after the initial holistic round are
    # ignored until the original set of problematic rows has been fixed.
    if round_no == 0:
        target_ids = new_target_ids
    else:
        # Keep only those ids from prev_targets that still appear in new_target_ids
        target_ids = [rid for rid in prev_targets if rid in new_target_ids]

    return {
        "holistic_ok": holistic_ok,
        "holistic_issues": issues if isinstance(issues, list) else [],
        "holistic_round": round_no,
        "holistic_suggestion": suggestion,
        "holistic_scope": scope,
        "holistic_target_ids": target_ids,
    }

# --- Cascade functions: regenerate at a given stage and propagate downward ---

def _cascade_l1(
    state: HazopGraphState,
    target_suffixes: List[str],
    suggestion: str,
) -> Dict[str, Any]:
    """Regenerate L1 rows and cascade changes to L2 and L3."""
    fix_hints = [suggestion] if suggestion else []
    l1_notes = _rag_notes(state, "rag_notes_l1")
    regenerated = reviewer_to_l1(
        state["functions"],
        state["guideword_map"],
        fix_hints,
        state.get("max_devs_per_gw", 2),
        notes=l1_notes,
    ) or []
    regenerated = _ensure_rows_list(regenerated)
    _ensure_row_ids("L1", regenerated)

    current = [_row_to_dict(r) for r in state["rows_l1"]]
    new_l1 = _merge_full_by_key(
        current, regenerated, target_suffixes,
        key_fields=["function", "guideword"],
    )

    # Cascade to L2
    l2_notes = _rag_notes(state, "rag_notes_l2")
    l2_full = l2_init_from_l1(new_l1, notes=l2_notes) or []
    l2_full = _ensure_rows_list(l2_full)
    _ensure_row_ids("L2", l2_full)
    new_l2 = _merge_full_by_key(
        [_row_to_dict(r) for r in (state["rows_l2"] or [])],
        l2_full, target_suffixes,
        key_fields=["function", "guideword"],
    )

    # Cascade to L3
    l3_notes = _rag_notes(state, "rag_notes_l3")
    l3_full = l3_init_from_l2(new_l2, notes=l3_notes) or []
    l3_full = _ensure_rows_list(l3_full)
    _ensure_row_ids("L3", l3_full)
    new_l3 = _merge_full_by_key(
        [_row_to_dict(r) for r in (state["rows_l3"] or [])],
        l3_full, target_suffixes,
        key_fields=["function", "guideword"],
    )

    return {"rows_l1": new_l1, "rows_l2": new_l2, "rows_l3": new_l3}


def _cascade_l2(
    state: HazopGraphState,
    target_suffixes: List[str],
    suggestion: str,
) -> Dict[str, Any]:
    """Regenerate L2 rows and cascade changes to L3."""
    cur_l2 = [_row_to_dict(r) for r in (state["rows_l2"] or [])]
    new_l2 = cur_l2

    l2_notes = _rag_notes(state, "rag_notes_l2")
    if target_suffixes:
        l2_subset = _pick_rows_by_suffix(cur_l2, target_suffixes)
        row_ids = [str(_row_get(r, "row_id")) for r in l2_subset]
        row_hints = [{"row_id": rid, "fields": ["cause"], "suggestion": suggestion or ""} for rid in row_ids]

        l2_new = reviewer_patch_l2(l2_subset, row_hints, notes=l2_notes) or []
        l2_new = _ensure_rows_list(l2_new)
        _ensure_row_ids("L2", l2_new)
        l2_new = _pick_rows_by_suffix(l2_new, target_suffixes)
        new_l2 = _merge_subset_by_suffix(cur_l2, l2_new)

        # Fallback if patch didn't change anything
        if not l2_new or _rows_equal_by_suffix(cur_l2, new_l2):
            l2_full = l2_init_from_l1(state["rows_l1"], notes=l2_notes) or []
            l2_full = _ensure_rows_list(l2_full)
            _ensure_row_ids("L2", l2_full)
            new_l2 = _merge_full_by_key(
                cur_l2, l2_full, target_suffixes, key_fields=["function", "guideword"]
            )

    # Cascade to L3
    l3_notes = _rag_notes(state, "rag_notes_l3")
    l3_full = l3_init_from_l2(new_l2, notes=l3_notes) or []
    l3_full = _ensure_rows_list(l3_full)
    _ensure_row_ids("L3", l3_full)
    new_l3 = _merge_full_by_key(
        [_row_to_dict(r) for r in (state["rows_l3"] or [])],
        l3_full, target_suffixes,
        key_fields=["function", "guideword"]
    )

    return {"rows_l2": new_l2, "rows_l3": new_l3}


def _cascade_l3(
    state: HazopGraphState,
    target_suffixes: List[str],
    suggestion: str,
) -> Dict[str, Any]:
    """Regenerate L3 rows only (no cascade)."""
    if not target_suffixes:
        return {}

    cur_l3 = [_row_to_dict(r) for r in (state["rows_l3"] or [])]
    l3_subset = _pick_rows_by_suffix(cur_l3, target_suffixes)
    row_ids = [str(_row_get(r, "row_id")) for r in l3_subset]
    # Include both "effect" and "potentially_dangerous" so the patcher
    # can update either field as needed.
    row_hints = [
        {"row_id": rid, "fields": ["effect", "potentially_dangerous"], "suggestion": suggestion or ""}
        for rid in row_ids
    ]

    l3_notes = _rag_notes(state, "rag_notes_l3")
    l3_new = reviewer_patch_l3(l3_subset, row_hints, notes=l3_notes) or []
    l3_new = _ensure_rows_list(l3_new)
    _ensure_row_ids("L3", l3_new)
    l3_new = _pick_rows_by_suffix(l3_new, target_suffixes)
    new_l3 = _merge_subset_by_suffix(cur_l3, l3_new)

    # Fallback if patch didn't change anything
    if not l3_new or _rows_equal_by_suffix(l3_subset, l3_new):
        l3_full = l3_init_from_l2(state["rows_l2"], notes=l3_notes) or []
        l3_full = _ensure_rows_list(l3_full)
        _ensure_row_ids("L3", l3_full)
        new_l3 = _merge_full_by_key(
            cur_l3, l3_full, target_suffixes,
            key_fields=["function", "guideword", "deviation", "cause"]
        )

    return {"rows_l3": new_l3}


# Dispatch table for cascade handlers
_CASCADE_HANDLERS = {
    "L1": _cascade_l1,
    "L2": _cascade_l2,
    "L3": _cascade_l3,
}


def _cascade_from_H_single_suggestion(
    state: HazopGraphState,
    H: Literal["L1", "L2", "L3"],
    target_suffixes: List[str],
    suggestion: str,
) -> Dict[str, Any]:
    """Dispatch cascade regeneration to the appropriate handler based on scope."""
    handler = _CASCADE_HANDLERS.get(H)
    if handler:
        return handler(state, target_suffixes, suggestion)
    return {}

def holistic_execute_node(state: HazopGraphState):
    """Regenerate only the unresolved rows identified by the holistic review.

    This node regenerates only the rows whose ``row_id`` values are present in
    ``state['holistic_target_ids']``.  Rows not in this list are preserved
    unchanged (i.e., frozen) across holistic rounds.  After regeneration the
    holistic round counter is incremented.  If the holistic review already
    reported success, no further work is done.
    """
    logger.debug("HOL_EXECUTE IN: rows_l3=%d", len(state['rows_l3']))
    # If previous holistic review reported OK, skip regeneration.
    if state.get("holistic_ok", False):
        return {"holistic_round": state.get("holistic_round", 0)}

    suggestion = state.get("holistic_suggestion", "")
    scope = state.get("holistic_scope", "ALL")
    H: Literal["L1","L2","L3"] = _scope_to_highest_stage(scope)

    # Regenerate only unresolved rows.  If none are marked (which should not
    # happen when holistic_ok is False), fall back to regenerating all rows.
    target_ids = state.get("holistic_target_ids") or []
    if not target_ids:
        target_ids = [str(r.get("row_id")) for r in (state.get("rows_l3") or []) if r.get("row_id")]
    target_suffixes = _suffixes_from_row_ids(target_ids)

    logger.debug("HOL_EXECUTE: H=%s targets=%s", H, target_suffixes)

    # Always cascade regeneration only on the selected suffixes at the chosen stage.
    # We do not perform an automatic fallback to L1; stage escalation is determined by the reviewer's scope.
    cascade_result = _cascade_from_H_single_suggestion(state, H, target_suffixes, suggestion)

    logger.debug("HOL_EXECUTE OUT: rows_l3=%d",
                 len(cascade_result.get("rows_l3", state.get("rows_l3") or [])))

    return {
        "rows_l1": cascade_result.get("rows_l1", state.get("rows_l1")),
        "rows_l2": cascade_result.get("rows_l2", state.get("rows_l2")),
        "rows_l3": cascade_result.get("rows_l3", state.get("rows_l3")),
        "holistic_round": state.get("holistic_round", 0) + 1,
    }

# ============================================================================
#                            GRAPH DEFINITION
# Wires all nodes and conditional edges into the LangGraph state machine.
# Conditional edges after each VALIDATE node decide whether to proceed to the
# next stage or loop back through REVIEW → REGEN for another repair attempt.
# ============================================================================
def build_graph() -> StateGraph:
    g = StateGraph(HazopGraphState)

    # RAG
    g.add_node("RAG_PREP", rag_prep_node)
    g.add_node("RAG_CTX_L1", rag_ctx_l1_node)
    g.add_node("RAG_CTX_L2", rag_ctx_l2_node)
    g.add_node("RAG_CTX_L3", rag_ctx_l3_node)

    # L1
    g.add_node("L1_INIT", l1_init_node)
    g.add_node("L1_VALIDATE", l1_validate_node)
    g.add_node("L1_REVIEW", l1_reviewer_node)
    g.add_node("L1_REGEN", l1_regen_node)

    # L2
    g.add_node("L2_INIT", l2_init_node)
    g.add_node("L2_VALIDATE", l2_validate_node)
    g.add_node("L2_REVIEW", l2_reviewer_node)
    g.add_node("L2_REGEN", l2_regen_node)

    # L3
    g.add_node("L3_INIT", l3_init_node)
    g.add_node("L3_VALIDATE", l3_validate_node)
    g.add_node("L3_REVIEW", l3_reviewer_node)
    g.add_node("L3_REGEN", l3_regen_node)

    # Holistic
    g.add_node("HOL_REVIEW", holistic_review_node)
    g.add_node("HOL_EXECUTE", holistic_execute_node)

    # START
    g.add_edge(START, "RAG_PREP")
    g.add_edge("RAG_PREP", "RAG_CTX_L1")
    g.add_edge("RAG_CTX_L1", "L1_INIT")

    # L1 edges
    g.add_edge("L1_INIT", "L1_VALIDATE")
    def l1_after_validate(state: HazopGraphState) -> str:
        if state.get("l1_ok"):
            return "L2_INIT"
        if state.get("l1_repair_round", 0) >= MAX_STAGE_REPAIR:
            return "L2_INIT"
        return "L1_REVIEW"
    g.add_conditional_edges("L1_VALIDATE", l1_after_validate,
                            {"L2_INIT": "RAG_CTX_L2", "L1_REVIEW": "L1_REVIEW"})
    g.add_edge("RAG_CTX_L2", "L2_INIT")
    g.add_edge("L1_REVIEW", "L1_REGEN")
    g.add_edge("L1_REGEN", "L1_VALIDATE")

    # L2 edges
    g.add_edge("L2_INIT", "L2_VALIDATE")
    def l2_after_validate(state: HazopGraphState) -> str:
        if state.get("l2_ok"):
            return "L3_INIT"
        if state.get("l2_repair_round", 0) >= MAX_STAGE_REPAIR:
            return "L3_INIT"
        return "L2_REVIEW"
    g.add_conditional_edges("L2_VALIDATE", l2_after_validate,
                            {"L3_INIT": "RAG_CTX_L3", "L2_REVIEW": "L2_REVIEW"})
    g.add_edge("RAG_CTX_L3", "L3_INIT")
    g.add_edge("L2_REVIEW", "L2_REGEN")
    g.add_edge("L2_REGEN", "L2_VALIDATE")

    # L3 edges
    g.add_edge("L3_INIT", "L3_VALIDATE")
    def l3_after_validate(state: HazopGraphState) -> str:
        if state.get("l3_ok"):
            return "HOL_REVIEW"
        if state.get("l3_repair_round", 0) >= MAX_STAGE_REPAIR:
            return "HOL_REVIEW"
        return "L3_REVIEW"
    g.add_conditional_edges("L3_VALIDATE", l3_after_validate,
                            {"HOL_REVIEW": "HOL_REVIEW", "L3_REVIEW": "L3_REVIEW"})
    g.add_edge("L3_REVIEW", "L3_REGEN")
    g.add_edge("L3_REGEN", "L3_VALIDATE")

    # Holistic edges
    def hol_after_review(state: HazopGraphState):
        if state.get("holistic_ok", False):
            return END
        if state.get("holistic_round", 0) >= MAX_HOLISTIC_ROUNDS:
            return END
        return "HOL_EXECUTE"
    g.add_conditional_edges("HOL_REVIEW", hol_after_review,
                            {"HOL_EXECUTE": "HOL_EXECUTE", END: END})
    g.add_edge("HOL_EXECUTE", "HOL_REVIEW")

    return g

def build_full_graph() -> StateGraph:
    return build_graph()

if __name__ == "__main__":
    print('Use: python -m src.run_pipeline src/functions.txt --notes "ABS demo" --outdir out --max_devs_per_gw 2')
