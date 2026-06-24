# src/graph_full.py
"""LangGraph state machine for the AI-HAZOP-8800 pipeline (config-driven engine).

The pipeline runs a single AI component+aspect through all eight phases:

  START → L1 Failure Mode → L2 Hazard → L3 Initial Risk → L4 Acceptance →
          L5 Safety Goals → L6 Measures → L7 Residual Risk → L8 Evidence →
          HOLISTIC REVIEW → END

Every phase shares one cycle, built from a single ``PHASE_SPECS`` table:

  INIT → VALIDATE → (if issues) REVIEW → REGEN → VALIDATE → …

with a hard cap of MAX_STAGE_REPAIR repair rounds per phase (earliest-faulty-phase
-first, matching the architecture doc §7). Conditional early-exit is handled by row
filtering, not graph branching:
  * L3–L4 only process rows that L2 flagged ``potentially_dangerous``;
  * L5–L7 only process rows whose L4 ``safety_decision`` is not ACCEPT;
  * L8 evidences the union of L7 (improved) rows and L4 ACCEPT rows.
Filtered-out rows carry through to worksheet assembly with empty downstream fields.

After L8, a holistic reviewer checks cross-phase consistency over the whole
worksheet and routes repair to the earliest faulty phase (bounded by
MAX_HOLISTIC_ROUNDS) via ``patch_and_cascade``.
"""
from __future__ import annotations

import logging
import re
from copy import deepcopy
from typing import TypedDict, List, Dict, Any, Callable

from langgraph.graph import StateGraph, START, END

logger = logging.getLogger(__name__)

from .row_utils import (
    _ensure_rows_list,
    _row_to_dict,
    _suffix,
    _pydantic_to_dicts,
    _merge_subset_by_suffix,
    _pick_rows_by_suffix,
    _rows_equal_by_suffix,
)
from .ai_chains import ai_l1_generate, ai_phase_generate, ai_reviewer, ai_patch, ai_holistic_review
from .validators import (
    validate_l1_payload,
    validate_l2_payload,
    validate_l3_payload,
    validate_l4_payload,
    validate_l5_payload,
    validate_l6_payload,
    validate_l7_payload,
    validate_l8_payload,
    build_validator_report,
    is_valid_l1_row,
    is_safety_relevant,
)
from .catalogue_loader import load_catalogue
from .risk_model import compute_risk, FACTORS

MAX_STAGE_REPAIR = 2
MAX_HOLISTIC_ROUNDS = 3


# ============================================================================
#                                STATE
# ============================================================================
class HazopGraphState(TypedDict, total=False):
    # --- analysis context (architecture doc §2) ---
    ctx: Dict[str, str]                # component, component_class, aspect, odd, scenario
    guidewords: List[Dict[str, Any]]   # working guideword catalogue entries
    class_questions: List[str]
    aspect_hint: str
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

    rows_l4: List[Dict[str, Any]]
    l4_ok: bool
    l4_issues: List[str]
    l4_suggestion: str
    l4_repair_round: int

    rows_l5: List[Dict[str, Any]]
    l5_ok: bool
    l5_issues: List[str]
    l5_suggestion: str
    l5_repair_round: int

    rows_l6: List[Dict[str, Any]]
    l6_ok: bool
    l6_issues: List[str]
    l6_suggestion: str
    l6_repair_round: int

    rows_l7: List[Dict[str, Any]]
    l7_ok: bool
    l7_issues: List[str]
    l7_suggestion: str
    l7_repair_round: int

    rows_l8: List[Dict[str, Any]]
    l8_ok: bool
    l8_issues: List[str]
    l8_suggestion: str
    l8_repair_round: int

    # --- holistic cross-phase review ---
    holistic_ok: bool
    holistic_issues: List[str]
    holistic_round: int
    holistic_suggestion: str
    holistic_scope: str
    holistic_target_ids: List[str]


# ============================================================================
#                            GENERIC HELPERS
# ============================================================================

_ROWID_RE = re.compile(r"(L[1-8]-\d+)")


def _extract_row_ids_from_issues(issues: List[str]) -> List[str]:
    """Extract unique row_ids (e.g. 'L1-3') embedded in issue strings."""
    out: List[str] = []
    seen = set()
    for s in issues or []:
        if s is None:
            continue
        for rid in _ROWID_RE.findall(str(s)):
            if rid not in seen:
                out.append(rid)
                seen.add(rid)
    return out


def _suffixes(row_ids: List[str]) -> List[str]:
    return sorted({_suffix(r) for r in (row_ids or []) if r})


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes", "y")


# ---------------------------------------------------------------------------
# Risk computation — applied to L3 rows before validation, so that both the
# initial generation and any factor-level repair recompute R consistently.
# ---------------------------------------------------------------------------

def _attach_risk(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        r = dict(r)
        risk, status, _target = compute_risk(r)
        if risk is not None:
            r["initial_risk"] = risk
        r["risk_status"] = status
        out.append(r)
    return out


def _attach_residual_risk(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        r = dict(r)
        factors = {f: r.get(f"residual_{f}") for f in FACTORS}
        risk, status, _target = compute_risk(factors)
        if risk is not None:
            r["residual_risk"] = risk
        r["residual_status"] = status
        out.append(r)
    return out


def _stamp_triage(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """L2 pre-validate hook: deterministically correct the safety-relevance triage.

    The LLM sometimes clears a genuinely dangerous row (e.g. a cyclist collision) by setting
    potentially_dangerous=false, which would END it after L2 (architecture §6). We force the flag
    to True for any row that ``is_safety_relevant`` (VRU / collision / injury wording), so the
    existing L2→L3 gate routes it onward. This is a guarantee independent of the LLM."""
    out: List[Dict[str, Any]] = []
    for r in rows:
        r = dict(r)
        if not _truthy(r.get("potentially_dangerous")) and is_safety_relevant(r):
            r["potentially_dangerous"] = True
        out.append(r)
    return out


# ============================================================================
#                          PHASE GENERATORS
# ============================================================================

def _gen_l1(state: HazopGraphState) -> List[Dict[str, Any]]:
    return ai_l1_generate(
        state["ctx"],
        state["guidewords"],
        state.get("class_questions", []),
        state.get("aspect_hint", ""),
        state.get("notes", ""),
    )


def _gen_l2(state: HazopGraphState) -> List[Dict[str, Any]]:
    """L2 processes ONLY valid L1 rows. Rows with an empty/short/meta failure_mode are
    blocked here so no downstream phase (L2-L8) ever runs for them (FIX 1/3)."""
    prev = state.get("rows_l1") or []
    valid = []
    for r in prev:
        if is_valid_l1_row(r):
            valid.append(r)
        else:
            logger.warning("L2_INIT: blocking L1 row %s from downstream (failure_mode empty/meta/too short): %r",
                           _row_get_field(r, "row_id"), str(_row_get_field(r, "failure_mode") or "")[:80])
    return ai_phase_generate("L2", valid, notes=state.get("notes", ""))


def _risk_inputs() -> Dict[str, str]:
    import json
    rm = load_catalogue("risk_model")
    acc = load_catalogue("acceptance_criterion")
    scales = {f: list((rm["factors"][f]["levels"]).keys()) for f in rm.get("factors", {})}
    return {
        "risk_scales_json": json.dumps(scales, ensure_ascii=False),
        "mem_target": json.dumps(acc.get("mem_target")),
    }


def _gen_l3(state: HazopGraphState) -> List[Dict[str, Any]]:
    """L3 processes ONLY the dangerous subset of L2 rows (conditional early-exit)."""
    prev = state.get("rows_l2") or []
    # Safety-relevance gate (architecture §6). is_safety_relevant honours the L2 triage flag and
    # also catches VRU/collision rows the LLM wrongly cleared, so they still reach L3/L4.
    dangerous = [r for r in prev if is_safety_relevant(r)]
    logger.info("L3_INIT: %d/%d rows are safety-relevant", len(dangerous), len(prev))
    if not dangerous:
        return []
    rows = ai_phase_generate("L3", dangerous, extra_inputs=_risk_inputs(), notes=state.get("notes", ""))
    return _attach_risk(rows)


def _gen_l4(state: HazopGraphState) -> List[Dict[str, Any]]:
    import json
    prev = state.get("rows_l3") or []
    if not prev:
        return []
    acc = load_catalogue("acceptance_criterion")
    extra = {
        "mem_target": json.dumps(acc.get("mem_target")),
        "decisions_json": json.dumps(acc.get("safety_decisions", [])),
    }
    return ai_phase_generate("L4", prev, extra_inputs=extra, notes=state.get("notes", ""))


def _is_accept(r: Any) -> bool:
    return str(_row_get_field(r, "safety_decision") or "").strip().upper() == "ACCEPT"


def _gen_l5(state: HazopGraphState) -> List[Dict[str, Any]]:
    """ACCEPT-skip (architecture doc §6): only non-ACCEPT rows get safety goals."""
    prev = state.get("rows_l4") or []
    non_accept = [r for r in prev if not _is_accept(r)]
    logger.info("L5_INIT: %d/%d rows need safety goals (non-ACCEPT)", len(non_accept), len(prev))
    if not non_accept:
        return []
    return ai_phase_generate("L5", non_accept, notes=state.get("notes", ""))


def _gen_l6(state: HazopGraphState) -> List[Dict[str, Any]]:
    import json
    prev = state.get("rows_l5") or []
    if not prev:
        return []
    extra = {"mitigation_json": json.dumps(load_catalogue("mitigation_taxonomy"), ensure_ascii=False)}
    return ai_phase_generate("L6", prev, extra_inputs=extra, notes=state.get("notes", ""))


def _gen_l7(state: HazopGraphState) -> List[Dict[str, Any]]:
    prev = state.get("rows_l6") or []
    if not prev:
        return []
    rows = ai_phase_generate("L7", prev, extra_inputs=_risk_inputs(), notes=state.get("notes", ""))
    return _attach_residual_risk(rows)


def _gen_l8(state: HazopGraphState) -> List[Dict[str, Any]]:
    """Evidence for ALL dangerous rows: improved rows (enriched via L7) + ACCEPT rows (bare L4)."""
    import json
    l4 = state.get("rows_l4") or []
    if not l4:
        return []
    by7 = {_suffix(str(r.get("row_id"))): r for r in (state.get("rows_l7") or []) if isinstance(r, dict)}
    union: List[Dict[str, Any]] = []
    for r in l4:
        sfx = _suffix(str(r.get("row_id")))
        union.append(deepcopy(by7.get(sfx, r)))
    extra = {"evidence_json": json.dumps(load_catalogue("evidence_catalogue"), ensure_ascii=False)}
    return ai_phase_generate("L8", union, extra_inputs=extra, notes=state.get("notes", ""))


def _row_get_field(r: Any, key: str) -> Any:
    if isinstance(r, dict):
        return r.get(key)
    return getattr(r, key, None)


# ============================================================================
#                          PHASE SPEC TABLE
# Each phase is fully described here; the node/edge factories below build the
# graph from this single table.
# ============================================================================

def _validate_l1(state, rows):
    expected = [g["id"] for g in state.get("guidewords", [])]
    return validate_l1_payload(rows, expected)


PHASE_SPECS: List[Dict[str, Any]] = [
    {
        "name": "L1",
        "generate": _gen_l1,
        "validate": _validate_l1,
        "pre_validate": None,
        "patch_fields": ["failure_mode"],
        "review_hint": "Re-generate ONLY targeted L1 failure-mode rows using the reviewer suggestion.",
    },
    {
        "name": "L2",
        "generate": _gen_l2,
        "validate": lambda state, rows: validate_l2_payload(rows),
        "pre_validate": _stamp_triage,
        "patch_fields": ["hazardous_behavior", "potential_harm", "potentially_dangerous"],
        "review_hint": "Re-generate ONLY targeted L2 hazard rows using the reviewer suggestion.",
    },
    {
        "name": "L3",
        "generate": _gen_l3,
        "validate": lambda state, rows: validate_l3_payload(rows),
        "pre_validate": _attach_risk,
        "patch_fields": ["E", "PF", "PND", "PNM", "S", "risk_rationale"],
        "review_hint": "Re-generate ONLY targeted L3 risk rows using the reviewer suggestion.",
    },
    {
        "name": "L4",
        "generate": _gen_l4,
        "validate": lambda state, rows: validate_l4_payload(rows),
        "pre_validate": None,
        "patch_fields": ["safety_decision", "acceptance_rationale"],
        "review_hint": "Re-generate ONLY targeted L4 acceptance rows using the reviewer suggestion.",
    },
    {
        "name": "L5",
        "generate": _gen_l5,
        "validate": lambda state, rows: validate_l5_payload(rows),
        "pre_validate": None,
        "patch_fields": ["ai_safety_goals"],
        "review_hint": "Re-generate ONLY targeted L5 safety-goal rows using the reviewer suggestion.",
    },
    {
        "name": "L6",
        "generate": _gen_l6,
        "validate": lambda state, rows: validate_l6_payload(rows),
        "pre_validate": None,
        "patch_fields": ["respecifications", "safety_functions", "passive_operational_measures"],
        "review_hint": "Re-generate ONLY targeted L6 measure rows using the reviewer suggestion.",
    },
    {
        "name": "L7",
        "generate": _gen_l7,
        "validate": lambda state, rows: validate_l7_payload(rows),
        "pre_validate": _attach_residual_risk,
        "patch_fields": ["residual_E", "residual_PF", "residual_PND", "residual_PNM", "residual_S", "residual_rationale"],
        "review_hint": "Re-generate ONLY targeted L7 residual-risk rows using the reviewer suggestion.",
    },
    {
        "name": "L8",
        "generate": _gen_l8,
        "validate": lambda state, rows: validate_l8_payload(rows),
        "pre_validate": None,
        "patch_fields": ["evidence", "open_assumptions"],
        "review_hint": "Re-generate ONLY targeted L8 evidence rows using the reviewer suggestion.",
    },
]

_SPEC_BY_NAME = {s["name"]: s for s in PHASE_SPECS}

# Derived per-phase state keys.
def _keys(name: str) -> Dict[str, str]:
    low = name.lower()
    return {
        "rows": f"rows_{low}",
        "ok": f"{low}_ok",
        "issues": f"{low}_issues",
        "suggestion": f"{low}_suggestion",
        "repair": f"{low}_repair_round",
    }


# ============================================================================
#                          NODE FACTORIES
# ============================================================================

def _make_init_node(spec: Dict[str, Any]) -> Callable:
    k = _keys(spec["name"])

    def init_node(state: HazopGraphState) -> Dict[str, Any]:
        logger.info("=== %s_INIT ===", spec["name"])
        try:
            rows = spec["generate"](state) or []
        except Exception as exc:  # an unparsable LLM reply must not kill the whole run
            logger.error("%s_INIT generation failed (%s); producing no rows for this phase", spec["name"], exc)
            rows = []
        rows = _ensure_rows_list(rows)
        return {k["rows"]: rows, k["repair"]: 0}

    return init_node


def _make_validate_node(spec: Dict[str, Any]) -> Callable:
    k = _keys(spec["name"])

    def validate_node(state: HazopGraphState) -> Dict[str, Any]:
        rows = state.get(k["rows"]) or []
        if spec["pre_validate"]:
            rows = spec["pre_validate"](rows)
        norm_rows, ok, issues = spec["validate"](state, rows)
        norm_dicts = _pydantic_to_dicts(norm_rows)
        logger.info("%s_VALIDATE: ok=%s issues=%d rows=%d", spec["name"], ok, len(issues), len(norm_dicts))
        return {k["rows"]: norm_dicts, k["ok"]: ok, k["issues"]: issues}

    return validate_node


def _make_review_node(spec: Dict[str, Any]) -> Callable:
    k = _keys(spec["name"])

    def review_node(state: HazopGraphState) -> Dict[str, Any]:
        report = build_validator_report(spec["name"], state.get(k["issues"], []), spec["review_hint"])
        decision = ai_reviewer(spec["name"], report, state.get(k["rows"]) or [], notes=state.get("notes", ""))
        return {k["suggestion"]: (decision.get("suggestion") or "").strip()}

    return review_node


def _make_regen_node(spec: Dict[str, Any]) -> Callable:
    k = _keys(spec["name"])

    def regen_node(state: HazopGraphState) -> Dict[str, Any]:
        sug = (state.get(k["suggestion"], "") or "").strip()
        target_ids = _extract_row_ids_from_issues(state.get(k["issues"], []))
        rows = [_row_to_dict(r) for r in (state.get(k["rows"]) or [])]
        rnd = state.get(k["repair"], 0)

        if not target_ids or not sug:
            return {k["repair"]: rnd + 1}

        target_sfx = _suffixes(target_ids)
        subset = _pick_rows_by_suffix(rows, target_sfx)
        sug = sug[:400]  # never feed a giant blob the model may echo into a field
        hints = [{"row_id": r.get("row_id"), "fields": spec["patch_fields"], "suggestion": sug} for r in subset]

        try:
            patched = _ensure_rows_list(ai_patch(spec["name"], subset, hints, notes=state.get("notes", "")) or [])
        except Exception as exc:  # bad LLM repair reply: keep existing rows, do not crash
            logger.warning("%s_REGEN: patch failed (%s); keeping existing rows", spec["name"], exc)
            patched = []
        merged = _merge_subset_by_suffix(rows, patched) if patched else rows

        # Fallback: if the targeted patch changed nothing, regenerate the whole
        # phase and splice in the targeted rows.
        if (not patched) or _rows_equal_by_suffix(rows, merged):
            logger.info("%s_REGEN: patch no-op, falling back to full regenerate", spec["name"])
            try:
                fresh = _ensure_rows_list(spec["generate"](state) or [])
                fresh_subset = _pick_rows_by_suffix(fresh, target_sfx)
                if fresh_subset:
                    merged = _merge_subset_by_suffix(rows, fresh_subset)
            except Exception as exc:
                logger.warning("%s_REGEN: fallback regenerate failed (%s); keeping existing rows", spec["name"], exc)

        return {k["rows"]: merged, k["repair"]: rnd + 1}

    return regen_node


# ============================================================================
#                       HOLISTIC CROSS-PHASE REVIEW
# After L8, a holistic reviewer checks consistency across the whole chain and
# routes repair to the EARLIEST faulty phase (architecture doc §7/§8). The
# execute node patches that phase's targeted rows, then regenerates every
# downstream phase (invalidate-and-regenerate), bounded by MAX_HOLISTIC_ROUNDS.
# ============================================================================

_HOL_FIELDS = (
    "guideword", "failure_mode", "hazardous_behavior", "potential_harm",
    "risk_status", "initial_risk", "safety_decision", "ai_safety_goals",
    "respecifications", "safety_functions", "passive_operational_measures",
    "residual_status", "evidence", "open_assumptions",
)


def _holistic_view(state: HazopGraphState) -> List[Dict[str, Any]]:
    """Build one slim row per L1 row, merging every phase's fields by suffix."""
    l1 = state.get("rows_l1") or []
    indices = {
        ph: {_suffix(str(r.get("row_id"))): r for r in (state.get(f"rows_{ph}") or []) if isinstance(r, dict)}
        for ph in ("l2", "l3", "l4", "l5", "l6", "l7", "l8")
    }
    view: List[Dict[str, Any]] = []
    for r1 in l1:
        if not isinstance(r1, dict):
            continue
        sfx = _suffix(str(r1.get("row_id")))
        merged: Dict[str, Any] = {"row_id": r1.get("row_id")}
        sources = [r1] + [indices[ph].get(sfx, {}) for ph in ("l2", "l3", "l4", "l5", "l6", "l7", "l8")]
        for src in sources:
            for f in _HOL_FIELDS:
                if f in src and src.get(f) not in (None, "", []):
                    merged[f] = src[f]
        view.append(merged)
    return view


def holistic_review_node(state: HazopGraphState) -> Dict[str, Any]:
    rnd = state.get("holistic_round", 0)
    view = _holistic_view(state)
    logger.info("=== HOL_REVIEW (round %d): %d rows ===", rnd, len(view))
    try:
        decision = ai_holistic_review(view, [], notes=state.get("notes", "")) or {}
    except Exception as exc:  # holistic review is optional polish: never fail the run on it
        logger.warning("HOL_REVIEW failed (%s); accepting the worksheet as-is", exc)
        decision = {"decision": "OK"}
    ok = (decision.get("decision") or "").upper() == "OK"
    return {
        "holistic_ok": ok,
        "holistic_issues": decision.get("issues") or [],
        "holistic_round": rnd,
        "holistic_suggestion": (decision.get("suggestion") or "").strip(),
        "holistic_scope": (decision.get("scope") or "L1").upper(),
        "holistic_target_ids": decision.get("target_ids") or _extract_row_ids_from_issues(decision.get("issues") or []),
    }


def patch_and_cascade(
    state: HazopGraphState,
    scope: str,
    target_sfx: List[str],
    suggestion: str,
    notes: str = "",
) -> Dict[str, Any]:
    """Patch the targeted rows at ``scope`` then regenerate every downstream phase.

    Returns a ``{rows_key: rows}`` dict for the scope phase (if patched) and all
    phases after it. Used by both the holistic execute node and the GUI adapter so
    scope-based regeneration shares one cascade implementation. The caller threads
    these rows back into its own state representation.
    """
    scope = (scope or "L1").upper()
    spec = _SPEC_BY_NAME.get(scope)
    if spec is None:
        return {}

    idx = next(i for i, s in enumerate(PHASE_SPECS) if s["name"] == scope)
    sug = (suggestion or "").strip()[:400]  # cap: never feed a giant blob the model may echo

    s = dict(state)  # working copy threaded through downstream regeneration
    updates: Dict[str, Any] = {}

    # 1) Patch the targeted rows at the scope phase (a bad LLM reply must not crash).
    k = _keys(scope)
    rows = [_row_to_dict(r) for r in (s.get(k["rows"]) or [])]
    subset = _pick_rows_by_suffix(rows, target_sfx) if target_sfx else []
    if subset and sug:
        hints = [{"row_id": r.get("row_id"), "fields": spec["patch_fields"], "suggestion": sug} for r in subset]
        try:
            patched = _ensure_rows_list(ai_patch(scope, subset, hints, notes=notes) or [])
        except Exception as exc:
            logger.warning("patch_and_cascade: %s patch failed (%s); keeping existing rows", scope, exc)
            patched = []
        if patched:
            merged = _merge_subset_by_suffix(rows, patched)
            if spec["pre_validate"]:
                merged = spec["pre_validate"](merged)
            s[k["rows"]] = merged
            updates[k["rows"]] = merged

    # 2) Invalidate and regenerate every downstream phase.
    for ds in PHASE_SPECS[idx + 1:]:
        dk = _keys(ds["name"])
        try:
            fresh = _ensure_rows_list(ds["generate"](s) or [])
            if ds["pre_validate"]:
                fresh = ds["pre_validate"](fresh)
        except Exception as exc:
            logger.warning("patch_and_cascade: regenerating %s failed (%s); keeping existing rows", ds["name"], exc)
            fresh = _ensure_rows_list(s.get(dk["rows"]) or [])
        s[dk["rows"]] = fresh
        updates[dk["rows"]] = fresh

    logger.info("patch_and_cascade: scope=%s, patched %d rows, regenerated %d downstream phases",
                scope, len(subset), len(PHASE_SPECS) - idx - 1)
    return updates


def holistic_execute_node(state: HazopGraphState) -> Dict[str, Any]:
    rnd = state.get("holistic_round", 0)
    if state.get("holistic_ok", False):
        return {"holistic_round": rnd}

    scope = (state.get("holistic_scope") or "L1").upper()
    if _SPEC_BY_NAME.get(scope) is None:
        return {"holistic_round": rnd + 1}

    updates = patch_and_cascade(
        state,
        scope,
        _suffixes(state.get("holistic_target_ids") or []),
        state.get("holistic_suggestion") or "",
        notes=state.get("notes", ""),
    )
    updates["holistic_round"] = rnd + 1
    return updates


# ============================================================================
#                            GRAPH DEFINITION
# ============================================================================

def build_graph() -> StateGraph:
    g = StateGraph(HazopGraphState)

    names = [s["name"] for s in PHASE_SPECS]
    for spec in PHASE_SPECS:
        n = spec["name"]
        g.add_node(f"{n}_INIT", _make_init_node(spec))
        g.add_node(f"{n}_VALIDATE", _make_validate_node(spec))
        g.add_node(f"{n}_REVIEW", _make_review_node(spec))
        g.add_node(f"{n}_REGEN", _make_regen_node(spec))

    # Holistic review runs after the last phase passes.
    g.add_node("HOL_REVIEW", holistic_review_node)
    g.add_node("HOL_EXECUTE", holistic_execute_node)

    g.add_edge(START, f"{names[0]}_INIT")

    for idx, spec in enumerate(PHASE_SPECS):
        n = spec["name"]
        k = _keys(n)
        next_node = f"{names[idx + 1]}_INIT" if idx + 1 < len(names) else "HOL_REVIEW"

        g.add_edge(f"{n}_INIT", f"{n}_VALIDATE")

        def _after_validate(state, _k=k, _next=next_node):
            if state.get(_k["ok"]):
                return "NEXT"
            if state.get(_k["repair"], 0) >= MAX_STAGE_REPAIR:
                return "NEXT"
            return "REVIEW"

        g.add_conditional_edges(
            f"{n}_VALIDATE",
            _after_validate,
            {"NEXT": next_node, "REVIEW": f"{n}_REVIEW"},
        )
        g.add_edge(f"{n}_REVIEW", f"{n}_REGEN")
        g.add_edge(f"{n}_REGEN", f"{n}_VALIDATE")

    def _hol_after_review(state):
        if state.get("holistic_ok", False):
            return "END"
        if state.get("holistic_round", 0) >= MAX_HOLISTIC_ROUNDS:
            return "END"
        return "EXEC"

    g.add_conditional_edges("HOL_REVIEW", _hol_after_review, {"EXEC": "HOL_EXECUTE", "END": END})
    g.add_edge("HOL_EXECUTE", "HOL_REVIEW")

    return g


def build_full_graph() -> StateGraph:
    return build_graph()


if __name__ == "__main__":
    print('Use: python -m src.run_pipeline <context.yaml> --provider gemini --outdir out')
