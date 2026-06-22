# src/chains.py
"""LLM chain functions for generation, review, and patching across all three stages.

This module provides the LLM-facing logic:
  - Generation chains (l1_init, l2_init, l3_init): produce rows for each stage
  - Reviewer chains (_to_reviewer, holistic): evaluate quality and suggest fixes
  - Patch chains (_patch_rows, reviewer_patch_*): targeted repair of specific rows
  - Batch verification (_verify_and_retry_batch): catches dropped rows and retries
"""
import json
import logging
from typing import Any, Dict, List

from .prompt_loader import load_prompt, render_payload
from .llm_client import chat_json, MODEL_CHEAP, MODEL_REVIEW
from .row_utils import _ensure_rows_list, _ensure_list_of_dicts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Batching constants — LLM output can be truncated on large inputs, so we
# split work into batches.  L1 batches by function count (each function
# generates 11 rows for the 11 guidewords); L2/L3/patch batch by row count.
# ---------------------------------------------------------------------------
L1_BATCH_SIZE = 3    # Functions per batch (33 rows max per batch)
ROW_BATCH_SIZE = 20  # Rows per batch for L2/L3/patch operations
_REVIEWER_FUNCS_PER_CHUNK = 3  # Functions per chunk for stage reviewers (3 × 11 gw = 33 rows/chunk)


def _to_json(obj: Any) -> str:
    """Convert object to JSON string with consistent formatting."""
    return json.dumps(obj, ensure_ascii=False, default=str)


def _build_prompt(p: dict, payload: str, instr: str = None, extra: str = "") -> str:
    """Assemble instructions + constraints + few-shot examples + extras + payload."""
    parts = [instr or p["instructions"]]
    if p.get("constraints"):
        parts.append(p["constraints"])
    if p.get("few_shot_examples"):
        ex_lines = []
        for ex in p["few_shot_examples"]:
            ex_lines.append(f"[{ex['role'].upper()}]\n{ex['content'].strip()}")
        parts.append("EXAMPLES:\n" + "\n\n".join(ex_lines))
    if extra:
        parts.append(extra)
    parts.append(payload)
    return "\n\n".join(parts)


# =========================
# Batch verification & retry helper — LLMs sometimes drop rows from a batch
# (especially when the batch is large).  This helper matches input→output by
# row_id (with composite-key fallback) and retries each missing row
# individually.  Two individual attempts per missing row, then log and skip.
# =========================

def _verify_and_retry_batch(
    input_rows: List[Dict[str, Any]],
    output_rows: List[Dict[str, Any]],
    retry_fn,
    stage: str,
) -> List[Dict[str, Any]]:
    """Verify that every input row has a corresponding output row; retry missing ones.

    Matching is done primarily by ``row_id``, with a fallback composite key of
    ``(function, guideword, deviation)`` for robustness when the LLM drops or
    renames ``row_id`` values.

    For each missing row the *retry_fn* is called with a **single-element list**
    containing that input row.  *retry_fn(rows) -> List[Dict]* must accept a
    list of input rows and return a list of output rows.

    If a single-row retry also fails, one more attempt is made.  If that also
    fails the row is logged and skipped (better to have partial data than crash
    the pipeline).

    Args:
        input_rows: The rows that were sent to the LLM.
        output_rows: The rows the LLM returned.
        retry_fn: Callable ``(List[Dict]) -> List[Dict]`` that generates output
                  for the given input rows.
        stage: Label for logging (e.g. ``"L2_INIT"``).

    Returns:
        A list of output rows guaranteed to cover every input row (order may
        differ from input).
    """
    if not input_rows:
        return output_rows or []

    # Build lookup of output rows by row_id
    out_by_id: Dict[str, Dict[str, Any]] = {}
    for r in (output_rows or []):
        rid = str(r.get("row_id", "")).strip()
        if rid:
            out_by_id[rid] = r

    # Build composite-key lookup as fallback
    def _composite(r: Dict[str, Any]) -> str:
        return "||".join(
            str(r.get(k, "")).strip().lower()
            for k in ("function", "guideword", "deviation")
        )

    out_by_composite: Dict[str, Dict[str, Any]] = {}
    for r in (output_rows or []):
        ck = _composite(r)
        if ck and ck != "||||":
            out_by_composite[ck] = r

    # Build positional result array — keeps output aligned with input order
    result: List[Any] = [None] * len(input_rows)
    missing_indices: List[int] = []

    for idx, inp in enumerate(input_rows):
        rid = str(inp.get("row_id", "")).strip()
        if rid and rid in out_by_id:
            result[idx] = out_by_id[rid]
        elif _composite(inp) in out_by_composite:
            result[idx] = out_by_composite[_composite(inp)]
        else:
            missing_indices.append(idx)

    if not missing_indices:
        # All accounted for – return positionally ordered result
        return [r for r in result if r is not None]

    logger.warning(
        "[%s] Batch returned %d/%d rows – retrying %d missing individually",
        stage, len(input_rows) - len(missing_indices), len(input_rows), len(missing_indices),
    )

    MAX_INDIVIDUAL_ATTEMPTS = 2
    for idx in missing_indices:
        inp_row = input_rows[idx]
        recovered = None
        for attempt in range(1, MAX_INDIVIDUAL_ATTEMPTS + 1):
            try:
                ret = retry_fn([inp_row])
                if ret:
                    recovered = ret[0] if isinstance(ret, list) and ret else ret
                    break
            except Exception as exc:
                logger.warning(
                    "[%s] Individual retry attempt %d failed for row_id=%s: %s",
                    stage, attempt, inp_row.get("row_id", "?"), exc,
                )
        if recovered and isinstance(recovered, dict):
            result[idx] = recovered
            logger.info("[%s] Recovered missing row_id=%s", stage, inp_row.get("row_id", "?"))
        else:
            logger.error(
                "[%s] Could not recover row_id=%s after %d attempts – row will be missing",
                stage, inp_row.get("row_id", "?"), MAX_INDIVIDUAL_ATTEMPTS,
            )

    return [r for r in result if r is not None]


# =========================
# Reviewer decision normalizer (single authoritative version)
# =========================

def _extract_analysis_issues(raw: Dict[str, Any]) -> List[str]:
    """Extract issues from 'analysis' block if present."""
    issues: List[str] = []
    analysis = raw.get("analysis")
    if not isinstance(analysis, dict):
        return issues

    for k, v in analysis.items():
        if isinstance(v, list):
            if v and isinstance(v[0], dict) and "missing_guidewords" in v[0]:
                for d in v:
                    fn = d.get("function", "?")
                    mg = d.get("missing_guidewords", [])
                    issues.append(f"Missing guidewords for '{fn}': {mg}")
            else:
                issues.append(f"{k}: {v}")
        else:
            issues.append(f"{k}: {v}")
    return issues


def _fold_fallback_hints(raw: Dict[str, Any]) -> List[str]:
    """Merge fix_hints, routing_decision, corrective_action into suggestions."""
    hints: List[str] = []
    fix_hints = raw.get("fix_hints")
    if isinstance(fix_hints, dict):
        for k, v in fix_hints.items():
            hints.append(f"{k}: {v}")
    elif isinstance(fix_hints, list):
        hints.extend(fix_hints)

    if raw.get("routing_decision"):
        hints.append(f"routing_decision: {raw['routing_decision']}")
    if raw.get("corrective_action"):
        hints.append(f"corrective_action: {raw['corrective_action']}")
    return hints


def _salvage_suggestions_by_row(raw: Dict[str, Any], suggestions: List[Any]) -> List[Dict[str, Any]]:
    """Extract suggestions_by_row from various locations in the response."""
    sbr = raw.get("suggestions_by_row")
    if isinstance(sbr, list):
        return sbr

    # A) Model stuffed row hints under "rows"
    rows_cand = raw.get("rows")
    if isinstance(rows_cand, list) and rows_cand:
        if isinstance(rows_cand[0], dict) and "row_id" in rows_cand[0] and "fields" in rows_cand[0]:
            return rows_cand

    # B) Model put dict-like row hints as the whole response (rare)
    if all(k in raw for k in ("row_id", "fields")):
        return [{k: raw[k] for k in ("row_id", "fields", "problem", "suggestion") if k in raw}]

    # C) Model put row-hint dicts inside "suggestions"
    if isinstance(suggestions, list) and suggestions:
        if isinstance(suggestions[0], dict) and "row_id" in suggestions[0] and "fields" in suggestions[0]:
            return suggestions

    return []


def _normalize_sbr_item(item: Any) -> Dict[str, Any]:
    """Normalize a single suggestions_by_row entry.

    Returns empty dict if item is invalid (caller should filter empty results).
    """
    if not isinstance(item, dict):
        return {}
    rid = item.get("row_id")
    flds = item.get("fields")
    if isinstance(rid, str) and isinstance(flds, list) and flds:
        return {
            "row_id": rid,
            "fields": [str(f) for f in flds],
            "problem": item.get("problem", ""),
            "suggestion": item.get("suggestion", ""),
        }
    return {}


def _normalize_reviewer_decision(raw: Dict[str, Any], default_scope: str) -> Dict[str, Any]:
    """
    Normalize reviewer outputs into our schema and ALWAYS surface 'suggestions_by_row' if present,
    salvaging from common mis-shapes (e.g., under 'rows' or embedded in 'suggestions').
    Returns:
      {
        "decision":"OK|RETURN|ESCALATE",
        "scope":"L1|L2|L3|ALL",
        "issues":[...],
        "suggestions":[...],
        "suggestions_by_row":[{"row_id": "...", "fields": [...], "problem": "...", "suggestion": "..."}]
      }
    """
    if not isinstance(raw, dict):
        return {
            "decision": "ESCALATE",
            "scope": default_scope.upper(),
            "issues": ["Reviewer returned non-object."],
            "suggestions": [],
            "suggestions_by_row": [],
        }

    decision = (raw.get("decision") or "").upper()
    scope = (raw.get("scope") or default_scope).upper()
    issues = raw.get("issues") or raw.get("errors") or []
    suggestions = raw.get("suggestions") or []

    # Absorb "analysis" blocks into issues if present
    if not issues:
        issues = _extract_analysis_issues(raw)

    # Fold fallback hints into suggestions if suggestions empty
    if not suggestions:
        suggestions = _fold_fallback_hints(raw)

    # Salvage suggestions_by_row from multiple places
    sbr = _salvage_suggestions_by_row(raw, suggestions)

    # Defaults & type safety
    if decision not in ("OK", "RETURN", "ESCALATE"):
        decision = "RETURN"
    if scope not in ("L1", "L2", "L3", "ALL"):
        scope = default_scope.upper()
    if not isinstance(issues, list):
        issues = [str(issues)]
    if not isinstance(suggestions, list):
        suggestions = [str(suggestions)]
    if not isinstance(sbr, list):
        sbr = []

    # Normalize each sbr item (filter empty dicts)
    norm_sbr = [n for n in (_normalize_sbr_item(item) for item in sbr) if n]

    return {
        "decision": decision,
        "scope": scope,
        "issues": issues,
        "suggestions": suggestions,
        "suggestion": " | ".join(suggestions),
        "suggestions_by_row": norm_sbr,
    }


# =========================
# Generic reviewer decision helper
# =========================

def _to_reviewer(validator_report: Dict[str, Any], rows: List[Dict[str, Any]], stage: str, notes: str = "") -> Dict[str, Any]:
    """Generic reviewer decision function for L1/L2/L3 with chunking."""
    stage_lower = stage.lower()
    p = load_prompt(f"{stage_lower}_to_reviewer")

    instr = p["instructions"]
    if notes:
        instr += "\n\nADDITIONAL CONTEXT (RAG / notes):\n" + notes.strip()

    chunks = _chunk_rows_by_function(rows, _REVIEWER_FUNCS_PER_CHUNK)
    if len(chunks) > 1:
        logger.info("%s_REVIEW: %d rows in %d chunk(s)", stage, len(rows), len(chunks))

    decisions = []
    for chunk in chunks:
        payload = render_payload(
            p["payload_template"],
            validator_report=_to_json(validator_report),
            **{f"rows_{stage_lower}_json": _to_json(chunk)},
        )
        raw = chat_json(_build_prompt(p, payload, instr=instr), model=MODEL_CHEAP)
        decisions.append(_normalize_reviewer_decision(raw, default_scope=stage))

    if len(decisions) == 1:
        return decisions[0]
    return _merge_reviewer_decisions(decisions, default_scope=stage)



# =========================
# L1 generation — produces one deviation row per (function × guideword)
# combination, batched by L1_BATCH_SIZE functions at a time.
# =========================

def l1_init_full_coverage(functions, guideword_map, max_devs_per_gw: int = 1, notes: str = "") -> List[Dict[str, Any]]:
    """Generate L1 rows, batching functions to avoid output truncation."""
    BATCH_SIZE = L1_BATCH_SIZE

    all_rows = []

    for i in range(0, len(functions), BATCH_SIZE):
        batch_functions = functions[i:i + BATCH_SIZE]

        p = load_prompt("l1_init")
        expected_min_rows = len(batch_functions) * 11  # 11 fixed guidewords

        payload = render_payload(
            p["payload_template"],
            functions_json=_to_json(batch_functions),
            guideword_map_json=_to_json(guideword_map),
            max_devs_per_gw=str(max_devs_per_gw),
            notes=_to_json(notes or ""),
        )
        instructions = (
            p["instructions"]
            .replace("{{max_devs_per_gw}}", str(max_devs_per_gw))
            .replace("{{expected_min_rows}}", str(expected_min_rows))
        )

        obj = chat_json(_build_prompt(p, payload, instr=instructions), model=MODEL_CHEAP)
        rows = _ensure_rows_list(obj)
        batch_rows = _ensure_list_of_dicts(rows, stage="L1_INIT")
        all_rows.extend(batch_rows)

    # --- Post-batch coverage check: retry completely missing functions ---
    covered_fns = {r.get("function") for r in all_rows}
    missing_fns = [f for f in functions if f not in covered_fns]

    MAX_FN_RETRIES = 2
    for fn in missing_fns:
        logger.warning("L1_INIT: Function '%s' missing from output, retrying individually", fn)
        for attempt in range(1, MAX_FN_RETRIES + 1):
            try:
                p_retry = load_prompt("l1_init")
                expected_retry = 11
                payload_retry = render_payload(
                    p_retry["payload_template"],
                    functions_json=_to_json([fn]),
                    guideword_map_json=_to_json(guideword_map),
                    max_devs_per_gw=str(max_devs_per_gw),
                    notes=_to_json(notes or ""),
                )
                instr_retry = (
                    p_retry["instructions"]
                    .replace("{{max_devs_per_gw}}", str(max_devs_per_gw))
                    .replace("{{expected_min_rows}}", str(expected_retry))
                )
                obj_retry = chat_json(
                    _build_prompt(p_retry, payload_retry, instr=instr_retry),
                    model=MODEL_CHEAP,
                )
                rows_retry = _ensure_rows_list(obj_retry)
                rows_retry = _ensure_list_of_dicts(rows_retry, stage="L1_INIT_FN_RETRY")
                fn_rows = [r for r in rows_retry if r.get("function") == fn]
                if fn_rows:
                    all_rows.extend(fn_rows)
                    logger.info("L1_INIT: Recovered %d rows for '%s' (attempt %d)", len(fn_rows), fn, attempt)
                    break
            except Exception as exc:
                logger.warning("L1_INIT: Retry %d for '%s' failed: %s", attempt, fn, exc)
        else:
            logger.error("L1_INIT: Could not recover '%s' after %d attempts", fn, MAX_FN_RETRIES)

    return all_rows


def l1_to_reviewer(validator_report: Dict[str, Any], rows_l1: List[Dict[str, Any]], notes: str = "") -> Dict[str, Any]:
    return _to_reviewer(validator_report, rows_l1, "L1", notes=notes)


def reviewer_to_l1(
    functions: List[str],
    guideword_map: Dict[str, str],
    fix_hints: List[str],
    max_devs_per_gw: int,
    notes: str = "",
) -> List[Dict[str, Any]]:
    """Regenerate L1 rows after review, batching functions."""
    BATCH_SIZE = L1_BATCH_SIZE

    all_rows = []
    p = load_prompt("reviewer_to_l1")
    instr = p["instructions"]
    if notes:
        instr += "\n\nADDITIONAL CONTEXT (RAG / notes):\n" + notes.strip()

    for i in range(0, len(functions), BATCH_SIZE):
        batch_functions = functions[i:i + BATCH_SIZE]
        payload = render_payload(
            p["payload_template"],
            functions_json=_to_json(batch_functions),
            guideword_map_json=_to_json(guideword_map),
            fix_hints=_to_json(fix_hints),
            max_devs_per_gw=str(max_devs_per_gw),
        )
        obj = chat_json(_build_prompt(p, payload, instr=instr), model=MODEL_CHEAP)

        try:
            rows = _ensure_rows_list(obj)
            batch_rows = _ensure_list_of_dicts(rows, stage="REVIEWER_TO_L1")
        except Exception:
            # Fallback to batched l1_init for this batch
            batch_rows = l1_init_full_coverage(
                batch_functions, guideword_map, max_devs_per_gw,
                notes=notes or "auto-regenerated after reviewer echo",
            )
        all_rows.extend(batch_rows)

    return all_rows


# =========================
# L2 generation — adds a root cause to each L1 row, batched by ROW_BATCH_SIZE.
# =========================

def l2_init_from_l1(rows_l1: List[Dict[str, Any]], notes: str = "") -> List[Dict[str, Any]]:
    """Generate L2 rows, batching input to avoid output truncation."""
    BATCH_SIZE = ROW_BATCH_SIZE

    all_rows = []
    p = load_prompt("l2_init")
    instr = p["instructions"]
    if notes:
        instr = instr + "\n\n" + "ADDITIONAL CONTEXT (RAG / notes):\n" + notes.strip()

    def _retry_l2(inp_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retry callable for individual missing rows."""
        pl = render_payload(p["payload_template"], rows_l1_json=_to_json(inp_rows))
        o = chat_json(_build_prompt(p, pl, instr=instr), model=MODEL_CHEAP)
        return _ensure_list_of_dicts(_ensure_rows_list(o), stage="L2_INIT_RETRY")

    for i in range(0, len(rows_l1), BATCH_SIZE):
        batch_input = rows_l1[i:i + BATCH_SIZE]
        payload = render_payload(p["payload_template"], rows_l1_json=_to_json(batch_input))
        obj = chat_json(_build_prompt(p, payload, instr=instr), model=MODEL_CHEAP)
        rows = _ensure_rows_list(obj)
        batch_output = _ensure_list_of_dicts(rows, stage="L2_INIT")
        batch_output = _verify_and_retry_batch(batch_input, batch_output, _retry_l2, "L2_INIT")
        all_rows.extend(batch_output)

    return all_rows


def l2_to_reviewer(validator_report: Dict[str, Any], rows_l2: List[Dict[str, Any]], notes: str = "") -> Dict[str, Any]:
    return _to_reviewer(validator_report, rows_l2, "L2", notes=notes)



# =========================
# L3 generation — adds system-level effect and safety triage boolean to each
# L2 row, batched by ROW_BATCH_SIZE.
# =========================

def l3_init_from_l2(rows_l2: List[Dict[str, Any]], notes: str = "") -> List[Dict[str, Any]]:
    """Generate L3 rows, batching input to avoid output truncation."""
    BATCH_SIZE = ROW_BATCH_SIZE

    logger.debug("L3_INIT_WRAPPER: Input L2 rows: %d", len(rows_l2))
    all_rows = []
    p = load_prompt("l3_init")
    instr = p["instructions"]
    if notes:
        instr = instr + "\n\n" + "ADDITIONAL CONTEXT (RAG / notes):\n" + notes.strip()

    def _retry_l3(inp_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Retry callable for individual missing rows."""
        pl = render_payload(p["payload_template"], rows_l2_json=_to_json(inp_rows), input_count=str(len(inp_rows)))
        o = chat_json(_build_prompt(p, pl, instr=instr), model=MODEL_CHEAP)
        return _ensure_list_of_dicts(_ensure_rows_list(o), stage="L3_INIT_RETRY")

    for i in range(0, len(rows_l2), BATCH_SIZE):
        batch_input = rows_l2[i:i + BATCH_SIZE]
        payload = render_payload(p["payload_template"], rows_l2_json=_to_json(batch_input), input_count=str(len(batch_input)))
        obj = chat_json(_build_prompt(p, payload, instr=instr), model=MODEL_CHEAP)
        logger.debug("L3_INIT batch %d: type=%s", i // BATCH_SIZE, type(obj))
        rows = _ensure_rows_list(obj)
        batch_output = _ensure_list_of_dicts(rows, stage="L3_INIT")
        batch_output = _verify_and_retry_batch(batch_input, batch_output, _retry_l3, "L3_INIT")
        all_rows.extend(batch_output)

    logger.debug("L3_INIT_WRAPPER: FINAL returned rows: %d", len(all_rows))
    return all_rows


def l3_to_reviewer(validator_report: Dict[str, Any], rows_l3: List[Dict[str, Any]], notes: str = "") -> Dict[str, Any]:
    return _to_reviewer(validator_report, rows_l3, "L3", notes=notes)



# =========================
# Targeted row patching — _patch_rows sends specific rows plus reviewer hints
# to the LLM for repair.  Non-targeted rows pass through unchanged.
# For human-initiated patches, verbatim copy detection catches cases where the
# LLM lazily copies the user's suggestion text into output fields instead of
# interpreting it as an instruction.  Detected copies trigger automatic retry.
# =========================

# Target fields per stage for verbatim copy detection (AI-HAZOP-8800 phases)
_VERBATIM_TARGET_FIELDS = {
    "L1": ["failure_mode"],
    "L2": ["hazardous_behavior", "potential_harm"],
    "L3": ["risk_rationale"],
    "L4": ["acceptance_rationale"],
    "L5": ["ai_safety_goals"],
    "L6": ["respecifications", "safety_functions", "passive_operational_measures"],
    "L7": ["residual_rationale"],
    "L8": ["evidence", "open_assumptions"],
}

# Max retries when verbatim copy detected
_MAX_VERBATIM_RETRIES = 2


def _detect_verbatim_copy(
    patched_rows: List[Dict[str, Any]],
    row_hints: List[Dict[str, Any]],
    target_fields: List[str],
) -> List[str]:
    """
    Detect rows where the user's suggestion was copied verbatim into output fields.

    Returns list of row_ids where verbatim copy was detected.
    """
    problematic = []

    # Build a map of row_id -> suggestion
    hint_map = {}
    for hint in row_hints:
        rid = hint.get("row_id", "")
        suggestion = hint.get("suggestion", "")
        if rid and suggestion and len(suggestion.strip()) >= 10:
            hint_map[rid] = suggestion

    if not hint_map:
        return []

    for row in patched_rows:
        rid = row.get("row_id", "")
        suggestion = hint_map.get(rid, "")
        if not suggestion:
            continue

        # Normalize suggestion for comparison
        suggestion_norm = ' '.join(suggestion.strip().lower().split())

        for field in target_fields:
            value = str(row.get(field, "")).strip()
            if not value:
                continue
            value_norm = ' '.join(value.lower().split())

            # Check for exact match or containment
            if suggestion_norm in value_norm or value_norm == suggestion_norm:
                problematic.append(rid)
                logger.warning(
                    "Verbatim copy detected: row_id=%s, field=%s, suggestion='%s...'",
                    rid, field, suggestion[:50]
                )
                break

    return problematic


def _patch_rows(
    rows: List[Dict[str, Any]],
    row_hints: List[Dict[str, Any]],
    stage: str,
    source: str = "automated",
    notes: str = "",
) -> List[Dict[str, Any]]:
    """
    Generic patch function for L1/L2/L3 rows with batching.
    Patch only specified rows (by row_id) based on reviewer hints.
    Non-targeted rows MUST be returned unchanged and in the same order.

    Args:
        rows: Current rows to patch
        row_hints: List of hints with row_id, fields, and suggestion
        stage: "L1", "L2", or "L3"
        source: "human" for user-initiated regeneration (uses stronger anti-copy prompts),
                "automated" for pipeline-internal repairs (uses standard reviewer prompts)
    """
    BATCH_SIZE = ROW_BATCH_SIZE
    stage_lower = stage.lower()

    # Select prompt based on source
    if source == "human":
        prompt_name = f"human_patch_{stage_lower}"
    else:
        prompt_name = f"reviewer_patch_{stage_lower}"

    p = load_prompt(prompt_name)
    instr = p["instructions"]
    if notes:
        instr += "\n\nADDITIONAL CONTEXT (RAG / notes):\n" + notes.strip()
    target_fields = _VERBATIM_TARGET_FIELDS.get(stage.upper(), [])

    all_rows = []

    for i in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[i:i + BATCH_SIZE]
        batch_row_ids = {r.get("row_id") for r in batch_rows}

        # Filter hints to only those for this batch
        batch_hints = [h for h in row_hints if h.get("row_id") in batch_row_ids]

        if not batch_hints:
            # No hints for this batch, return unchanged
            all_rows.extend(batch_rows)
            continue

        payload = render_payload(
            p["payload_template"],
            **{f"rows_{stage_lower}_json": _to_json(batch_rows)},
            row_hints_json=_to_json(batch_hints),
        )
        obj = chat_json(_build_prompt(p, payload, instr=instr), model=MODEL_CHEAP)
        result = _ensure_rows_list(obj)
        result = _ensure_list_of_dicts(result, stage=f"REVIEWER_PATCH_{stage}")

        # Check for verbatim copy and retry if needed (per batch). This runs for BOTH
        # human and automated patches: automated repairs pass the reviewer/holistic
        # suggestion as the hint, and the LLM sometimes copies that instruction text
        # straight into a field (e.g. "Ensure consistency between risk_status and
        # safety_decision." landing in failure_mode). Detect and regenerate.
        if target_fields:
            for retry in range(_MAX_VERBATIM_RETRIES):
                problematic_ids = _detect_verbatim_copy(result, batch_hints, target_fields)
                if not problematic_ids:
                    break

                logger.info(
                    "Verbatim copy retry %d/%d for stage %s, rows: %s",
                    retry + 1, _MAX_VERBATIM_RETRIES, stage, problematic_ids
                )

                # Add reinforcement message to instructions
                reinforcement = (
                    "\n\n### CRITICAL CORRECTION ###\n"
                    "Your previous output copied the user's suggestion text VERBATIM into the output fields. "
                    "This is WRONG. You must INTERPRET and TRANSFORM the suggestion into technical HAZOP content. "
                    f"The following rows had verbatim copies: {problematic_ids}. "
                    "Generate NEW technical content that addresses the user's intent WITHOUT copying their words.\n"
                )

                obj = chat_json(
                    _build_prompt(p, payload, instr=instr, extra=reinforcement),
                    model=MODEL_CHEAP
                )
                result = _ensure_rows_list(obj)
                result = _ensure_list_of_dicts(result, stage=f"REVIEWER_PATCH_{stage}")

        all_rows.extend(result)

    return all_rows


def reviewer_patch_l1(
    rows_l1: List[Dict[str, Any]],
    row_hints: List[Dict[str, Any]],
    source: str = "automated",
    notes: str = "",
) -> List[Dict[str, Any]]:
    """Patch L1 rows. Use source='human' for user-initiated regeneration."""
    return _patch_rows(rows_l1, row_hints, "L1", source=source, notes=notes)


def reviewer_patch_l2(
    rows_l2: List[Dict[str, Any]],
    row_hints: List[Dict[str, Any]],
    source: str = "automated",
    notes: str = "",
) -> List[Dict[str, Any]]:
    """Patch L2 rows. Use source='human' for user-initiated regeneration."""
    return _patch_rows(rows_l2, row_hints, "L2", source=source, notes=notes)


def reviewer_patch_l3(
    rows_l3: List[Dict[str, Any]],
    row_hints: List[Dict[str, Any]],
    source: str = "automated",
    notes: str = "",
) -> List[Dict[str, Any]]:
    """Patch L3 rows. Use source='human' for user-initiated regeneration."""
    return _patch_rows(rows_l3, row_hints, "L3", source=source, notes=notes)


# =========================
# Holistic reviewer — final cross-stage consistency check performed after all
# three stages pass validation.  Reviews the full L3 table in chunks and
# returns a decision (OK / RETURN) with a scope indicating which stage to fix.
# =========================

_HOL_KEEP_KEYS = ("row_id", "function", "guideword", "deviation", "cause",
                   "effect", "potentially_dangerous")
_HOL_TRUNCATE = {"deviation": 120, "cause": 120}
_HOL_FUNCS_PER_CHUNK = 3  # 3 functions × 11 guidewords = 33 rows/chunk ≈ 12K tokens


def _chunk_rows_by_function(rows, funcs_per_chunk):
    """Group rows by function, then batch function-groups into chunks."""
    from collections import OrderedDict
    grouped = OrderedDict()
    for r in rows:
        grouped.setdefault(r.get("function", ""), []).append(r)
    chunks, current, count = [], [], 0
    for fn_rows in grouped.values():
        if count >= funcs_per_chunk and current:
            chunks.append(current)
            current, count = [], 0
        current.extend(fn_rows)
        count += 1
    if current:
        chunks.append(current)
    return chunks


def _merge_holistic_decisions(decisions):
    """Merge chunk decisions: any RETURN → RETURN, highest scope wins, union issues."""
    if not decisions:
        return {"decision": "OK", "scope": "ALL", "issues": [], "suggestion": ""}
    scope_rank = {"L1": 3, "L2": 2, "L3": 1, "ALL": 0}
    has_return = any(d.get("decision") == "RETURN" for d in decisions)
    best_scope = max(
        (d.get("scope", "ALL") for d in decisions),
        key=lambda s: scope_rank.get(s, 0),
    )
    all_issues, seen = [], set()
    for d in decisions:
        for iss in d.get("issues", []):
            if iss not in seen:
                all_issues.append(iss)
                seen.add(iss)
    suggestions = [d.get("suggestion", "") for d in decisions
                   if d.get("decision") == "RETURN" and d.get("suggestion", "").strip()]
    return {
        "decision": "RETURN" if has_return else "OK",
        "scope": best_scope,
        "issues": all_issues,
        "suggestion": " | ".join(suggestions) if suggestions else "",
    }


def _merge_reviewer_decisions(decisions, default_scope):
    """Merge chunked reviewer decisions: any RETURN → RETURN, union issues/suggestions."""
    if not decisions:
        return {"decision": "OK", "scope": default_scope, "issues": [],
                "suggestions": [], "suggestions_by_row": []}
    has_return = any(d.get("decision") == "RETURN" for d in decisions)
    all_issues, seen_issues = [], set()
    all_suggestions, seen_suggestions = [], set()
    all_sbr = []
    for d in decisions:
        for iss in d.get("issues", []):
            if iss not in seen_issues:
                all_issues.append(iss)
                seen_issues.add(iss)
        for sug in d.get("suggestions", []):
            if sug not in seen_suggestions:
                all_suggestions.append(sug)
                seen_suggestions.add(sug)
        all_sbr.extend(d.get("suggestions_by_row", []))
    return {
        "decision": "RETURN" if has_return else "OK",
        "scope": default_scope,
        "issues": all_issues,
        "suggestions": all_suggestions,
        "suggestions_by_row": all_sbr,
    }


def l3_pass_to_reviewer_holistic(rows_l3: List[Dict[str, Any]], lite_findings: List[str], notes: str = "") -> Dict[str, Any]:
    p = load_prompt("l3_pass_to_reviewer_holistic")

    instr = p["instructions"]
    if notes:
        instr += "\n\nADDITIONAL CONTEXT (RAG / notes):\n" + notes.strip()

    chunks = _chunk_rows_by_function(rows_l3, _HOL_FUNCS_PER_CHUNK)
    logger.info("HOL_REVIEW: %d rows in %d chunk(s) (%d funcs/chunk)",
                len(rows_l3), len(chunks), _HOL_FUNCS_PER_CHUNK)

    decisions = []
    for i, chunk in enumerate(chunks):
        slim = []
        for r in chunk:
            sr = {k: r[k] for k in _HOL_KEEP_KEYS if k in r}
            for field, limit in _HOL_TRUNCATE.items():
                val = sr.get(field, "")
                if len(val) > limit:
                    sr[field] = val[:limit] + "..."
            slim.append(sr)
        payload = render_payload(
            p["payload_template"],
            rows_l3_json=_to_json(slim),
            lite_validator_findings=_to_json(lite_findings if i == 0 else []),
        )
        raw = chat_json(_build_prompt(p, payload, instr=instr), model=MODEL_REVIEW)
        decisions.append(_normalize_reviewer_decision(raw, default_scope="ALL"))

    if len(decisions) == 1:
        return decisions[0]
    return _merge_holistic_decisions(decisions)
