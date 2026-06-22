"""LLM generation chains for the AI-HAZOP-8800 pipeline (phases L1–L8).

These functions are thin, methodology-specific wrappers around the generic chain
machinery in ``chains.py`` (batching, dropped-row recovery, targeted patching,
reviewer-decision normalisation), which is reused verbatim. The phase generators
follow one pattern:

  * L1 generates one failure-mode row per guideword for a single component+aspect.
  * L2–L8 take the previous phase's rows and return ONLY the new fields keyed by
    row_id; ``_overlay_new_fields`` merges them onto the carried rows so all
    upstream fields (and the authoritative user context) are preserved.
"""
from __future__ import annotations

import logging
from copy import deepcopy
from typing import Any, Dict, List

from .prompt_loader import load_prompt, render_payload
from .llm_client import chat_json, MODEL_CHEAP, MODEL_REVIEW
from .row_utils import _ensure_rows_list, _ensure_list_of_dicts, _suffix
from .chains import (
    _to_json,
    _build_prompt,
    _verify_and_retry_batch,
    _to_reviewer,
    _patch_rows,
    _normalize_sbr_item,
    ROW_BATCH_SIZE,
)

_HOLISTIC_SCOPES = ("L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8")

logger = logging.getLogger(__name__)

# Context fields that are authoritative from user input and stamped by code onto
# every row — never invented or altered by the LLM.
CONTEXT_FIELDS = ("component", "component_class", "aspect", "odd", "scenario")


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


# ===========================================================================
# L1 — failure modes (one row per guideword for a single component+aspect)
# ===========================================================================

def ai_l1_generate(
    ctx: Dict[str, str],
    guidewords: List[Dict[str, Any]],
    class_questions: List[str],
    aspect_hint: str = "",
    notes: str = "",
) -> List[Dict[str, Any]]:
    """Generate one L1 failure-mode row per enabled guideword.

    Args:
        ctx: the analysis context (component, component_class, aspect, odd, scenario).
        guidewords: the working guideword catalogue entries (id/name/meaning/ai_examples).
        class_questions: class-specific questions for ctx['component_class'].
        aspect_hint: typical hazard-contribution hint for the aspect.
        notes: optional extra context (RAG/notes).
    """
    p = load_prompt("l1_init")
    instr = p["instructions"]
    if notes:
        instr += "\n\nADDITIONAL CONTEXT (RAG / notes):\n" + notes.strip()

    valid_ids = [g["id"] for g in guidewords]
    # resolver maps an LLM-returned guideword token (id or name) -> canonical id
    resolver: Dict[str, str] = {}
    for g in guidewords:
        resolver[_norm(g["id"])] = g["id"]
        resolver[_norm(g.get("name"))] = g["id"]

    def _call(gw_subset: List[Dict[str, Any]]) -> Dict[str, str]:
        payload = render_payload(
            p["payload_template"],
            component=_to_json(ctx["component"]),
            component_class=_to_json(ctx["component_class"]),
            aspect=_to_json(ctx["aspect"]),
            aspect_hint=_to_json(aspect_hint or ""),
            odd=_to_json(ctx["odd"]),
            scenario=_to_json(ctx["scenario"]),
            guidewords_json=_to_json(gw_subset),
            class_questions_json=_to_json(class_questions or []),
            notes=_to_json(notes or ""),
        )
        obj = chat_json(_build_prompt(p, payload, instr=instr), model=MODEL_CHEAP)
        rows = _ensure_list_of_dicts(_ensure_rows_list(obj), stage="L1_INIT")
        out: Dict[str, str] = {}
        for r in rows:
            gid = resolver.get(_norm(r.get("guideword")))
            fm = str(r.get("failure_mode") or "").strip()
            if gid and fm:
                out[gid] = fm
        return out

    # Generate in small guideword batches so a long prompt + many guidewords cannot
    # truncate the output and silently drop the first/last guidewords (FIX: coverage).
    GW_BATCH = 6
    fm_by_gw: Dict[str, str] = {}
    for i in range(0, len(guidewords), GW_BATCH):
        sub = guidewords[i:i + GW_BATCH]
        try:
            fm_by_gw.update(_call(sub))
        except Exception as exc:
            logger.warning("L1_INIT: batch %d failed: %s", i // GW_BATCH, exc)

    # Coverage retry: retry each missing guideword INDIVIDUALLY so one bad token in a
    # batch can't leave a whole set of rows with an empty failure_mode.
    missing = [g for g in guidewords if g["id"] not in fm_by_gw]
    if missing:
        logger.warning("L1_INIT: %d guidewords missing, retrying individually", len(missing))
        for g in missing:
            try:
                fm_by_gw.update(_call([g]))
            except Exception as exc:
                logger.warning("L1_INIT: coverage retry failed for %s: %s", g["id"], exc)
    still_missing = [g["id"] for g in guidewords if g["id"] not in fm_by_gw]
    if still_missing:
        logger.error("L1_INIT: %d guideword(s) still empty (will be blocked downstream): %s",
                     len(still_missing), still_missing)

    # Assemble canonical rows in guideword order with stamped context + row_ids.
    rows: List[Dict[str, Any]] = []
    for i, g in enumerate(guidewords):
        row = {"row_id": f"L1-{i}"}
        for f in CONTEXT_FIELDS:
            row[f] = ctx[f]
        row["guideword"] = g["id"]
        row["failure_mode"] = fm_by_gw.get(g["id"], "")
        rows.append(row)
    logger.info("L1_INIT: generated %d rows (%d guidewords)", len(rows), len(valid_ids))
    return rows


# ===========================================================================
# Generic forward phase (L2–L8): add new fields to the previous phase's rows
# ===========================================================================

def _item_to_str(x: Any) -> str:
    """Flatten a list item to a readable string.

    LLMs sometimes return list fields (evidence, measures, goals) as objects, e.g.
    {"evidence_type": "DatasetAudit", "test_activity": "Coverage matrix"}. Join the
    values with ' — ' so downstream (worksheet, GUI, CSV) shows readable text rather
    than a dict repr / "[object Object]".
    """
    if isinstance(x, dict):
        return " — ".join(str(v).strip() for v in x.values() if str(v).strip())
    if isinstance(x, list):
        return "; ".join(str(i).strip() for i in x if str(i).strip())
    return str(x).strip()


def _clean_field_value(v: Any) -> Any:
    """Normalise a field value; coerce list items to strings (see _item_to_str)."""
    if isinstance(v, list):
        return [_item_to_str(x) for x in v if _item_to_str(x)]
    return v


def _overlay_new_fields(
    prev_rows: List[Dict[str, Any]],
    new_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Overlay LLM-returned new fields onto carried rows, matched by row_id suffix.

    Preserves every upstream field (and authoritative context); only the keys
    present in ``new_rows`` (other than row_id) are written. Output keeps prev order.
    """
    by_suffix: Dict[str, Dict[str, Any]] = {}
    for nr in new_rows:
        rid = nr.get("row_id")
        if rid is not None:
            by_suffix[_suffix(str(rid))] = nr
    out: List[Dict[str, Any]] = []
    for pr in prev_rows:
        merged = deepcopy(pr)
        nr = by_suffix.get(_suffix(str(pr.get("row_id", ""))))
        if nr:
            for k, v in nr.items():
                if k == "row_id":
                    continue
                merged[k] = _clean_field_value(v)
        out.append(merged)
    return out


def ai_phase_generate(
    phase: str,
    prev_rows: List[Dict[str, Any]],
    extra_inputs: Dict[str, Any] | None = None,
    notes: str = "",
) -> List[Dict[str, Any]]:
    """Generate phase fields from the previous phase's rows (L2–L8).

    The phase ``{phase}_init`` prompt receives ``rows_prev_json`` plus any
    ``extra_inputs`` (already-stringified payload values keyed by placeholder
    name, e.g. ``risk_scales_json``). The LLM returns minimal {row_id, new fields}
    objects which are overlaid onto ``prev_rows``.
    """
    extra_inputs = extra_inputs or {}
    p = load_prompt(f"{phase.lower()}_init")
    instr = p["instructions"]
    if notes:
        instr += "\n\nADDITIONAL CONTEXT (RAG / notes):\n" + notes.strip()

    def _retry(inp_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        pl = render_payload(p["payload_template"], rows_prev_json=_to_json(inp_rows), **extra_inputs)
        o = chat_json(_build_prompt(p, pl, instr=instr), model=MODEL_CHEAP)
        return _ensure_list_of_dicts(_ensure_rows_list(o), stage=f"{phase}_INIT_RETRY")

    all_new: List[Dict[str, Any]] = []
    for i in range(0, len(prev_rows), ROW_BATCH_SIZE):
        batch = prev_rows[i:i + ROW_BATCH_SIZE]
        payload = render_payload(p["payload_template"], rows_prev_json=_to_json(batch), **extra_inputs)
        obj = chat_json(_build_prompt(p, payload, instr=instr), model=MODEL_CHEAP)
        new_batch = _ensure_list_of_dicts(_ensure_rows_list(obj), stage=f"{phase}_INIT")
        new_batch = _verify_and_retry_batch(batch, new_batch, _retry, f"{phase}_INIT")
        all_new.extend(new_batch)

    merged = _overlay_new_fields(prev_rows, all_new)
    logger.info("%s_INIT: produced %d rows from %d input rows", phase, len(merged), len(prev_rows))
    return merged


# ===========================================================================
# Reviewer + patch wrappers (reuse the generic chains.py machinery)
# ===========================================================================

def ai_reviewer(phase: str, validator_report: Dict[str, Any], rows: List[Dict[str, Any]], notes: str = "") -> Dict[str, Any]:
    """Run the phase reviewer; returns a normalised decision dict."""
    return _to_reviewer(validator_report, rows, phase.upper(), notes=notes) or {}


def ai_patch(phase: str, rows: List[Dict[str, Any]], row_hints: List[Dict[str, Any]], notes: str = "") -> List[Dict[str, Any]]:
    """Targeted repair of specific rows for a phase (automated source)."""
    return _patch_rows(rows, row_hints, phase.upper(), source="automated", notes=notes)


# ===========================================================================
# Holistic reviewer — cross-phase consistency check over the whole worksheet.
# Scope may name any phase L1–L8 (unlike the per-phase reviewer normaliser in
# chains.py, which only knows L1–L3/ALL), so it has a dedicated normaliser.
# ===========================================================================

def _normalize_holistic_decision(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"decision": "OK", "scope": "L1", "issues": [], "suggestion": "",
                "suggestions_by_row": [], "target_ids": []}
    decision = (raw.get("decision") or "").upper()
    if decision not in ("OK", "RETURN"):
        decision = "RETURN" if (raw.get("issues") or raw.get("suggestions_by_row")) else "OK"
    scope = (raw.get("scope") or "L1").upper()
    if scope not in _HOLISTIC_SCOPES:
        scope = "L1"
    issues = raw.get("issues") if isinstance(raw.get("issues"), list) else []
    suggestions = raw.get("suggestions") if isinstance(raw.get("suggestions"), list) else []
    sbr = raw.get("suggestions_by_row") if isinstance(raw.get("suggestions_by_row"), list) else []
    norm_sbr = [n for n in (_normalize_sbr_item(it) for it in sbr) if n]
    target_ids = [s["row_id"] for s in norm_sbr if s.get("row_id")]
    return {
        "decision": decision,
        "scope": scope,
        "issues": issues,
        "suggestion": " | ".join([str(s) for s in suggestions] + [s.get("suggestion", "") for s in norm_sbr]).strip(" |"),
        "suggestions_by_row": norm_sbr,
        "target_ids": target_ids,
    }


def ai_holistic_review(rows_view: List[Dict[str, Any]], lite_findings: List[str], notes: str = "") -> Dict[str, Any]:
    """Run the holistic cross-phase reviewer over a slim worksheet view."""
    p = load_prompt("holistic_review")
    instr = p["instructions"]
    if notes:
        instr += "\n\nADDITIONAL CONTEXT (RAG / notes):\n" + notes.strip()
    payload = render_payload(
        p["payload_template"],
        rows_json=_to_json(rows_view),
        lite_findings=_to_json(lite_findings or []),
    )
    raw = chat_json(_build_prompt(p, payload, instr=instr), model=MODEL_REVIEW)
    return _normalize_holistic_decision(raw)
