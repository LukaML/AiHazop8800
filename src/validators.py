"""Pydantic-based validation for each AI-HAZOP-8800 phase output (L1–L8).

Validation flow per phase:
  1. Unwrap LLM output from common dict wrappers ({"rows": [...]})
  2. Parse each row into the corresponding Pydantic model
  3. Run structural checks: non-empty required fields, no duplicates
  4. For L1: verify full guideword coverage (every enabled guideword present)

Rows that fail Pydantic validation are kept as plain dicts (with issues logged by
row_id) so the repair loop can target and fix them.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Type

from pydantic import ValidationError

from .models import (
    AIHazopL1Row,
    AIHazopL2Row,
    AIHazopL3Row,
    AIHazopL4Row,
    AIHazopL5Row,
    AIHazopL6Row,
    AIHazopL7Row,
    AIHazopL8Row,
    SAFETY_DECISIONS,
)
from .risk_model import factor_value, FACTORS
from .row_utils import _row_get, WRAPPER_KEYS, _is_meta_text

__all__ = [
    "parse_rows_safely",
    "validate_l1_payload",
    "validate_l2_payload",
    "validate_l3_payload",
    "validate_l4_payload",
    "validate_l5_payload",
    "validate_l6_payload",
    "validate_l7_payload",
    "validate_l8_payload",
    "build_validator_report",
    "is_valid_l1_row",
    "is_exportable_row",
]

_MODEL_PREFIX = {
    AIHazopL1Row: "L1",
    AIHazopL2Row: "L2",
    AIHazopL3Row: "L3",
    AIHazopL4Row: "L4",
    AIHazopL5Row: "L5",
    AIHazopL6Row: "L6",
    AIHazopL7Row: "L7",
    AIHazopL8Row: "L8",
}


def _unwrap_to_list(payloads: Any) -> Any:
    if isinstance(payloads, list):
        return payloads
    if isinstance(payloads, dict):
        for k in WRAPPER_KEYS:
            if k in payloads and isinstance(payloads[k], list):
                return payloads[k]
        if len(payloads) == 1:
            sole_val = next(iter(payloads.values()))
            if isinstance(sole_val, list):
                return sole_val
    return payloads


def parse_rows_safely(payloads: Any, model: Type[Any]) -> Tuple[List[Any], List[str]]:
    """Unwrap → parse each row into the Pydantic model → collect issues.

    Rows that fail validation are retained as dicts (row_id injected) so the
    repair loop can still reference them.
    """
    issues: List[str] = []
    rows: List[Any] = []

    payloads = _unwrap_to_list(payloads)
    if not isinstance(payloads, list):
        issues.append(f"Top-level payload is not a list, got: {type(payloads).__name__}")
        return rows, issues

    prefix = _MODEL_PREFIX.get(model, "L1")
    for i, payload in enumerate(payloads):
        if isinstance(payload, str):
            import json
            try:
                payload = json.loads(payload)
            except Exception as e:
                issues.append(f"row[{i}] is a string but not JSON-decodable: {e}")
                continue
        if not isinstance(payload, dict):
            issues.append(f"row[{i}] is not a dict after normalization (got {type(payload).__name__}).")
            continue

        if "row_id" not in payload:
            payload["row_id"] = f"{prefix}-{i}"

        try:
            rows.append(model(**payload))
        except ValidationError as ve:
            issues.append(f"row[{i}] (row_id={payload.get('row_id')}) validation error: {ve.errors()}")
            d = dict(payload)
            if "row_id" in d and not isinstance(d["row_id"], str):
                d["row_id"] = str(d["row_id"])
            rows.append(d)

    return rows, issues


# ---------------------------------------------------------------------------
# Structural quality checks
# ---------------------------------------------------------------------------

def _check_duplicates(rows: List[Any], fields: List[str]) -> List[str]:
    seen = set()
    dups: List[str] = []
    for i, r in enumerate(rows):
        key = tuple(_row_get(r, f) for f in fields)
        if key in seen:
            dups.append(f"duplicate row at index {i} for key {fields}={key}")
        else:
            seen.add(key)
    return dups


def _check_non_empty(rows: List[Any], field: str) -> List[str]:
    issues: List[str] = []
    for i, r in enumerate(rows):
        val = _row_get(r, field)
        rid = _row_get(r, "row_id")
        if val is None or (isinstance(val, str) and not val.strip()):
            issues.append(f"row[{i}] (row_id={rid}) has empty '{field}'")
    return issues


def _check_guideword_coverage(rows: List[Any], expected_guidewords: List[str]) -> List[str]:
    """Verify every enabled guideword id appears at least once (L1 coverage)."""
    if not expected_guidewords:
        return []
    seen = {str(_row_get(r, "guideword") or "").strip() for r in rows}
    missing = [g for g in expected_guidewords if g not in seen]
    if missing:
        return [f"missing guidewords (no failure_mode row): {missing}"]
    return []


_DISTINCT_STOPWORDS = set(
    "a an the of to in on at for is are be was were by with and or as it its that this from into "
    "out over under after before too not no than then when where which who whom whose will would "
    "can could may might shall should component system shuttle vehicle output outputs input inputs "
    "value values when due causing leading".split()
)


def _content_tokens(s: str) -> set:
    words = re.findall(r"[a-z0-9]+", str(s).lower())
    return {w for w in words if len(w) > 2 and w not in _DISTINCT_STOPWORDS}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _check_distinct(rows: List[Any], field: str, noun: str, threshold: float = 0.7) -> List[str]:
    """Each guideword must yield a SUBSTANTIALLY different value for ``field``.

    Flags any value that is identical (after normalisation) or highly similar (content-word
    Jaccard >= threshold) to an earlier one, so the repair loop rewrites it to express that
    guideword's distinct deviation instead of a generic paraphrase. Used for L1 failure_mode
    and L2 hazardous_behavior.
    """
    issues: List[str] = []
    norm = []
    for r in rows:
        v = str(_row_get(r, field) or "").strip()
        exact = re.sub(r"\s+", " ", v.lower()).strip(" .")
        norm.append((_row_get(r, "row_id"), _row_get(r, "guideword"), v, _content_tokens(v), exact))

    flagged = set()
    for j in range(len(norm)):
        rid_j, gw_j, v_j, tok_j, exact_j = norm[j]
        if not v_j or j in flagged:
            continue
        for i in range(j):
            rid_i, gw_i, v_i, tok_i, exact_i = norm[i]
            if not v_i:
                continue
            if exact_i == exact_j or _jaccard(tok_i, tok_j) >= threshold:
                issues.append(
                    f"row[{j}] (row_id={rid_j}) {noun} for guideword '{gw_j}' is too similar to "
                    f"guideword '{gw_i}'; rewrite it to express the distinct {noun} for the '{gw_j}' deviation"
                )
                flagged.add(j)
                break
    return issues


_VAGUE_HAZARD_PHRASES = (
    "hazardous situation", "dangerous situation", "unsafe situation", "hazardous condition",
    "may cause harm", "could cause harm", "may cause a hazard", "cause a hazard",
    "potential hazard", "safety issue", "safety problem",
)


def _check_specific_hazard(rows: List[Any]) -> List[str]:
    """Flag generic, non-specific hazardous_behavior text (e.g. 'may cause hazardous situation')."""
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        v = str(_row_get(r, "hazardous_behavior") or "").strip().lower()
        if v and any(p in v for p in _VAGUE_HAZARD_PHRASES):
            issues.append(
                f"row[{i}] (row_id={rid}) hazardous_behavior is vague; describe the specific unsafe "
                f"vehicle-level behavior caused by this deviation, not a generic 'hazardous situation'"
            )
    return issues


L1_MIN_WORDS = 8


def is_valid_l1_row(row: Any) -> bool:
    """Hard L1 gate: a failure_mode must be present, substantial, and not meta text.

    Used to BLOCK invalid L1 rows from downstream phases and from export (a row that
    fails this never gets L2-L8 generated and is omitted from the worksheet)."""
    fm = _row_get(row, "failure_mode")
    if not isinstance(fm, str) or not fm.strip():
        return False
    if _is_meta_text(fm):
        return False
    if len(fm.split()) < L1_MIN_WORDS:
        return False
    return True


def _check_min_words(rows: List[Any], field: str, n: int, noun: str) -> List[str]:
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        v = str(_row_get(r, field) or "").strip()
        if v and len(v.split()) < n:
            issues.append(
                f"row[{i}] (row_id={rid}) {noun} is too short ({len(v.split())} words); write a full "
                f"component-level sentence of at least {n} words"
            )
    return issues


def _check_no_meta_text(rows: List[Any], fields: List[str]) -> List[str]:
    """Reject reviewer/validator/repair instruction text leaking into worksheet fields (#1/#10)."""
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        for f in fields:
            v = _row_get(r, f)
            items = v if isinstance(v, list) else [v]
            if any(_is_meta_text(x) for x in items):
                issues.append(
                    f"row[{i}] (row_id={rid}) {f} contains reviewer/validator instruction text; "
                    f"replace it with real HAZOP content for this field"
                )
                break
    return issues


_VAGUE_HARM_PHRASES = (
    "harm to people", "harm to the vehicle", "possible harm", "potential harm",
    "could cause harm", "may cause harm", "unsafe outcome", "safety impact",
    "negative impact", "damage may occur", "harm may occur", "injury may occur",
)


def _check_specific_text(rows: List[Any], field: str, vague_phrases: tuple, noun: str) -> List[str]:
    """Flag vague, non-specific text in a scalar field (e.g. potential_harm)."""
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        v = str(_row_get(r, field) or "").strip()
        if not v:
            continue
        low = v.lower()
        if any(p in low for p in vague_phrases) or len(_content_tokens(v)) < 3:
            issues.append(
                f"row[{i}] (row_id={rid}) {noun} is vague; state specifically who/what is harmed and how severely"
            )
    return issues


_GENERIC_GOAL_PHRASES = (
    "improve detection", "improve accuracy", "ensure safety", "improve performance",
    "enhance robustness", "improve robustness", "increase reliability", "be safe",
    "improve quality", "better detection", "improve the model", "make it safer",
)


def _check_specific_goals(rows: List[Any]) -> List[str]:
    """Reject generic AI safety goals; each must be a hazard-specific objective (#7)."""
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        goals = _row_get(r, "ai_safety_goals")
        if not isinstance(goals, list):
            continue
        for g in goals:
            gl = str(g).strip().lower()
            if not gl:
                continue
            if any(p in gl for p in _GENERIC_GOAL_PHRASES) or len(_content_tokens(g)) < 3:
                issues.append(
                    f"row[{i}] (row_id={rid}) safety goal '{g}' is too generic; state a specific objective "
                    f"tied to this hazard"
                )
    return issues


def _check_decision_consistency(rows: List[Any]) -> List[str]:
    """safety_decision must match the code-computed risk_status (#9)."""
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        status = str(_row_get(r, "risk_status") or "").strip().upper()
        decision = str(_row_get(r, "safety_decision") or "").strip().upper()
        if not status or not decision or status == "INCOMPLETE":
            continue
        acceptable = status == "ACCEPTABLE"
        if acceptable and decision != "ACCEPT":
            issues.append(
                f"row[{i}] (row_id={rid}) risk_status is ACCEPTABLE but safety_decision is {decision}; "
                f"it must be ACCEPT"
            )
        if not acceptable and decision == "ACCEPT":
            issues.append(
                f"row[{i}] (row_id={rid}) risk_status is {status} but safety_decision is ACCEPT; "
                f"choose IMPROVE, RESTRICT, or INVESTIGATE"
            )
    return issues


_MEASURE_ID_RE = re.compile(r"\((SF|R|P)\s*(\d+)\)", re.I)


def _measure_ids(text: str) -> set:
    """Extract measure ids like (R1)/(SF1)/(P1) from a string, normalised e.g. 'SF1'."""
    return {f"{m.group(1).upper()}{m.group(2)}" for m in _MEASURE_ID_RE.finditer(str(text or ""))}


def _check_evidence(rows: List[Any]) -> List[str]:
    """Evidence must verify the concrete measures by id, not restate the hazard (#4/#6)."""
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        ev = _row_get(r, "evidence")
        if not isinstance(ev, list):
            continue
        items = [str(x).strip() for x in ev if str(x).strip()]
        if not items:
            continue

        hb_tokens = _content_tokens(str(_row_get(r, "hazardous_behavior") or ""))
        measure_ids: set = set()
        for f in ("respecifications", "safety_functions", "passive_operational_measures"):
            v = _row_get(r, f)
            if isinstance(v, list):
                for m in v:
                    measure_ids |= _measure_ids(m)

        if hb_tokens and any(_jaccard(_content_tokens(e), hb_tokens) >= 0.7 for e in items):
            issues.append(
                f"row[{i}] (row_id={rid}) evidence merely restates the hazardous behavior; describe a "
                f"test or review activity that verifies a specific measure"
            )

        if measure_ids:
            referenced: set = set()
            for e in items:
                referenced |= _measure_ids(e)
            uncovered = sorted(measure_ids - referenced)
            if uncovered:
                issues.append(
                    f"row[{i}] (row_id={rid}) measures {uncovered} have no evidence; add an evidence item "
                    f"that references each measure id (e.g. '(SF1) fault injection ...')"
                )
    return issues


def _check_accept_assumptions(rows: List[Any]) -> List[str]:
    """An ACCEPT of a dangerous hazard must state the assumptions that justify acceptance (#6)."""
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        decision = str(_row_get(r, "safety_decision") or "").strip().upper()
        dangerous = bool(_row_get(r, "potentially_dangerous"))
        if decision == "ACCEPT" and dangerous:
            oa = _row_get(r, "open_assumptions")
            if not (isinstance(oa, list) and any(str(x).strip() for x in oa)):
                issues.append(
                    f"row[{i}] (row_id={rid}) accepts a dangerous hazard but lists no open_assumptions; "
                    f"an ACCEPT of a dangerous hazard must state the assumptions that justify acceptance"
                )
    return issues


def _check_risk_factors(rows: List[Any]) -> List[str]:
    """Verify each L3 row's factor levels resolve and initial_risk is numeric."""
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        for f in FACTORS:
            if factor_value(f, _row_get(r, f)) is None:
                issues.append(f"row[{i}] (row_id={rid}) invalid risk level for {f}: {_row_get(r, f)!r}")
        ir = _row_get(r, "initial_risk")
        if not isinstance(ir, (int, float)):
            issues.append(f"row[{i}] (row_id={rid}) initial_risk not computed")
    return issues


# Guidewords whose failure is "silent" — it hides/under-reports a road user, so at L3
# (pre-mitigation, no monitor yet) it cannot plausibly have low PND/PNM.
_SILENT_GUIDEWORDS = {
    "no", "less", "late", "frozen", "uncertain_but_confident", "valid_but_unsafe",
    "unmonitored", "misordered", "inverted", "wrong",
}


def _check_silent_failure_risk(rows: List[Any]) -> List[str]:
    """A silent failure that hides a VRU with serious/fatal severity cannot be low-risk at L3.

    For such rows, PND and PNM must be at least medium (pre-mitigation), which keeps the
    computed risk realistic instead of defaulting to ACCEPTABLE/ACCEPT (#2/#5)."""
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        gw = str(_row_get(r, "guideword") or "").strip().lower()
        if gw not in _SILENT_GUIDEWORDS:
            continue
        s = factor_value("S", _row_get(r, "S"))
        if s is None or s < 0.6:  # only serious/fatal severity
            continue
        pnd = factor_value("PND", _row_get(r, "PND"))
        pnm = factor_value("PNM", _row_get(r, "PNM"))
        if pnd is not None and pnd < 0.4:
            issues.append(
                f"row[{i}] (row_id={rid}) PND is too low for a silent '{gw}' failure that hides a "
                f"vulnerable road user with serious/fatal severity; at L3 (no monitor yet) PND must be "
                f"medium or higher"
            )
        if pnm is not None and pnm < 0.4:
            issues.append(
                f"row[{i}] (row_id={rid}) PNM is too low for a silent '{gw}' failure with serious/fatal "
                f"severity at L3 (no dedicated mitigation yet); PNM must be medium or higher"
            )
    return issues


def _check_safety_decision(rows: List[Any]) -> List[str]:
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        sd = str(_row_get(r, "safety_decision") or "").strip().upper()
        if sd not in SAFETY_DECISIONS:
            issues.append(f"row[{i}] (row_id={rid}) invalid safety_decision: {_row_get(r, 'safety_decision')!r}")
    return issues


# ---------------------------------------------------------------------------
# Public validators — called by the phase engine validate nodes.
# Each returns (rows, ok, issues).
# ---------------------------------------------------------------------------

def validate_l1_payload(payload: Any, expected_guidewords: List[str]) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL1Row)
    issues += _check_non_empty(rows, "failure_mode")
    issues += _check_guideword_coverage(rows, expected_guidewords)
    issues += _check_duplicates(rows, ["guideword", "failure_mode"])
    issues += _check_distinct(rows, "failure_mode", "failure mode")
    issues += _check_min_words(rows, "failure_mode", L1_MIN_WORDS, "failure_mode")
    issues += _check_no_meta_text(rows, ["failure_mode"])
    return rows, len(issues) == 0, issues


def validate_l2_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL2Row)
    issues += _check_non_empty(rows, "hazardous_behavior")
    issues += _check_non_empty(rows, "potential_harm")
    issues += _check_specific_hazard(rows)
    issues += _check_distinct(rows, "hazardous_behavior", "hazardous behavior")
    issues += _check_specific_text(rows, "potential_harm", _VAGUE_HARM_PHRASES, "potential_harm")
    issues += _check_no_meta_text(rows, ["hazardous_behavior", "potential_harm"])
    return rows, len(issues) == 0, issues


def validate_l3_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL3Row)
    issues += _check_risk_factors(rows)
    issues += _check_silent_failure_risk(rows)
    issues += _check_no_meta_text(rows, ["risk_rationale"])
    return rows, len(issues) == 0, issues


def validate_l4_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL4Row)
    issues += _check_safety_decision(rows)
    issues += _check_decision_consistency(rows)
    issues += _check_no_meta_text(rows, ["acceptance_rationale"])
    return rows, len(issues) == 0, issues


def _check_list_non_empty(rows: List[Any], field: str) -> List[str]:
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        v = _row_get(r, field)
        if not isinstance(v, list) or not [x for x in v if str(x).strip()]:
            issues.append(f"row[{i}] (row_id={rid}) '{field}' must be a non-empty list")
    return issues


_GOAL_TAG = re.compile(r'^\[?\s*SG\s*(\d+)\s*\]?\s*[-.:]?\s*', re.I)
_MEASURE_FIELDS = ("respecifications", "safety_functions", "passive_operational_measures")


def _goal_tag(s: Any) -> int | None:
    m = _GOAL_TAG.match(str(s).strip())
    return int(m.group(1)) if m else None


def _check_goal_coverage(rows: List[Any]) -> List[str]:
    """Every L5 safety goal must be addressed by >=1 L6 measure (ISO 26262 traceability).

    When measures carry [SGk] tags, require each goal 1..N to be tagged. If the model
    produced no tags at all, fall back to requiring at least as many measures as goals
    (so the bounded repair loop can still converge instead of spinning)."""
    issues: List[str] = []
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        goals = _row_get(r, "ai_safety_goals")
        ng = len([g for g in goals if str(g).strip()]) if isinstance(goals, list) else 0

        measures: List[str] = []
        for f in _MEASURE_FIELDS:
            v = _row_get(r, f)
            if isinstance(v, list):
                measures += [m for m in v if str(m).strip()]

        if not measures:
            issues.append(f"row[{i}] (row_id={rid}) has no measures across R/SF/P")
            continue
        if ng < 1:
            continue

        covered = {t for t in (_goal_tag(m) for m in measures) if t}
        if covered:
            missing = [k for k in range(1, ng + 1) if k not in covered]
            if missing:
                issues.append(
                    f"row[{i}] (row_id={rid}) safety goals {missing} have no measure; "
                    f"add a [SGk]-tagged measure for each uncovered goal"
                )
        elif len(measures) < ng:
            issues.append(
                f"row[{i}] (row_id={rid}) has {len(measures)} measures for {ng} safety goals; "
                f"each goal needs at least one [SGk]-tagged measure"
            )
    return issues


def _check_residual_factors(rows: List[Any]) -> List[str]:
    """Verify residual factor levels resolve, residual_risk is numeric, and residual <= initial."""
    issues: List[str] = []
    rm_factor_keys = {f: f"residual_{f}" for f in FACTORS}
    for i, r in enumerate(rows):
        rid = _row_get(r, "row_id")
        for f in FACTORS:
            rk = rm_factor_keys[f]
            rv = factor_value(f, _row_get(r, rk))
            if rv is None:
                issues.append(f"row[{i}] (row_id={rid}) invalid residual level for {rk}: {_row_get(r, rk)!r}")
                continue
            iv = factor_value(f, _row_get(r, f))
            if iv is not None and rv > iv:
                issues.append(f"row[{i}] (row_id={rid}) residual {f} riskier than initial")
        rr = _row_get(r, "residual_risk")
        if not isinstance(rr, (int, float)):
            issues.append(f"row[{i}] (row_id={rid}) residual_risk not computed")
    return issues


def validate_l5_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL5Row)
    issues += _check_list_non_empty(rows, "ai_safety_goals")
    issues += _check_specific_goals(rows)
    issues += _check_no_meta_text(rows, ["ai_safety_goals"])
    return rows, len(issues) == 0, issues


def validate_l6_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL6Row)
    issues += _check_goal_coverage(rows)
    issues += _check_no_meta_text(rows, ["respecifications", "safety_functions", "passive_operational_measures"])
    return rows, len(issues) == 0, issues


def validate_l7_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL7Row)
    issues += _check_residual_factors(rows)
    issues += _check_no_meta_text(rows, ["residual_rationale"])
    return rows, len(issues) == 0, issues


def validate_l8_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL8Row)
    issues += _check_list_non_empty(rows, "evidence")
    issues += _check_evidence(rows)
    issues += _check_accept_assumptions(rows)
    issues += _check_no_meta_text(rows, ["evidence", "open_assumptions"])
    return rows, len(issues) == 0, issues


def is_exportable_row(row: Any) -> bool:
    """Final export gate (FIX 8): drop rows that are broken or carry meta text.

    Enforces the hard invariants only (so a valid worksheet is never emptied out):
    valid L1, no meta text in any field, and — for dangerous rows — non-empty hazard/harm,
    a computed initial_risk, decision⇄risk_status consistency, and goals present for
    non-ACCEPT decisions. Goal→measure and measure→evidence linkage are enforced by the
    per-phase validators + repair loop, not by silently dropping rows here.
    Non-dangerous rows (legitimately L1/L2 only) pass.
    """
    if not is_valid_l1_row(row):
        return False

    for k in ("failure_mode", "hazardous_behavior", "potential_harm", "risk_rationale",
              "acceptance_rationale", "residual_rationale"):
        if _is_meta_text(_row_get(row, k)):
            return False
    for k in ("ai_safety_goals", "respecifications", "safety_functions",
              "passive_operational_measures", "evidence", "open_assumptions"):
        v = _row_get(row, k)
        if isinstance(v, list) and any(_is_meta_text(x) for x in v):
            return False

    if not bool(_row_get(row, "potentially_dangerous")):
        return True  # non-dangerous rows carry only L1/L2 fields — that's fine

    if not str(_row_get(row, "hazardous_behavior") or "").strip():
        return False
    if not str(_row_get(row, "potential_harm") or "").strip():
        return False
    if not isinstance(_row_get(row, "initial_risk"), (int, float)):
        return False

    status = str(_row_get(row, "risk_status") or "").strip().upper()
    decision = str(_row_get(row, "safety_decision") or "").strip().upper()
    if status == "ACCEPTABLE" and decision and decision != "ACCEPT":
        return False
    if status and status not in ("INCOMPLETE", "ACCEPTABLE") and decision == "ACCEPT":
        return False
    if decision in ("IMPROVE", "RESTRICT", "INVESTIGATE"):
        goals = _row_get(row, "ai_safety_goals")
        if not (isinstance(goals, list) and any(str(x).strip() for x in goals)):
            return False
    return True


def build_validator_report(stage: str, issues: List[str], hint: str | None = None) -> Dict[str, Any]:
    """Build a compact report the reviewer can consume."""
    return {"stage": stage, "errors": issues, "hint": hint or ""}
