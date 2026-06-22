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
from .row_utils import _row_get, WRAPPER_KEYS

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
    return rows, len(issues) == 0, issues


def validate_l2_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL2Row)
    issues += _check_non_empty(rows, "hazardous_behavior")
    issues += _check_non_empty(rows, "potential_harm")
    issues += _check_specific_hazard(rows)
    issues += _check_distinct(rows, "hazardous_behavior", "hazardous behavior")
    return rows, len(issues) == 0, issues


def validate_l3_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL3Row)
    issues += _check_risk_factors(rows)
    return rows, len(issues) == 0, issues


def validate_l4_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL4Row)
    issues += _check_safety_decision(rows)
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
    return rows, len(issues) == 0, issues


def validate_l6_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL6Row)
    issues += _check_goal_coverage(rows)
    return rows, len(issues) == 0, issues


def validate_l7_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL7Row)
    issues += _check_residual_factors(rows)
    return rows, len(issues) == 0, issues


def validate_l8_payload(payload: Any) -> Tuple[List[Any], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, AIHazopL8Row)
    issues += _check_list_non_empty(rows, "evidence")
    return rows, len(issues) == 0, issues


def build_validator_report(stage: str, issues: List[str], hint: str | None = None) -> Dict[str, Any]:
    """Build a compact report the reviewer can consume."""
    return {"stage": stage, "errors": issues, "hint": hint or ""}
