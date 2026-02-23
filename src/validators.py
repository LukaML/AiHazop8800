"""Pydantic-based validation for each pipeline stage output (L1, L2, L3).

Validation flow per stage:
  1. Unwrap LLM output from common dict wrappers ({"rows": [...]})
  2. Parse each row into the corresponding Pydantic model
  3. Run structural checks: non-empty required fields, no duplicates
  4. For L1: verify full guideword coverage (all 11 guidewords per function)
"""
from __future__ import annotations

from typing import Dict, List, Tuple, Any, Type
import json

from pydantic import ValidationError

from .models import (
    HazopL1Row,
    HazopL2Row,
    HazopL3Row,
    Guideword,
)
from .row_utils import _row_get, WRAPPER_KEYS

__all__ = [
    "parse_rows_safely",
    "validate_l1_payload",
    "validate_l2_payload",
    "validate_l3_payload",
    "build_validator_report",
]

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

# ---------------------------------------------------------------------------
# Core parsing: unwrap LLM dict wrappers → parse each row into Pydantic
# model → collect validation issues.  Rows that fail validation are kept
# as plain dicts (with issues logged) so downstream stages can still
# reference them for repair.
# ---------------------------------------------------------------------------

def parse_rows_safely(payloads: Any, model: Type[Any]) -> Tuple[List[Any], List[str]]:
    issues: List[str] = []
    rows: List[Any] = []

    payloads = _unwrap_to_list(payloads)  # unwrap dict wrappers

    if not isinstance(payloads, list):
        issues.append(f"Top-level payload is not a list, got: {type(payloads).__name__}")
        return rows, issues

    for i, payload in enumerate(payloads):
        # Accept string rows like '{"function":"..."}'
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except Exception as e:
                issues.append(f"row[{i}] is a string but not JSON-decodable: {e}")
                continue

        if not isinstance(payload, dict):
            issues.append(f"row[{i}] is not a dict after normalization (got {type(payload).__name__}).")
            continue

        # --------- Patch: inject row_id if missing (common L1 issue) ---------
        if "row_id" not in payload:
            # Use a stable prefix per model class
            if model is HazopL1Row:
                payload["row_id"] = f"L1-{i}"
            elif model is HazopL2Row:
                payload["row_id"] = f"L2-{i}"
            elif model is HazopL3Row:
                payload["row_id"] = f"L3-{i}"
        # ---------------------------------------------------------------------

        try:
            # Parse into Pydantic model for validation.  Valid models are appended.
            rows.append(model(**payload))
        except ValidationError as ve:
            # If validation fails, record the issue but do not drop the row.
            # Instead, append the original payload (with row_id injected) so that
            # downstream processing can preserve the row and mark it for review.
            issues.append(f"row[{i}] validation error: {ve.errors()}")
            # Append a copy of the payload dict so that it remains in the list.  Use
            # `dict(payload)` to avoid accidentally mutating the original payload.
            d = dict(payload)
            # Coerce row_id to string — the LLM sometimes returns integers
            if "row_id" in d and not isinstance(d["row_id"], str):
                d["row_id"] = str(d["row_id"])
            rows.append(d)

    return rows, issues


# ---------------------------------------------------------------------------
# Structural quality checks — duplicate detection, non-empty field
# verification, function membership, and full guideword coverage for L1.
# ---------------------------------------------------------------------------

_GUIDEWORDS_SET = {g.value for g in Guideword}

def _check_duplicates(rows: List[Any], fields: List[str]) -> List[str]:
    seen = set()
    dups = []
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
            if rid:
                issues.append(f"row[{i}] (row_id={rid}) has empty '{field}'")
            else:
                issues.append(f"row[{i}] has empty '{field}'")
    return issues

def _check_functions_belong(rows: List[Any], expected_functions: List[str]) -> List[str]:
    if not expected_functions:
        return []
    allowed = set(expected_functions)
    issues = []
    for i, r in enumerate(rows):
        fn = _row_get(r, "function")
        if fn not in allowed:
            issues.append(f"row[{i}] function not in expected set: {fn}")
    return issues


def _norm_gw(gw: Any) -> str:
    return gw.value if isinstance(gw, Guideword) else str(gw)


def _check_full_guideword_coverage(rows: List[Any], expected_functions: List[str]) -> List[str]:
    """
    Verify that for each expected function, all guidewords are represented at least once.
    This helper supports both Pydantic model instances and plain dictionaries.

    Args:
        rows: A list of L1 rows (HazopL1Row or dict) containing function and guideword fields.
        expected_functions: The list of functions for which full guideword coverage is required.

    Returns:
        A list of error strings indicating which functions are missing guidewords.
    """
    issues: List[str] = []
    by_fn = {fn: set() for fn in expected_functions} if expected_functions else {}

    for r in rows:
        fn = _row_get(r, "function")
        gw_raw = _row_get(r, "guideword")
        gw = _norm_gw(gw_raw)  # normalize to string
        if expected_functions and fn in by_fn:
            by_fn[fn].add(gw)

    for fn, seen in by_fn.items():
        missing = _GUIDEWORDS_SET - seen  # both sets of strings now
        if missing:
            issues.append(f"function '{fn}' missing guidewords: {sorted(list(missing))}")
    return issues


# ---------------------------------------------------------------------------
# Public validators — called by graph_full.py validate nodes.  Each returns
# (rows, ok, issues) where ok=True means the stage output is acceptable.
# ---------------------------------------------------------------------------

def validate_l1_payload(payload: Any, expected_functions: List[str]) -> Tuple[List[HazopL1Row], bool, List[str]]:
    """
    Returns (rows, ok, issues)
    """
    rows, issues = parse_rows_safely(payload, HazopL1Row)

    # structural & quality checks
    issues += _check_non_empty(rows, "deviation")
    issues += _check_functions_belong(rows, expected_functions)
    issues += _check_full_guideword_coverage(rows, expected_functions)
    # duplicates across (function, guideword, deviation)
    issues += _check_duplicates(rows, ["function", "guideword", "deviation"])

    ok = len(issues) == 0
    return rows, ok, issues


def validate_l2_payload(payload: Any) -> Tuple[List[HazopL2Row], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, HazopL2Row)

    # L2 must have cause
    issues += _check_non_empty(rows, "cause")
    # duplicates across (function, guideword, deviation, cause)
    issues += _check_duplicates(rows, ["function", "guideword", "deviation", "cause"])

    ok = len(issues) == 0
    return rows, ok, issues


def validate_l3_payload(payload: Any) -> Tuple[List[HazopL3Row], bool, List[str]]:
    rows, issues = parse_rows_safely(payload, HazopL3Row)

    # L3 must have effect
    issues += _check_non_empty(rows, "effect")
    # Phase 1: L3 must have the boolean triage indicator
    issues += _check_non_empty(rows, "potentially_dangerous")
    # duplicates across (function, guideword, deviation, cause, effect)
    issues += _check_duplicates(rows, ["function", "guideword", "deviation", "cause", "effect"])

    ok = len(issues) == 0
    return rows, ok, issues


def build_validator_report(stage: str, issues: List[str], hint: str | None = None) -> Dict[str, Any]:
    """
    Build a compact report the Reviewer can consume.
    """
    return {
        "stage": stage,
        "errors": issues,
        "hint": hint or "",
    }