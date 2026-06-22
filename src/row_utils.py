# src/row_utils.py
"""Shared utility functions for row ID handling, dict conversion, and merge operations.

This module is the single source of truth for:
  - Converting between Pydantic models and plain dicts
  - Row ID suffix extraction and sequential ID assignment
  - Merging regenerated row subsets back into the full row list
  - Unwrapping LLM output that may be wrapped in common dict keys

All row-matching across pipeline stages is done by numeric suffix (the part
after the dash in "L1-5"), NOT by prefix, because each stage uses its own
prefix (L1, L2, L3) and LLMs sometimes use the wrong one.
"""
from __future__ import annotations
import json
import logging
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Unmistakable reviewer/validator/repair "meta" text that must NEVER appear in a
# worksheet field. Kept deliberately narrow (multi-word phrases) so it cannot match
# legitimate HAZOP content such as a safety goal beginning "Ensure a cyclist ...".
META_PATTERNS = re.compile(
    r"ensure consistency"
    r"|consistency between (the )?risk"
    r"|risk[_ ]status and safety[_ ]decision"
    r"|fix (the )?decision"
    r"|check (the )?risk[_ ]status"
    r"|repair instruction"
    r"|reviewer (comment|hint|message|suggestion)"
    r"|validator (hint|message|comment)"
    r"|validation instruction",
    re.I,
)


def _is_meta_text(value: Any) -> bool:
    """True if ``value`` is a string containing unmistakable reviewer/validator meta text."""
    return isinstance(value, str) and bool(META_PATTERNS.search(value))


def _scrub_meta(value: Any) -> Any:
    """Last-resort output guard: blank any field value that is reviewer/validator meta text.

    Strings matching META_PATTERNS become ""; list items are scrubbed and dropped if
    they become empty; other values pass through unchanged.
    """
    if isinstance(value, list):
        out = []
        for x in value:
            s = _scrub_meta(x)
            if not (isinstance(s, str) and not s.strip()):
                out.append(s)
        return out
    if _is_meta_text(value):
        return ""
    return value

# Common dict keys that LLMs wrap row arrays in.  Used by _ensure_rows_list,
# validators.py, and llm_client.py to unwrap {"rows": [...]} → [...].
WRAPPER_KEYS = (
    "rows", "data", "items", "result", "output",
    "deviations", "results", "causes", "effects",
    "failure_modes", "hazards", "measures", "evidence",
    "rows_l1", "rows_l2", "rows_l3", "rows_l4",
    "rows_l5", "rows_l6", "rows_l7", "rows_l8",
)


# ---------------------------------------------------------------------------
# Input normalisation — LLM output arrives in many shapes (bare list, wrapped
# in {"rows": [...]}, single dict, list-of-lists).  These helpers coerce any
# shape into List[Dict].
# ---------------------------------------------------------------------------

def _ensure_rows_list(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept either a list[dict], a dict wrapping the list under common keys,
    or a single row dict (wrap into a list). Also handles list-of-lists by
    flattening one level.
    """
    if obj is None:
        return []

    # 1) Unwrap common dict wrappers
    if isinstance(obj, dict):
        for key in WRAPPER_KEYS:
            val = obj.get(key)
            if isinstance(val, list):
                obj = val
                break
        else:
            # 2) If it's a single row dict (LLM sometimes returns one row), wrap it
            row_sig_sets = [
                # classic HAZOP shapes (legacy)
                {"function", "guideword", "deviation"},
                {"function", "guideword", "cause"},
                {"function", "guideword", "effect"},
                # AI-HAZOP-8800 shapes — LLM returns row_id + new fields
                {"row_id", "failure_mode"},
                {"row_id", "hazardous_behavior"},
                {"row_id", "safety_decision"},
                {"guideword", "failure_mode"},
            ]
            obj_keys = set(obj.keys())
            if any(sig.issubset(obj_keys) for sig in row_sig_sets):
                obj = [obj]
            else:
                # 3) Generic fallback: pick the first value that is a list of dicts
                for v in obj.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        obj = v
                        break
                else:
                    # single-value dict holding a list
                    if len(obj) == 1:
                        sole_val = next(iter(obj.values()))
                        if isinstance(sole_val, list):
                            obj = sole_val

    # 4) If still a dict after all fallback attempts, wrap it as a single-row list.
    # This is a permissive fallback that matches the old graph_full.py behavior.
    if isinstance(obj, dict):
        return [obj]

    # 5) Must be list now - if not, log warning and return empty list
    if not isinstance(obj, list):
        logger.warning(
            "_ensure_rows_list: received non-list/non-dict type %s, returning empty list",
            type(obj).__name__
        )
        return []

    # 6) Flatten one level if list-of-lists
    if obj and isinstance(obj[0], list):
        flat: List[Any] = []
        for sub in obj:
            if isinstance(sub, list):
                flat.extend(sub)
            else:
                flat.append(sub)
        obj = flat

    # 7) Ensure list[dict] (filter out any stray non-dict items)
    return [x for x in obj if isinstance(x, dict)]


def _row_get(row: Any, key: str) -> Any:
    """Get a key from a row, handling both dicts and Pydantic models."""
    if isinstance(row, dict):
        return row.get(key)
    if hasattr(row, "model_dump"):
        return row.model_dump().get(key)
    if hasattr(row, "dict"):
        return row.dict().get(key)
    if hasattr(row, key):
        return getattr(row, key)
    return None


def _row_to_dict(row: Any) -> Dict[str, Any]:
    """Convert a row (dict or Pydantic model) to a plain dict."""
    if isinstance(row, dict):
        return dict(row)
    if hasattr(row, "model_dump"):
        return row.model_dump()
    if hasattr(row, "dict"):
        return row.dict()
    try:
        return dict(row.__dict__)
    except Exception:
        return {}


def _ensure_row_ids(prefix: str, rows: List[Dict[str, Any]]) -> None:
    """Assign every row a sequential ``{prefix}-{i}`` id.

    This guarantees that L1 rows always carry ``L1-*``, L2 rows ``L2-*``, and
    L3 rows ``L3-*``.  Previous behaviour preserved LLM-assigned ids when they
    were unique, but LLMs frequently assign the *wrong* prefix (e.g. ``L1-0``
    for an L2 row).  Downstream code matches rows across stages by suffix, so
    mixed prefixes caused suffix collisions and data loss in the GUI.
    """
    for i, r in enumerate(rows):
        r["row_id"] = f"{prefix}-{i}"


def _suffix(row_id: Optional[str]) -> str:
    """Extract the numeric suffix from a row_id (e.g., 'L1-5' -> '5')."""
    if row_id is None:
        return ""
    try:
        return str(row_id).split("-", 1)[1]
    except (IndexError, AttributeError):
        return str(row_id) if row_id else ""


def _pydantic_to_dicts(rows: List[Any]) -> List[Dict[str, Any]]:
    """Convert list of Pydantic models/dicts to plain dicts.

    This helper consolidates the repeated pattern found in validate nodes
    where rows need to be converted back to plain dicts for consistent
    state handling.
    """
    result: List[Dict[str, Any]] = []
    for r in rows:
        if hasattr(r, "model_dump"):
            result.append(r.model_dump())
        elif hasattr(r, "dict"):
            result.append(r.dict())
        else:
            result.append(dict(r))
    return result


def _merge_rag_notes(base_notes: str, rag_notes: str) -> str:
    """Merge base notes with RAG context.

    This helper consolidates the repeated pattern found in init nodes
    where base notes and RAG context need to be combined.

    Args:
        base_notes: The original notes/context string.
        rag_notes: RAG-retrieved context to append.

    Returns:
        Combined notes string with RAG context prefixed by marker.
    """
    base = (base_notes or "").strip()
    rag = (rag_notes or "").strip()
    if not rag:
        return base
    return (base + "\n\n" if base else "") + "[RAG CONTEXT]\n" + rag


# =============================================================================
# List-of-dicts coercion — ensures every element is a dict (parses JSON
# strings that some LLMs return as individual row values).
# =============================================================================

def _ensure_list_of_dicts(rows: Any, stage: str = "UNKNOWN") -> List[Dict[str, Any]]:
    """
    Ensure we have a List[Dict]; if rows are strings, attempt json.loads per row.
    Flattens one level if rows contains sublists.
    """
    # unwrap wrappers if needed
    if not isinstance(rows, list):
        rows = _ensure_rows_list(rows)

    # flatten one level if list-of-lists
    if rows and isinstance(rows[0], list):
        flat: List[Any] = []
        for sub in rows:
            if isinstance(sub, list):
                flat.extend(sub)
            else:
                flat.append(sub)
        rows = flat

    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        if isinstance(r, dict):
            out.append(r)
        elif isinstance(r, str):
            try:
                out.append(json.loads(r))
            except Exception:
                raise ValueError(f"[{stage}] Row[{i}] is a string but not JSON-decodable: {r!r}")
        else:
            raise ValueError(f"[{stage}] Row[{i}] must be dict or JSON-string, got {type(r).__name__}")
    return out


# =============================================================================
# Merge helpers — two strategies for combining regenerated rows back into the
# full row list:
#   1. Suffix-based (_merge_subset_by_suffix): match by row_id numeric suffix.
#      Used when row_ids are stable and only a subset was regenerated.
#   2. Key-based (_merge_full_by_key): match by composite field values
#      (e.g. function+guideword).  Used when row_ids may have shifted.
# =============================================================================

def _row_key_by_fields(row: Any, fields: List[str]) -> str:
    """Build a key from row fields for matching during merge operations."""
    return "||".join(str(_row_get(row, f) or "").strip() for f in fields)


def _rows_equal_by_suffix(a: List[Any], b: List[Any]) -> bool:
    """Check if two row lists are equal by comparing rows with matching suffixes."""
    def m(rows):
        d = {}
        for r in rows:
            rid = str(_row_get(r, "row_id") or "")
            d[_suffix(rid)] = json.dumps(_row_to_dict(r), sort_keys=True)
        return d
    return m(a) == m(b)


def _merge_full_by_key(
    old_rows: List[Any],
    regenerated_rows: List[Any],
    target_suffixes: List[str],
    key_fields: List[str],
) -> List[Dict[str, Any]]:
    """
    Merge regenerated rows into old rows based on key fields.
    Only rows with suffixes in target_suffixes are replaced.
    """
    regen_map = {_row_key_by_fields(r, key_fields): _row_to_dict(r) for r in regenerated_rows}
    tset = set(target_suffixes)
    merged = []
    for old in old_rows:
        od = _row_to_dict(old)
        rid = str(od.get("row_id", ""))
        if _suffix(rid) in tset:
            key = _row_key_by_fields(od, key_fields)
            cand = regen_map.get(key)
            if cand is not None:
                nd = cand.copy()
                nd["row_id"] = rid
                merged.append(nd)
                continue
        merged.append(od)
    return merged


def _merge_subset_by_suffix(
    current_rows: List[Dict[str, Any]],
    new_subset: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Merge new_subset into current_rows by matching row_id suffixes.
    Rows in current_rows are replaced by matching rows from new_subset.
    """
    cur = deepcopy(current_rows)
    by_suf_new = {_suffix(r["row_id"]): r for r in new_subset if r.get("row_id")}
    out: List[Dict[str, Any]] = []
    for r in cur:
        s = _suffix(str(r.get("row_id") or ""))
        out.append(deepcopy(by_suf_new.get(s, r)))
    return out


def _pick_rows_by_suffix(
    rows: List[Dict[str, Any]],
    target_suffixes: List[str],
) -> List[Dict[str, Any]]:
    """Pick rows whose row_id suffix is in target_suffixes."""
    tset = set(target_suffixes)
    return [deepcopy(r) for r in rows if _suffix(str(r.get("row_id", ""))) in tset]
