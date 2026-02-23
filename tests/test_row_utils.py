# tests/test_row_utils.py
"""Tests for src/row_utils.py utility functions."""
import pytest
from typing import Dict, Any, List

from src.row_utils import (
    _ensure_rows_list,
    _merge_full_by_key,
    _merge_subset_by_suffix,
    _pick_rows_by_suffix,
    _row_get,
    _row_to_dict,
    _rows_equal_by_suffix,
    _ensure_row_ids,
    _suffix,
    _pydantic_to_dicts,
    _merge_rag_notes,
)


class TestEnsureRowsList:
    """Tests for _ensure_rows_list() function."""

    def test_none_returns_empty_list(self):
        """None input returns empty list."""
        assert _ensure_rows_list(None) == []

    def test_list_passthrough(self):
        """Plain list of dicts passes through."""
        rows = [{"a": 1}, {"b": 2}]
        result = _ensure_rows_list(rows)
        assert result == rows

    def test_dict_wrapper_rows_key(self):
        """Dict with 'rows' key is unwrapped."""
        wrapped = {"rows": [{"a": 1}]}
        assert _ensure_rows_list(wrapped) == [{"a": 1}]

    def test_dict_wrapper_data_key(self):
        """Dict with 'data' key is unwrapped."""
        wrapped = {"data": [{"a": 1}]}
        assert _ensure_rows_list(wrapped) == [{"a": 1}]

    def test_dict_wrapper_items_key(self):
        """Dict with 'items' key is unwrapped."""
        wrapped = {"items": [{"a": 1}]}
        assert _ensure_rows_list(wrapped) == [{"a": 1}]

    def test_dict_wrapper_deviations_key(self):
        """Dict with 'deviations' key is unwrapped."""
        wrapped = {"deviations": [{"function": "Test", "guideword": "no", "deviation": "d"}]}
        result = _ensure_rows_list(wrapped)
        assert len(result) == 1
        assert result[0]["function"] == "Test"

    def test_single_row_dict_wrapped(self):
        """Single row dict is wrapped into a list."""
        single = {"function": "Test", "guideword": "no", "deviation": "d"}
        result = _ensure_rows_list(single)
        assert len(result) == 1
        assert result[0] == single

    def test_list_of_lists_flattened(self):
        """Nested list-of-lists is flattened one level."""
        nested = [[{"a": 1}], [{"b": 2}]]
        result = _ensure_rows_list(nested)
        assert result == [{"a": 1}, {"b": 2}]

    def test_filters_non_dict_items(self):
        """Non-dict items in list are filtered out."""
        mixed = [{"a": 1}, "string", 123, {"b": 2}]
        result = _ensure_rows_list(mixed)
        assert result == [{"a": 1}, {"b": 2}]

    def test_sole_value_dict_unwrap(self):
        """Single-key dict with list value is unwrapped."""
        wrapped = {"custom_key": [{"a": 1}]}
        result = _ensure_rows_list(wrapped)
        assert result == [{"a": 1}]

    # === Edge cases added for bug fix verification ===

    def test_unknown_wrapper_with_list_of_dicts(self):
        """Unknown dict key containing list of dicts should be unwrapped.

        This tests the generic fallback path where the LLM returns a response
        with an unexpected wrapper key (not in the known list).
        """
        # Multi-key dict where one value is a list of dicts
        wrapped = {"status": "ok", "custom_response": [{"a": 1}, {"b": 2}]}
        result = _ensure_rows_list(wrapped)
        assert result == [{"a": 1}, {"b": 2}]

    def test_dict_no_list_values_wrapped_as_single_row(self):
        """Dict with no list values should be wrapped as single-row list.

        This is the permissive fallback behavior - if LLM returns a dict
        that doesn't match known patterns, we assume it's a single row
        rather than silently returning empty.
        """
        # Dict with scalar values only (no lists) - should be wrapped as [dict]
        result = _ensure_rows_list({"status": "ok", "count": 5})
        assert result == [{"status": "ok", "count": 5}]

    def test_non_list_non_dict_returns_empty(self):
        """Non-list, non-dict types (string, int, etc.) return empty list."""
        assert _ensure_rows_list("string") == []
        assert _ensure_rows_list(123) == []
        assert _ensure_rows_list(45.67) == []

    def test_gemini_style_response(self):
        """Handles Gemini-style response with nested structure."""
        # Gemini sometimes returns candidates[0].content structure
        gemini_like = {
            "candidates": [{"content": {"parts": [{"text": "..."}]}}],
            "rows": [{"function": "F1", "guideword": "no", "deviation": "d1"}]
        }
        result = _ensure_rows_list(gemini_like)
        # Should unwrap the "rows" key even with other nested structures
        assert len(result) == 1
        assert result[0]["function"] == "F1"

    def test_ollama_style_response_single_row(self):
        """Handles Ollama-style response returning single row without wrapper."""
        # Some models return just the row dict without wrapping
        ollama_like = {
            "function": "TestFunc",
            "guideword": "more",
            "deviation": "Excessive output",
            "row_id": "L1-0"
        }
        result = _ensure_rows_list(ollama_like)
        assert len(result) == 1
        assert result[0]["function"] == "TestFunc"

    def test_empty_list_in_wrapper(self):
        """Empty list in wrapper returns empty list."""
        wrapped = {"rows": []}
        result = _ensure_rows_list(wrapped)
        assert result == []

    def test_deeply_nested_list_not_flattened(self):
        """Only one level of list nesting is flattened."""
        # Three-level nesting - only outermost list-of-lists is flattened
        nested = [[[{"a": 1}]]]
        result = _ensure_rows_list(nested)
        # After flatten: [[{"a": 1}]], then filter non-dicts: []
        assert result == []


class TestRowGet:
    """Tests for _row_get() function."""

    def test_dict_access(self):
        """Gets value from dict row."""
        row = {"key": "value"}
        assert _row_get(row, "key") == "value"

    def test_dict_missing_key(self):
        """Returns None for missing key in dict."""
        row = {"other": "value"}
        assert _row_get(row, "key") is None

    def test_pydantic_model_access(self, sample_l1_pydantic):
        """Gets value from Pydantic model."""
        assert _row_get(sample_l1_pydantic, "function") == "Braking"

    def test_pydantic_model_missing_key(self, sample_l1_pydantic):
        """Returns None for missing attribute on Pydantic model."""
        assert _row_get(sample_l1_pydantic, "nonexistent") is None


class TestRowToDict:
    """Tests for _row_to_dict() function."""

    def test_dict_passthrough(self):
        """Dict is returned as copy."""
        row = {"a": 1, "b": 2}
        result = _row_to_dict(row)
        assert result == row
        # Should be a copy, not same object
        assert result is not row

    def test_pydantic_model_conversion(self, sample_l1_pydantic):
        """Pydantic model is converted to dict."""
        result = _row_to_dict(sample_l1_pydantic)
        assert isinstance(result, dict)
        assert result["function"] == "Braking"
        assert result["guideword"] == "no"


class TestEnsureRowIds:
    """Tests for _ensure_row_ids() function."""

    def test_assigns_missing_ids(self):
        """Missing row_ids are assigned with prefix."""
        rows = [{"a": 1}, {"b": 2}]
        _ensure_row_ids("L1", rows)
        assert rows[0]["row_id"] == "L1-0"
        assert rows[1]["row_id"] == "L1-1"

    def test_always_assigns_sequential_ids(self):
        """Existing row_ids are overwritten with sequential prefix-i IDs."""
        rows = [{"row_id": "L1-5", "a": 1}, {"b": 2}]
        _ensure_row_ids("L1", rows)
        assert rows[0]["row_id"] == "L1-0"
        assert rows[1]["row_id"] == "L1-1"

    def test_different_prefix(self):
        """Different prefixes work correctly."""
        rows = [{"a": 1}]
        _ensure_row_ids("L2", rows)
        assert rows[0]["row_id"] == "L2-0"

    def test_empty_string_id_replaced(self):
        """Empty string row_id is replaced."""
        rows = [{"row_id": "", "a": 1}]
        _ensure_row_ids("L1", rows)
        assert rows[0]["row_id"] == "L1-0"

    def test_whitespace_id_replaced(self):
        """Whitespace-only row_id is replaced."""
        rows = [{"row_id": "  ", "a": 1}]
        _ensure_row_ids("L1", rows)
        assert rows[0]["row_id"] == "L1-0"

    def test_forces_correct_prefix(self):
        """Rows with wrong prefix get reassigned to correct prefix."""
        rows = [{"row_id": "L3-5", "a": 1}, {"row_id": "L3-6", "b": 2}]
        _ensure_row_ids("L1", rows)
        assert rows[0]["row_id"] == "L1-0"
        assert rows[1]["row_id"] == "L1-1"

    def test_duplicate_suffix_different_prefix(self):
        """Rows like 'L1-0' and 'L3-0' both get unique sequential IDs."""
        rows = [
            {"row_id": "L1-0", "a": 1},
            {"row_id": "L3-0", "b": 2},
            {"row_id": "L2-0", "c": 3},
        ]
        _ensure_row_ids("L1", rows)
        assert rows[0]["row_id"] == "L1-0"
        assert rows[1]["row_id"] == "L1-1"
        assert rows[2]["row_id"] == "L1-2"


class TestSuffix:
    """Tests for _suffix() function."""

    def test_extracts_numeric_suffix(self):
        """Extracts suffix from row_id."""
        assert _suffix("L1-5") == "5"
        assert _suffix("L2-10") == "10"
        assert _suffix("L3-0") == "0"

    def test_none_returns_empty(self):
        """None input returns empty string."""
        assert _suffix(None) == ""

    def test_no_dash_returns_original(self):
        """No dash returns original string."""
        assert _suffix("nodash") == "nodash"

    def test_empty_string(self):
        """Empty string returns empty."""
        assert _suffix("") == ""

    def test_multiple_dashes(self):
        """Multiple dashes - returns everything after first dash."""
        assert _suffix("L1-5-extra") == "5-extra"


class TestPydanticToDicts:
    """Tests for _pydantic_to_dicts() function."""

    def test_empty_list(self):
        """Empty list returns empty list."""
        assert _pydantic_to_dicts([]) == []

    def test_dict_list_passthrough(self):
        """List of dicts is converted (copied)."""
        rows = [{"a": 1}, {"b": 2}]
        result = _pydantic_to_dicts(rows)
        assert result == rows

    def test_pydantic_list_conversion(self, sample_l1_pydantic, sample_l2_pydantic):
        """List of Pydantic models is converted to dicts."""
        rows = [sample_l1_pydantic, sample_l2_pydantic]
        result = _pydantic_to_dicts(rows)
        assert len(result) == 2
        assert all(isinstance(r, dict) for r in result)
        assert result[0]["function"] == "Braking"
        assert result[1]["cause"] == "Hydraulic failure"

    def test_mixed_list_conversion(self, sample_l1_pydantic):
        """Mixed list of dicts and Pydantic models is converted."""
        rows = [sample_l1_pydantic, {"manual": "dict"}]
        result = _pydantic_to_dicts(rows)
        assert len(result) == 2
        assert result[0]["function"] == "Braking"
        assert result[1]["manual"] == "dict"


class TestMergeRagNotes:
    """Tests for _merge_rag_notes() function."""

    def test_no_rag_returns_base(self):
        """Empty RAG notes returns base notes."""
        result = _merge_rag_notes("Base notes", "")
        assert result == "Base notes"

    def test_no_base_returns_rag(self):
        """Empty base notes returns RAG with marker."""
        result = _merge_rag_notes("", "RAG content")
        assert result == "[RAG CONTEXT]\nRAG content"

    def test_both_merged(self):
        """Both base and RAG are merged with marker."""
        result = _merge_rag_notes("Base notes", "RAG content")
        assert result == "Base notes\n\n[RAG CONTEXT]\nRAG content"

    def test_whitespace_trimmed(self):
        """Whitespace is trimmed from inputs."""
        result = _merge_rag_notes("  Base  ", "  RAG  ")
        assert result == "Base\n\n[RAG CONTEXT]\nRAG"

    def test_none_handled_as_empty(self):
        """None values handled as empty strings."""
        result = _merge_rag_notes(None, None)
        assert result == ""

    def test_both_empty(self):
        """Both empty returns empty."""
        result = _merge_rag_notes("", "")
        assert result == ""


# ============================================================================
#                     MERGE / PICK HELPER TESTS
# ============================================================================


class TestRowsEqualBySuffix:
    """Tests for _rows_equal_by_suffix()."""

    def test_empty_lists_equal(self):
        assert _rows_equal_by_suffix([], []) is True

    def test_identical_rows_equal(self):
        a = [{"row_id": "L1-0", "function": "F1", "deviation": "d"}]
        b = [{"row_id": "L1-0", "function": "F1", "deviation": "d"}]
        assert _rows_equal_by_suffix(a, b) is True

    def test_different_content_not_equal(self):
        a = [{"row_id": "L1-0", "function": "F1", "deviation": "d1"}]
        b = [{"row_id": "L1-0", "function": "F1", "deviation": "d2"}]
        assert _rows_equal_by_suffix(a, b) is False

    def test_same_suffix_different_prefix_compared_by_content(self):
        a = [{"row_id": "L1-5", "function": "F1", "deviation": "d"}]
        b = [{"row_id": "L2-5", "function": "F1", "deviation": "d"}]
        # Different row_id means different dict content, so NOT equal
        assert _rows_equal_by_suffix(a, b) is False

    def test_different_number_of_rows(self):
        a = [{"row_id": "L1-0", "x": "a"}]
        b = [{"row_id": "L1-0", "x": "a"}, {"row_id": "L1-1", "x": "b"}]
        assert _rows_equal_by_suffix(a, b) is False

    def test_order_independent_by_suffix(self):
        """Rows matched by suffix, not by position."""
        a = [
            {"row_id": "L1-0", "v": "a"},
            {"row_id": "L1-1", "v": "b"},
        ]
        b = [
            {"row_id": "L1-1", "v": "b"},
            {"row_id": "L1-0", "v": "a"},
        ]
        assert _rows_equal_by_suffix(a, b) is True


class TestMergeFullByKey:
    """Tests for _merge_full_by_key()."""

    def test_empty_regen_returns_old(self):
        old = [{"row_id": "L1-0", "function": "F1", "guideword": "no", "deviation": "d"}]
        result = _merge_full_by_key(old, [], ["0"], ["function", "guideword"])
        assert result == old

    def test_empty_targets_returns_old(self):
        old = [{"row_id": "L1-0", "function": "F1", "guideword": "no", "deviation": "old"}]
        regen = [{"row_id": "L1-0", "function": "F1", "guideword": "no", "deviation": "new"}]
        result = _merge_full_by_key(old, regen, [], ["function", "guideword"])
        assert result[0]["deviation"] == "old"

    def test_targeted_row_replaced(self):
        old = [
            {"row_id": "L1-0", "function": "F1", "guideword": "no", "deviation": "old"},
            {"row_id": "L1-1", "function": "F1", "guideword": "more", "deviation": "keep"},
        ]
        regen = [
            {"row_id": "X-0", "function": "F1", "guideword": "no", "deviation": "new"},
        ]
        result = _merge_full_by_key(old, regen, ["0"], ["function", "guideword"])
        assert result[0]["deviation"] == "new"
        assert result[1]["deviation"] == "keep"

    def test_merged_row_keeps_old_row_id(self):
        old = [{"row_id": "L1-0", "function": "F1", "guideword": "no", "deviation": "old"}]
        regen = [{"row_id": "REGEN-99", "function": "F1", "guideword": "no", "deviation": "new"}]
        result = _merge_full_by_key(old, regen, ["0"], ["function", "guideword"])
        assert result[0]["row_id"] == "L1-0"
        assert result[0]["deviation"] == "new"

    def test_no_key_match_preserves_old(self):
        old = [{"row_id": "L1-0", "function": "F1", "guideword": "no", "deviation": "old"}]
        regen = [{"row_id": "L1-0", "function": "F2", "guideword": "no", "deviation": "new"}]
        result = _merge_full_by_key(old, regen, ["0"], ["function", "guideword"])
        assert result[0]["deviation"] == "old"

    def test_non_targeted_rows_preserved(self):
        old = [
            {"row_id": "L1-0", "function": "F1", "guideword": "no", "deviation": "d0"},
            {"row_id": "L1-1", "function": "F1", "guideword": "more", "deviation": "d1"},
            {"row_id": "L1-2", "function": "F2", "guideword": "no", "deviation": "d2"},
        ]
        regen = [
            {"row_id": "L1-1", "function": "F1", "guideword": "more", "deviation": "new_d1"},
        ]
        result = _merge_full_by_key(old, regen, ["1"], ["function", "guideword"])
        assert len(result) == 3
        assert result[0]["deviation"] == "d0"
        assert result[1]["deviation"] == "new_d1"
        assert result[2]["deviation"] == "d2"


class TestPickRowsBySuffix:
    """Tests for _pick_rows_by_suffix()."""

    def test_empty_rows(self):
        assert _pick_rows_by_suffix([], ["0", "1"]) == []

    def test_empty_suffixes(self):
        rows = [{"row_id": "L1-0", "v": "a"}]
        assert _pick_rows_by_suffix(rows, []) == []

    def test_correct_rows_selected(self):
        rows = [
            {"row_id": "L1-0", "v": "a"},
            {"row_id": "L1-1", "v": "b"},
            {"row_id": "L1-2", "v": "c"},
        ]
        result = _pick_rows_by_suffix(rows, ["0", "2"])
        assert len(result) == 2
        assert result[0]["v"] == "a"
        assert result[1]["v"] == "c"

    def test_returns_deep_copies(self):
        rows = [{"row_id": "L1-0", "v": "original"}]
        result = _pick_rows_by_suffix(rows, ["0"])
        result[0]["v"] = "mutated"
        assert rows[0]["v"] == "original"

    def test_no_matching_suffixes(self):
        rows = [{"row_id": "L1-0", "v": "a"}, {"row_id": "L1-1", "v": "b"}]
        result = _pick_rows_by_suffix(rows, ["99"])
        assert result == []

    def test_different_prefix_same_suffix(self):
        rows = [{"row_id": "L2-5", "v": "x"}]
        result = _pick_rows_by_suffix(rows, ["5"])
        assert len(result) == 1
        assert result[0]["v"] == "x"
