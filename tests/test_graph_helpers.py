# tests/test_graph_helpers.py
"""Tests for pure-logic helper functions in src/graph_full.py."""
import pytest

from src.graph_full import _suffixes_from_row_ids


class TestSuffixesFromRowIds:
    """Tests for _suffixes_from_row_ids()."""

    def test_empty_list(self):
        assert _suffixes_from_row_ids([]) == []

    def test_none_input(self):
        assert _suffixes_from_row_ids(None) == []

    def test_single_id(self):
        assert _suffixes_from_row_ids(["L1-5"]) == ["5"]

    def test_multiple_ids_sorted(self):
        result = _suffixes_from_row_ids(["L1-10", "L1-2", "L1-5"])
        assert result == ["10", "2", "5"]

    def test_duplicates_removed(self):
        result = _suffixes_from_row_ids(["L1-3", "L2-3", "L3-3"])
        assert result == ["3"]

    def test_mixed_prefixes_same_suffix(self):
        result = _suffixes_from_row_ids(["L1-0", "L2-0", "L3-1", "L1-1"])
        assert result == ["0", "1"]

    def test_non_string_values_ignored(self):
        result = _suffixes_from_row_ids([None, 123, "", "L1-5"])
        assert result == ["5"]

    def test_empty_string_ignored(self):
        result = _suffixes_from_row_ids(["", "L1-2"])
        assert result == ["2"]
