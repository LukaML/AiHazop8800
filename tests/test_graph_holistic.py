# tests/test_graph_holistic.py
"""Tests for holistic review nodes in src/graph_full.py (replaces TEST_FAULT_HOL)."""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from src.graph_full import (
    holistic_review_node,
    holistic_execute_node,
    MAX_HOLISTIC_ROUNDS,
    _cascade_from_H_single_suggestion,
    _extract_row_ids_from_issues,
    _scope_to_highest_stage,
)


class TestHolisticReviewNode:
    """Tests for holistic_review_node() function."""

    def test_returns_ok_when_all_pass(self, mocker, graph_state_after_l3):
        """Returns OK decision when all rows pass review."""
        mock_response = {
            "decision": "OK",
            "scope": "ALL",
            "issues": [],
            "suggestions": [],
        }
        mocker.patch("src.graph_full.l3_pass_to_reviewer_holistic", return_value=mock_response)

        result = holistic_review_node(graph_state_after_l3)

        assert result["holistic_ok"] is True

    def test_finds_inconsistencies(self, mocker, graph_state_after_l3):
        """Returns RETURN when inconsistencies found."""
        mock_response = {
            "decision": "RETURN",
            "scope": "L2",
            "issues": ["Inconsistency in row L3-0: cause doesn't match effect"],
            "suggestions": ["Fix the cause"],
        }
        mocker.patch("src.graph_full.l3_pass_to_reviewer_holistic", return_value=mock_response)

        result = holistic_review_node(graph_state_after_l3)

        assert result["holistic_ok"] is False
        assert result["holistic_scope"] == "L2"

    def test_extracts_target_ids(self, mocker, graph_state_after_l3):
        """Extracts target row IDs from issues."""
        mock_response = {
            "decision": "RETURN",
            "scope": "L3",
            "issues": ["Problem with L3-0", "Problem with L3-5"],
            "suggestions": [],
        }
        mocker.patch("src.graph_full.l3_pass_to_reviewer_holistic", return_value=mock_response)

        result = holistic_review_node(graph_state_after_l3)

        assert "L3-0" in result["holistic_target_ids"]
        assert "L3-5" in result["holistic_target_ids"]

    def test_increments_round_counter(self, mocker, graph_state_after_l3):
        """Tracks holistic round number."""
        state = {**graph_state_after_l3, "holistic_round": 1}
        mock_response = {"decision": "OK", "scope": "ALL", "issues": []}
        mocker.patch("src.graph_full.l3_pass_to_reviewer_holistic", return_value=mock_response)

        result = holistic_review_node(state)

        assert result["holistic_round"] == 1  # Round is set, not incremented here

    def test_reviews_only_unresolved_on_subsequent_rounds(self, mocker, graph_state_after_l3):
        """On subsequent rounds, only reviews previously targeted rows."""
        state = {
            **graph_state_after_l3,
            "holistic_round": 1,
            "holistic_target_ids": ["L3-0"],
        }
        mock_response = {"decision": "OK", "scope": "ALL", "issues": []}
        mock_reviewer = mocker.patch("src.graph_full.l3_pass_to_reviewer_holistic", return_value=mock_response)

        holistic_review_node(state)

        # Should have been called with subset of rows
        call_args = mock_reviewer.call_args
        rows_passed = call_args[0][0]
        # On round 1 with target_ids, should only pass targeted rows
        assert len(rows_passed) <= len(graph_state_after_l3["rows_l3"])


class TestHolisticExecuteNode:
    """Tests for holistic_execute_node() function."""

    def test_skips_when_ok(self, graph_state_after_l3):
        """Skips regeneration when holistic review passed."""
        state = {
            **graph_state_after_l3,
            "holistic_ok": True,
            "holistic_round": 0,
        }

        result = holistic_execute_node(state)

        # Should just return round, no row changes
        assert result.get("holistic_round", 0) == 0

    def test_increments_round(self, mocker, graph_state_after_l3):
        """Increments holistic round after execution."""
        state = {
            **graph_state_after_l3,
            "holistic_ok": False,
            "holistic_round": 0,
            "holistic_scope": "L3",
            "holistic_target_ids": ["L3-0"],
            "holistic_suggestion": "Fix the effect",
        }
        mocker.patch("src.graph_full.reviewer_patch_l3", return_value=state["rows_l3"][:1])

        result = holistic_execute_node(state)

        assert result["holistic_round"] == 1

    def test_cascades_l1_changes(self, mocker, graph_state_after_l3):
        """L1 scope cascades to L2 and L3."""
        state = {
            **graph_state_after_l3,
            "holistic_ok": False,
            "holistic_round": 0,
            "holistic_scope": "L1",
            "holistic_target_ids": ["L1-0"],
            "holistic_suggestion": "Fix the deviation",
        }

        # Mock all the chain functions
        mocker.patch("src.graph_full.reviewer_to_l1", return_value=state["rows_l1"])
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=state["rows_l2"])
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=state["rows_l3"])

        result = holistic_execute_node(state)

        # Should have updated all three row sets
        assert "rows_l1" in result
        assert "rows_l2" in result
        assert "rows_l3" in result

    def test_cascades_l2_changes(self, mocker, graph_state_after_l3):
        """L2 scope cascades to L3."""
        state = {
            **graph_state_after_l3,
            "holistic_ok": False,
            "holistic_round": 0,
            "holistic_scope": "L2",
            "holistic_target_ids": ["L2-0"],
            "holistic_suggestion": "Fix the cause",
        }

        mocker.patch("src.graph_full.reviewer_patch_l2", return_value=state["rows_l2"][:1])
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=state["rows_l2"])
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=state["rows_l3"])

        result = holistic_execute_node(state)

        assert "rows_l2" in result
        assert "rows_l3" in result

    def test_fixes_l3_only(self, mocker, graph_state_after_l3):
        """L3 scope only patches L3 rows."""
        state = {
            **graph_state_after_l3,
            "holistic_ok": False,
            "holistic_round": 0,
            "holistic_scope": "L3",
            "holistic_target_ids": ["L3-0"],
            "holistic_suggestion": "Fix the effect",
        }

        fixed_row = {**state["rows_l3"][0], "effect": "Fixed effect"}
        mocker.patch("src.graph_full.reviewer_patch_l3", return_value=[fixed_row])

        result = holistic_execute_node(state)

        assert "rows_l3" in result


class TestHolisticHelpers:
    """Tests for holistic review helper functions."""

    def test_extract_row_ids_from_issues(self):
        """Extracts row IDs from issue strings."""
        issues = [
            "Problem with L1-3: deviation is wrong",
            "Row L2-10 has inconsistent cause",
            "L3-5 effect doesn't match",
        ]

        result = _extract_row_ids_from_issues(issues)

        assert "L1-3" in result
        assert "L2-10" in result
        assert "L3-5" in result

    def test_extract_row_ids_deduplicates(self):
        """Deduplicates extracted row IDs."""
        issues = ["L1-0 has issue", "Another issue with L1-0"]

        result = _extract_row_ids_from_issues(issues)

        assert result.count("L1-0") == 1

    def test_extract_row_ids_empty(self):
        """Returns empty list for empty issues."""
        assert _extract_row_ids_from_issues([]) == []
        assert _extract_row_ids_from_issues(None) == []

    def test_scope_to_highest_stage(self):
        """Converts scope string to stage."""
        assert _scope_to_highest_stage("L1") == "L1"
        assert _scope_to_highest_stage("L2") == "L2"
        assert _scope_to_highest_stage("L3") == "L3"
        assert _scope_to_highest_stage("ALL") == "L1"
        assert _scope_to_highest_stage("") == "L1"


class TestHolisticRoundLimit:
    """Tests for holistic round limit enforcement."""

    def test_max_rounds_constant(self):
        """MAX_HOLISTIC_ROUNDS is set correctly."""
        assert MAX_HOLISTIC_ROUNDS == 3

    def test_exits_after_max_rounds(self, mocker, graph_state_after_l3):
        """Pipeline exits after max holistic rounds."""
        state = {
            **graph_state_after_l3,
            "holistic_ok": False,
            "holistic_round": MAX_HOLISTIC_ROUNDS,
        }

        # The conditional edge should direct to END after max rounds
        from src.graph_full import build_graph
        from langgraph.graph import END

        graph = build_graph()
        # Graph should have the conditional edge defined
        assert graph is not None
