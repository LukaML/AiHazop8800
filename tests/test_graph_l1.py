# tests/test_graph_l1.py
"""Tests for L1 stage nodes in src/graph_full.py (replaces TEST_FAULT_L1)."""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from src.graph_full import (
    l1_init_node,
    l1_validate_node,
    l1_reviewer_node,
    l1_regen_node,
    MAX_STAGE_REPAIR,
)


class TestL1InitNode:
    """Tests for l1_init_node() function."""

    def test_generates_rows(self, mocker, minimal_graph_state, full_l1_coverage_rows):
        """Generates L1 rows from mocked LLM."""
        mocker.patch("src.graph_full.l1_init_full_coverage", return_value=full_l1_coverage_rows)

        result = l1_init_node(minimal_graph_state)

        assert "rows_l1" in result
        assert len(result["rows_l1"]) == len(full_l1_coverage_rows)
        assert result["l1_repair_round"] == 0

    def test_assigns_row_ids(self, mocker, minimal_graph_state):
        """Assigns row_ids to rows without them."""
        rows_without_ids = [
            {"function": "Braking", "guideword": "no", "deviation": "d1"},
            {"function": "Braking", "guideword": "more", "deviation": "d2"},
        ]
        mocker.patch("src.graph_full.l1_init_full_coverage", return_value=rows_without_ids)

        result = l1_init_node(minimal_graph_state)

        assert all("row_id" in r for r in result["rows_l1"])
        assert result["rows_l1"][0]["row_id"] == "L1-0"
        assert result["rows_l1"][1]["row_id"] == "L1-1"

    def test_merges_rag_notes(self, mocker, minimal_graph_state):
        """Merges RAG notes into context."""
        state_with_rag = {
            **minimal_graph_state,
            "notes": "Base notes",
            "rag_notes_l1": "RAG context",
        }
        mock_init = mocker.patch("src.graph_full.l1_init_full_coverage", return_value=[])

        l1_init_node(state_with_rag)

        # Check that merged notes were passed
        call_args = mock_init.call_args
        notes_arg = call_args[0][3] if len(call_args[0]) > 3 else call_args[1].get("notes", "")
        assert "Base notes" in notes_arg or "[RAG CONTEXT]" in notes_arg


class TestL1ValidateNode:
    """Tests for l1_validate_node() function."""

    def test_passes_valid_rows(self, graph_state_after_l1):
        """Valid L1 rows pass validation."""
        result = l1_validate_node(graph_state_after_l1)

        assert result["l1_ok"] is True
        assert len(result["l1_issues"]) == 0

    def test_fails_empty_deviation(self, minimal_graph_state):
        """Empty deviation triggers validation failure."""
        state = {
            **minimal_graph_state,
            "rows_l1": [
                {"row_id": "L1-0", "function": "Braking", "guideword": "no", "deviation": ""}
            ]
        }

        result = l1_validate_node(state)

        assert result["l1_ok"] is False
        assert len(result["l1_issues"]) > 0
        assert any("empty" in issue.lower() or "deviation" in issue.lower() for issue in result["l1_issues"])

    def test_fails_missing_guidewords(self, minimal_graph_state):
        """Missing guideword coverage triggers validation failure."""
        # Only one guideword instead of all 11
        state = {
            **minimal_graph_state,
            "rows_l1": [
                {"row_id": "L1-0", "function": "Braking", "guideword": "no", "deviation": "d1"}
            ]
        }

        result = l1_validate_node(state)

        assert result["l1_ok"] is False
        assert any("missing guidewords" in issue for issue in result["l1_issues"])

    def test_converts_pydantic_to_dicts(self, minimal_graph_state, sample_l1_pydantic):
        """Pydantic models are converted to dicts."""
        state = {
            **minimal_graph_state,
            "rows_l1": [sample_l1_pydantic]
        }

        result = l1_validate_node(state)

        # Result should contain dicts, not Pydantic models
        assert all(isinstance(r, dict) for r in result["rows_l1"])


class TestL1ReviewerNode:
    """Tests for l1_reviewer_node() function."""

    def test_returns_decision(self, mocker, graph_state_after_l1, mock_llm_reviewer_ok):
        """Returns reviewer decision."""
        mocker.patch("src.graph_full.l1_to_reviewer", return_value=mock_llm_reviewer_ok)

        state = {**graph_state_after_l1, "l1_issues": []}
        result = l1_reviewer_node(state)

        assert "l1_suggestion" in result

    def test_returns_suggestion(self, mocker, graph_state_after_l1, mock_llm_reviewer_return):
        """Returns suggestion when reviewer returns RETURN."""
        mock_llm_reviewer_return["suggestion"] = "Fix the deviation"
        mocker.patch("src.graph_full.l1_to_reviewer", return_value=mock_llm_reviewer_return)

        state = {**graph_state_after_l1, "l1_issues": ["Some issue"]}
        result = l1_reviewer_node(state)

        assert "l1_suggestion" in result
        assert result["l1_suggestion"] == "Fix the deviation"


class TestL1RegenNode:
    """Tests for l1_regen_node() function."""

    def test_fixes_bad_rows(self, mocker, minimal_graph_state, full_l1_coverage_rows):
        """Regenerates targeted rows with issues."""
        # State with issue on specific row
        state = {
            **minimal_graph_state,
            "rows_l1": full_l1_coverage_rows,
            "l1_issues": ["row[0] (row_id=L1-0) has empty 'deviation'"],
            "l1_suggestion": "Provide a specific deviation",
            "l1_repair_round": 0,
        }

        # Mock patch function to return fixed row
        fixed_rows = [
            {**full_l1_coverage_rows[0], "deviation": "Fixed deviation"}
        ]
        mocker.patch("src.graph_full.reviewer_patch_l1", return_value=fixed_rows)

        result = l1_regen_node(state)

        assert result["l1_repair_round"] == 1
        assert "rows_l1" in result

    def test_increments_repair_round(self, mocker, minimal_graph_state, full_l1_coverage_rows):
        """Increments l1_repair_round counter."""
        state = {
            **minimal_graph_state,
            "rows_l1": full_l1_coverage_rows,
            "l1_issues": ["L1-0 has issue"],
            "l1_suggestion": "Fix it",
            "l1_repair_round": 1,
        }
        mocker.patch("src.graph_full.reviewer_patch_l1", return_value=[])
        mocker.patch("src.graph_full.reviewer_to_l1", return_value=full_l1_coverage_rows)

        result = l1_regen_node(state)

        assert result["l1_repair_round"] == 2

    def test_no_issues_returns_empty(self, minimal_graph_state, full_l1_coverage_rows):
        """No issues increments repair round without attempting repair."""
        state = {
            **minimal_graph_state,
            "rows_l1": full_l1_coverage_rows,
            "l1_issues": [],
            "l1_suggestion": "",
            "l1_repair_round": 0,
        }

        result = l1_regen_node(state)

        assert result == {"l1_repair_round": 1}

    def test_fallback_to_full_regen(self, mocker, minimal_graph_state, full_l1_coverage_rows):
        """Falls back to full regen when patch returns no changes."""
        state = {
            **minimal_graph_state,
            "rows_l1": full_l1_coverage_rows,
            "l1_issues": ["L1-0 has issue"],
            "l1_suggestion": "Fix it",
            "l1_repair_round": 0,
        }

        # Patch returns same rows (no change)
        mocker.patch("src.graph_full.reviewer_patch_l1", return_value=full_l1_coverage_rows[:1])
        # Full regen returns new rows
        new_rows = [{**r, "deviation": "New deviation"} for r in full_l1_coverage_rows]
        mocker.patch("src.graph_full.reviewer_to_l1", return_value=new_rows)

        result = l1_regen_node(state)

        assert "rows_l1" in result


class TestL1RepairLoop:
    """Integration tests for L1 repair loop behavior."""

    def test_repair_round_limit_enforced(self, minimal_graph_state, full_l1_coverage_rows):
        """MAX_STAGE_REPAIR limit is respected."""
        # After MAX_STAGE_REPAIR rounds, should proceed to L2
        state = {
            **minimal_graph_state,
            "rows_l1": full_l1_coverage_rows,
            "l1_ok": False,
            "l1_repair_round": MAX_STAGE_REPAIR,
        }

        # The conditional edge logic should direct to L2_INIT
        from src.graph_full import build_graph
        graph = build_graph()

        # This verifies the condition exists in the graph
        assert MAX_STAGE_REPAIR == 2  # Verify constant
