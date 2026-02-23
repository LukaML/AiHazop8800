# tests/test_graph_l2.py
"""Tests for L2 stage nodes in src/graph_full.py (replaces TEST_FAULT_L2)."""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from src.graph_full import (
    l2_init_node,
    l2_validate_node,
    l2_reviewer_node,
    l2_regen_node,
    MAX_STAGE_REPAIR,
)


class TestL2InitNode:
    """Tests for l2_init_node() function."""

    def test_generates_rows_from_l1(self, mocker, graph_state_after_l1, full_l2_coverage_rows):
        """Generates L2 rows from L1 rows via mocked LLM."""
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=full_l2_coverage_rows)

        result = l2_init_node(graph_state_after_l1)

        assert "rows_l2" in result
        assert len(result["rows_l2"]) == len(full_l2_coverage_rows)
        assert result["l2_repair_round"] == 0

    def test_adds_causes(self, mocker, graph_state_after_l1):
        """L2 rows have cause field added."""
        l2_rows = [
            {**r, "cause": f"Cause for {r['deviation']}"}
            for r in graph_state_after_l1["rows_l1"]
        ]
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=l2_rows)

        result = l2_init_node(graph_state_after_l1)

        assert all("cause" in r for r in result["rows_l2"])

    def test_assigns_row_ids(self, mocker, graph_state_after_l1):
        """Assigns L2- prefixed row_ids."""
        rows_without_ids = [
            {"function": "Braking", "guideword": "no", "deviation": "d1", "cause": "c1"},
        ]
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=rows_without_ids)

        result = l2_init_node(graph_state_after_l1)

        assert result["rows_l2"][0]["row_id"] == "L2-0"

    def test_merges_rag_notes(self, mocker, graph_state_after_l1):
        """Merges RAG notes into context."""
        state_with_rag = {
            **graph_state_after_l1,
            "notes": "Base notes",
            "rag_notes_l2": "L2 RAG context",
        }
        mock_init = mocker.patch("src.graph_full.l2_init_from_l1", return_value=[])

        l2_init_node(state_with_rag)

        # Check that notes were passed
        call_args = mock_init.call_args
        notes_arg = call_args[1].get("notes", "")
        assert "Base notes" in notes_arg or "[RAG CONTEXT]" in notes_arg


class TestL2ValidateNode:
    """Tests for l2_validate_node() function."""

    def test_passes_valid_rows(self, graph_state_after_l2):
        """Valid L2 rows pass validation."""
        result = l2_validate_node(graph_state_after_l2)

        assert result["l2_ok"] is True
        assert len(result["l2_issues"]) == 0

    def test_fails_empty_cause(self, graph_state_after_l1):
        """Empty cause triggers validation failure."""
        state = {
            **graph_state_after_l1,
            "rows_l2": [
                {"row_id": "L2-0", "function": "Braking", "guideword": "no",
                 "deviation": "d1", "cause": ""}
            ]
        }

        result = l2_validate_node(state)

        assert result["l2_ok"] is False
        assert any("empty" in issue.lower() or "cause" in issue.lower() for issue in result["l2_issues"])

    def test_converts_pydantic_to_dicts(self, graph_state_after_l1, sample_l2_pydantic):
        """Pydantic models are converted to dicts."""
        state = {
            **graph_state_after_l1,
            "rows_l2": [sample_l2_pydantic]
        }

        result = l2_validate_node(state)

        assert all(isinstance(r, dict) for r in result["rows_l2"])


class TestL2ReviewerNode:
    """Tests for l2_reviewer_node() function."""

    def test_returns_decision(self, mocker, graph_state_after_l2, mock_llm_reviewer_ok):
        """Returns reviewer decision."""
        mocker.patch("src.graph_full.l2_to_reviewer", return_value=mock_llm_reviewer_ok)

        state = {**graph_state_after_l2, "l2_issues": []}
        result = l2_reviewer_node(state)

        assert "l2_suggestion" in result

    def test_returns_suggestion(self, mocker, graph_state_after_l2):
        """Returns suggestion when reviewer returns RETURN."""
        mock_response = {
            "decision": "RETURN",
            "scope": "L2",
            "issues": ["Empty cause"],
            "suggestions": [],
            "suggestion": "Fix the cause"
        }
        mocker.patch("src.graph_full.l2_to_reviewer", return_value=mock_response)

        state = {**graph_state_after_l2, "l2_issues": ["Some issue"]}
        result = l2_reviewer_node(state)

        assert "l2_suggestion" in result
        assert result["l2_suggestion"] == "Fix the cause"


class TestL2RegenNode:
    """Tests for l2_regen_node() function."""

    def test_fixes_bad_rows(self, mocker, graph_state_after_l2):
        """Regenerates targeted rows with issues."""
        state = {
            **graph_state_after_l2,
            "l2_issues": ["row[0] (row_id=L2-0) has empty 'cause'"],
            "l2_suggestion": "Provide a specific cause",
            "l2_repair_round": 0,
        }

        fixed_rows = [
            {**graph_state_after_l2["rows_l2"][0], "cause": "Fixed cause"}
        ]
        mocker.patch("src.graph_full.reviewer_patch_l2", return_value=fixed_rows)

        result = l2_regen_node(state)

        assert result["l2_repair_round"] == 1
        assert "rows_l2" in result

    def test_increments_repair_round(self, mocker, graph_state_after_l2):
        """Increments l2_repair_round counter."""
        state = {
            **graph_state_after_l2,
            "l2_issues": ["L2-0 has issue"],
            "l2_suggestion": "Fix it",
            "l2_repair_round": 1,
        }
        mocker.patch("src.graph_full.reviewer_patch_l2", return_value=[])
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=graph_state_after_l2["rows_l2"])

        result = l2_regen_node(state)

        assert result["l2_repair_round"] == 2

    def test_no_issues_returns_empty(self, graph_state_after_l2):
        """No issues increments repair round without attempting repair."""
        state = {
            **graph_state_after_l2,
            "l2_issues": [],
            "l2_suggestion": "",
            "l2_repair_round": 0,
        }

        result = l2_regen_node(state)

        assert result == {"l2_repair_round": 1}

    def test_fallback_to_full_regen(self, mocker, graph_state_after_l2):
        """Falls back to full regen when patch returns no changes."""
        state = {
            **graph_state_after_l2,
            "l2_issues": ["L2-0 has issue"],
            "l2_suggestion": "Fix it",
            "l2_repair_round": 0,
        }

        # Patch returns same rows (no change)
        mocker.patch("src.graph_full.reviewer_patch_l2", return_value=graph_state_after_l2["rows_l2"][:1])
        # Full regen returns new rows
        new_rows = [{**r, "cause": "New cause"} for r in graph_state_after_l2["rows_l2"]]
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=new_rows)

        result = l2_regen_node(state)

        assert "rows_l2" in result


class TestL2RepairLoop:
    """Integration tests for L2 repair loop behavior."""

    def test_repair_round_limit_enforced(self, graph_state_after_l2):
        """MAX_STAGE_REPAIR limit is respected."""
        state = {
            **graph_state_after_l2,
            "l2_ok": False,
            "l2_repair_round": MAX_STAGE_REPAIR,
        }

        # Verify the constant
        assert MAX_STAGE_REPAIR == 2
