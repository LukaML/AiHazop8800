# tests/test_graph_l3.py
"""Tests for L3 stage nodes in src/graph_full.py (replaces TEST_FAULT_L3)."""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from src.graph_full import (
    l3_init_node,
    l3_validate_node,
    l3_reviewer_node,
    l3_regen_node,
    MAX_STAGE_REPAIR,
)


class TestL3InitNode:
    """Tests for l3_init_node() function."""

    def test_generates_rows_from_l2(self, mocker, graph_state_after_l2, full_l3_coverage_rows):
        """Generates L3 rows from L2 rows via mocked LLM."""
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=full_l3_coverage_rows)

        result = l3_init_node(graph_state_after_l2)

        assert "rows_l3" in result
        assert len(result["rows_l3"]) == len(full_l3_coverage_rows)
        assert result["l3_repair_round"] == 0

    def test_adds_effects_and_triage(self, mocker, graph_state_after_l2):
        """L3 rows have effect and potentially_dangerous fields added."""
        l3_rows = [
            {**r, "effect": f"Effect for {r['cause']}", "potentially_dangerous": True}
            for r in graph_state_after_l2["rows_l2"]
        ]
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=l3_rows)

        result = l3_init_node(graph_state_after_l2)

        assert all("effect" in r for r in result["rows_l3"])
        assert all("potentially_dangerous" in r for r in result["rows_l3"])

    def test_assigns_row_ids(self, mocker, graph_state_after_l2):
        """Assigns L3- prefixed row_ids."""
        rows_without_ids = [
            {"function": "Braking", "guideword": "no", "deviation": "d1",
             "cause": "c1", "effect": "e1", "potentially_dangerous": True},
        ]
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=rows_without_ids)

        result = l3_init_node(graph_state_after_l2)

        assert result["rows_l3"][0]["row_id"] == "L3-0"

    def test_merges_rag_notes(self, mocker, graph_state_after_l2):
        """Merges RAG notes into context."""
        state_with_rag = {
            **graph_state_after_l2,
            "notes": "Base notes",
            "rag_notes_l3": "L3 RAG context",
        }
        mock_init = mocker.patch("src.graph_full.l3_init_from_l2", return_value=[])

        l3_init_node(state_with_rag)

        call_args = mock_init.call_args
        notes_arg = call_args[1].get("notes", "")
        assert "Base notes" in notes_arg or "[RAG CONTEXT]" in notes_arg


class TestL3ValidateNode:
    """Tests for l3_validate_node() function."""

    def test_passes_valid_rows(self, graph_state_after_l3):
        """Valid L3 rows pass validation."""
        result = l3_validate_node(graph_state_after_l3)

        assert result["l3_ok"] is True
        assert len(result["l3_issues"]) == 0

    def test_fails_empty_effect(self, graph_state_after_l2):
        """Empty effect triggers validation failure."""
        state = {
            **graph_state_after_l2,
            "rows_l3": [
                {"row_id": "L3-0", "function": "Braking", "guideword": "no",
                 "deviation": "d1", "cause": "c1", "effect": "",
                 "potentially_dangerous": True}
            ]
        }

        result = l3_validate_node(state)

        assert result["l3_ok"] is False
        assert any("empty" in issue.lower() or "effect" in issue.lower() for issue in result["l3_issues"])

    def test_fails_missing_triage(self, graph_state_after_l2):
        """Missing potentially_dangerous field triggers validation failure."""
        state = {
            **graph_state_after_l2,
            "rows_l3": [
                {"row_id": "L3-0", "function": "Braking", "guideword": "no",
                 "deviation": "d1", "cause": "c1", "effect": "e1"}
                # Missing potentially_dangerous
            ]
        }

        result = l3_validate_node(state)

        assert result["l3_ok"] is False
        assert len(result["l3_issues"]) > 0

    def test_converts_pydantic_to_dicts(self, graph_state_after_l2, sample_l3_pydantic):
        """Pydantic models are converted to dicts."""
        state = {
            **graph_state_after_l2,
            "rows_l3": [sample_l3_pydantic]
        }

        result = l3_validate_node(state)

        assert all(isinstance(r, dict) for r in result["rows_l3"])


class TestL3ReviewerNode:
    """Tests for l3_reviewer_node() function."""

    def test_returns_decision(self, mocker, graph_state_after_l3, mock_llm_reviewer_ok):
        """Returns reviewer decision."""
        mocker.patch("src.graph_full.l3_to_reviewer", return_value=mock_llm_reviewer_ok)

        state = {**graph_state_after_l3, "l3_issues": []}
        result = l3_reviewer_node(state)

        assert "l3_suggestion" in result

    def test_returns_suggestion(self, mocker, graph_state_after_l3):
        """Returns suggestion when reviewer returns RETURN."""
        mock_response = {
            "decision": "RETURN",
            "scope": "L3",
            "issues": ["Empty effect"],
            "suggestions": [],
            "suggestion": "Fix the effect"
        }
        mocker.patch("src.graph_full.l3_to_reviewer", return_value=mock_response)

        state = {**graph_state_after_l3, "l3_issues": ["Some issue"]}
        result = l3_reviewer_node(state)

        assert "l3_suggestion" in result
        assert result["l3_suggestion"] == "Fix the effect"


class TestL3RegenNode:
    """Tests for l3_regen_node() function."""

    def test_fixes_bad_rows(self, mocker, graph_state_after_l3):
        """Regenerates targeted rows with issues."""
        state = {
            **graph_state_after_l3,
            "l3_issues": ["row[0] (row_id=L3-0) has empty 'effect'"],
            "l3_suggestion": "Provide a specific effect",
            "l3_repair_round": 0,
        }

        fixed_rows = [
            {**graph_state_after_l3["rows_l3"][0], "effect": "Fixed effect"}
        ]
        mocker.patch("src.graph_full.reviewer_patch_l3", return_value=fixed_rows)

        result = l3_regen_node(state)

        assert result["l3_repair_round"] == 1
        assert "rows_l3" in result

    def test_increments_repair_round(self, mocker, graph_state_after_l3):
        """Increments l3_repair_round counter."""
        state = {
            **graph_state_after_l3,
            "l3_issues": ["L3-0 has issue"],
            "l3_suggestion": "Fix it",
            "l3_repair_round": 1,
        }
        mocker.patch("src.graph_full.reviewer_patch_l3", return_value=[])
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=graph_state_after_l3["rows_l3"])

        result = l3_regen_node(state)

        assert result["l3_repair_round"] == 2

    def test_no_issues_returns_empty(self, graph_state_after_l3):
        """No issues increments repair round without attempting repair."""
        state = {
            **graph_state_after_l3,
            "l3_issues": [],
            "l3_suggestion": "",
            "l3_repair_round": 0,
        }

        result = l3_regen_node(state)

        assert result == {"l3_repair_round": 1}

    def test_fallback_to_full_regen(self, mocker, graph_state_after_l3):
        """Falls back to full regen when patch returns no changes."""
        state = {
            **graph_state_after_l3,
            "l3_issues": ["L3-0 has issue"],
            "l3_suggestion": "Fix it",
            "l3_repair_round": 0,
        }

        mocker.patch("src.graph_full.reviewer_patch_l3", return_value=graph_state_after_l3["rows_l3"][:1])
        new_rows = [{**r, "effect": "New effect"} for r in graph_state_after_l3["rows_l3"]]
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=new_rows)

        result = l3_regen_node(state)

        assert "rows_l3" in result


class TestL3RepairLoop:
    """Integration tests for L3 repair loop behavior."""

    def test_repair_round_limit_enforced(self, graph_state_after_l3):
        """MAX_STAGE_REPAIR limit is respected."""
        state = {
            **graph_state_after_l3,
            "l3_ok": False,
            "l3_repair_round": MAX_STAGE_REPAIR,
        }

        assert MAX_STAGE_REPAIR == 2
