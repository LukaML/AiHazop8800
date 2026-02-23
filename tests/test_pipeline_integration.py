# tests/test_pipeline_integration.py
"""Integration tests for the full HAZOP pipeline with mocked LLM."""
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from src.graph_full import build_graph, build_full_graph, HazopGraphState


class TestPipelineHappyPath:
    """Tests for the happy path through the pipeline."""

    def test_full_pipeline_completes(self, mocker, sample_functions, sample_guideword_map,
                                      full_l1_coverage_rows, full_l2_coverage_rows,
                                      full_l3_coverage_rows):
        """Full pipeline completes with mocked LLM responses."""
        # Mock all LLM chain functions
        mocker.patch("src.graph_full.l1_init_full_coverage", return_value=full_l1_coverage_rows)
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=full_l2_coverage_rows)
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=full_l3_coverage_rows)

        # Mock reviewers to return OK
        ok_response = {"decision": "OK", "scope": "ALL", "issues": [], "suggestions": []}
        mocker.patch("src.graph_full.l1_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l2_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l3_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l3_pass_to_reviewer_holistic", return_value=ok_response)

        # Build and run graph
        graph = build_graph()
        app = graph.compile()

        initial_state = {
            "functions": sample_functions,
            "guideword_map": sample_guideword_map,
            "max_devs_per_gw": 1,
            "rag_enabled": False,
        }

        result = app.invoke(initial_state)

        # Verify final state
        assert "rows_l1" in result
        assert "rows_l2" in result
        assert "rows_l3" in result
        assert len(result["rows_l1"]) > 0
        assert len(result["rows_l2"]) > 0
        assert len(result["rows_l3"]) > 0

    def test_build_full_graph_alias(self):
        """build_full_graph() is alias for build_graph()."""
        g1 = build_graph()
        g2 = build_full_graph()
        # Both should return StateGraph instances
        assert type(g1) == type(g2)


class TestPipelineWithL1Repair:
    """Tests for pipeline with L1 repair cycle."""

    def test_l1_repair_then_success(self, mocker, sample_functions, sample_guideword_map,
                                     full_l1_coverage_rows, full_l2_coverage_rows,
                                     full_l3_coverage_rows):
        """Pipeline repairs L1 issues then completes."""
        # First call returns bad rows, second returns good rows
        bad_l1_rows = [{**r, "deviation": ""} for r in full_l1_coverage_rows[:2]]
        bad_l1_rows.extend(full_l1_coverage_rows[2:])

        l1_call_count = [0]

        def mock_l1_init(*args, **kwargs):
            l1_call_count[0] += 1
            if l1_call_count[0] == 1:
                return bad_l1_rows
            return full_l1_coverage_rows

        mocker.patch("src.graph_full.l1_init_full_coverage", side_effect=mock_l1_init)
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=full_l2_coverage_rows)
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=full_l3_coverage_rows)

        # L1 reviewer returns RETURN first, then OK
        return_response = {"decision": "RETURN", "scope": "L1", "issues": ["L1-0 empty"],
                          "suggestions": ["Fix it"], "suggestion": "Fix deviation"}
        ok_response = {"decision": "OK", "scope": "ALL", "issues": [], "suggestions": []}

        review_call_count = [0]

        def mock_l1_review(*args, **kwargs):
            review_call_count[0] += 1
            if review_call_count[0] == 1:
                return return_response
            return ok_response

        mocker.patch("src.graph_full.l1_to_reviewer", side_effect=mock_l1_review)
        mocker.patch("src.graph_full.reviewer_to_l1", return_value=full_l1_coverage_rows)
        mocker.patch("src.graph_full.reviewer_patch_l1", return_value=full_l1_coverage_rows[:2])
        mocker.patch("src.graph_full.l2_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l3_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l3_pass_to_reviewer_holistic", return_value=ok_response)

        graph = build_graph()
        app = graph.compile()

        initial_state = {
            "functions": sample_functions,
            "guideword_map": sample_guideword_map,
            "max_devs_per_gw": 1,
            "rag_enabled": False,
        }

        result = app.invoke(initial_state)

        # Pipeline should complete even after repair
        assert "rows_l3" in result


class TestPipelineWithL2Repair:
    """Tests for pipeline with L2 repair cycle."""

    def test_l2_repair_then_success(self, mocker, sample_functions, sample_guideword_map,
                                     full_l1_coverage_rows, full_l2_coverage_rows,
                                     full_l3_coverage_rows):
        """Pipeline repairs L2 issues then completes."""
        mocker.patch("src.graph_full.l1_init_full_coverage", return_value=full_l1_coverage_rows)
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=full_l3_coverage_rows)

        # L2 init returns bad rows first
        bad_l2_rows = [{**r, "cause": ""} for r in full_l2_coverage_rows[:1]]
        bad_l2_rows.extend(full_l2_coverage_rows[1:])

        l2_call_count = [0]

        def mock_l2_init(*args, **kwargs):
            l2_call_count[0] += 1
            if l2_call_count[0] == 1:
                return bad_l2_rows
            return full_l2_coverage_rows

        mocker.patch("src.graph_full.l2_init_from_l1", side_effect=mock_l2_init)

        ok_response = {"decision": "OK", "scope": "ALL", "issues": [], "suggestions": []}
        return_response = {"decision": "RETURN", "scope": "L2", "issues": ["L2-0 empty cause"],
                          "suggestions": [], "suggestion": "Fix cause"}

        mocker.patch("src.graph_full.l1_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l2_to_reviewer", side_effect=[return_response, ok_response])
        mocker.patch("src.graph_full.reviewer_patch_l2", return_value=full_l2_coverage_rows[:1])
        mocker.patch("src.graph_full.l3_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l3_pass_to_reviewer_holistic", return_value=ok_response)

        graph = build_graph()
        app = graph.compile()

        initial_state = {
            "functions": sample_functions,
            "guideword_map": sample_guideword_map,
            "max_devs_per_gw": 1,
            "rag_enabled": False,
        }

        result = app.invoke(initial_state)

        assert "rows_l3" in result


class TestPipelineWithL3Repair:
    """Tests for pipeline with L3 repair cycle."""

    def test_l3_repair_then_success(self, mocker, sample_functions, sample_guideword_map,
                                     full_l1_coverage_rows, full_l2_coverage_rows,
                                     full_l3_coverage_rows):
        """Pipeline repairs L3 issues then completes."""
        mocker.patch("src.graph_full.l1_init_full_coverage", return_value=full_l1_coverage_rows)
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=full_l2_coverage_rows)

        # L3 init returns bad rows first
        bad_l3_rows = [{**r, "effect": ""} for r in full_l3_coverage_rows[:1]]
        bad_l3_rows.extend(full_l3_coverage_rows[1:])

        l3_call_count = [0]

        def mock_l3_init(*args, **kwargs):
            l3_call_count[0] += 1
            if l3_call_count[0] == 1:
                return bad_l3_rows
            return full_l3_coverage_rows

        mocker.patch("src.graph_full.l3_init_from_l2", side_effect=mock_l3_init)

        ok_response = {"decision": "OK", "scope": "ALL", "issues": [], "suggestions": []}
        return_response = {"decision": "RETURN", "scope": "L3", "issues": ["L3-0 empty effect"],
                          "suggestions": [], "suggestion": "Fix effect"}

        mocker.patch("src.graph_full.l1_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l2_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l3_to_reviewer", side_effect=[return_response, ok_response])
        mocker.patch("src.graph_full.reviewer_patch_l3", return_value=full_l3_coverage_rows[:1])
        mocker.patch("src.graph_full.l3_pass_to_reviewer_holistic", return_value=ok_response)

        graph = build_graph()
        app = graph.compile()

        initial_state = {
            "functions": sample_functions,
            "guideword_map": sample_guideword_map,
            "max_devs_per_gw": 1,
            "rag_enabled": False,
        }

        result = app.invoke(initial_state)

        assert "rows_l3" in result


class TestPipelineWithHolisticRepair:
    """Tests for pipeline with holistic repair cycle."""

    def test_holistic_repair_then_success(self, mocker, sample_functions, sample_guideword_map,
                                           full_l1_coverage_rows, full_l2_coverage_rows,
                                           full_l3_coverage_rows):
        """Pipeline repairs holistic issues then completes."""
        mocker.patch("src.graph_full.l1_init_full_coverage", return_value=full_l1_coverage_rows)
        mocker.patch("src.graph_full.l2_init_from_l1", return_value=full_l2_coverage_rows)
        mocker.patch("src.graph_full.l3_init_from_l2", return_value=full_l3_coverage_rows)

        ok_response = {"decision": "OK", "scope": "ALL", "issues": [], "suggestions": []}
        return_response = {
            "decision": "RETURN",
            "scope": "L3",
            "issues": ["L3-0 has inconsistent effect"],
            "suggestions": ["Fix the effect"],
            "suggestion": "Fix effect"
        }

        mocker.patch("src.graph_full.l1_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l2_to_reviewer", return_value=ok_response)
        mocker.patch("src.graph_full.l3_to_reviewer", return_value=ok_response)

        # Holistic reviewer returns RETURN first, then OK
        mocker.patch("src.graph_full.l3_pass_to_reviewer_holistic",
                     side_effect=[return_response, ok_response])
        mocker.patch("src.graph_full.reviewer_patch_l3", return_value=full_l3_coverage_rows[:1])

        graph = build_graph()
        app = graph.compile()

        initial_state = {
            "functions": sample_functions,
            "guideword_map": sample_guideword_map,
            "max_devs_per_gw": 1,
            "rag_enabled": False,
        }

        result = app.invoke(initial_state)

        assert "rows_l3" in result
        assert result.get("holistic_ok", False) or result.get("holistic_round", 0) > 0


class TestPipelineGraphStructure:
    """Tests for the graph structure."""

    def test_graph_has_all_nodes(self):
        """Graph has all required nodes."""
        graph = build_graph()
        compiled = graph.compile()

        # Check that graph compiles (has valid structure)
        assert compiled is not None

    def test_graph_state_type(self):
        """Graph state is HazopGraphState TypedDict."""
        from src.graph_full import HazopGraphState

        # Verify key fields exist
        assert "functions" in HazopGraphState.__annotations__
        assert "rows_l1" in HazopGraphState.__annotations__
        assert "rows_l2" in HazopGraphState.__annotations__
        assert "rows_l3" in HazopGraphState.__annotations__
