# tests/test_chains.py
"""Tests for src/chains.py chain functions with mocked LLM."""
import pytest
import json
from unittest.mock import patch, MagicMock
from typing import Dict, Any, List

from src.chains import (
    _build_prompt,
    _chunk_rows_by_function,
    _detect_verbatim_copy,
    _ensure_list_of_dicts,
    _extract_analysis_issues,
    _fold_fallback_hints,
    _merge_holistic_decisions,
    _merge_reviewer_decisions,
    _normalize_reviewer_decision,
    _normalize_sbr_item,
    _salvage_suggestions_by_row,
    _verify_and_retry_batch,
    l1_init_full_coverage,
    l1_to_reviewer,
    reviewer_to_l1,
    l2_init_from_l1,
    l2_to_reviewer,
    l3_init_from_l2,
    l3_to_reviewer,
    reviewer_patch_l1,
    reviewer_patch_l2,
    reviewer_patch_l3,
    l3_pass_to_reviewer_holistic,
)


class TestEnsureListOfDicts:
    """Tests for _ensure_list_of_dicts() function."""

    def test_list_of_dicts_passthrough(self):
        """List of dicts passes through."""
        rows = [{"a": 1}, {"b": 2}]
        result = _ensure_list_of_dicts(rows, stage="TEST")
        assert result == rows

    def test_json_string_rows_parsed(self):
        """JSON string rows are parsed."""
        rows = ['{"a": 1}', '{"b": 2}']
        result = _ensure_list_of_dicts(rows, stage="TEST")
        assert result == [{"a": 1}, {"b": 2}]

    def test_invalid_string_raises(self):
        """Invalid JSON string raises ValueError."""
        rows = ['not json']
        with pytest.raises(ValueError) as exc_info:
            _ensure_list_of_dicts(rows, stage="TEST")
        assert "TEST" in str(exc_info.value)

    def test_non_dict_type_raises(self):
        """Non-dict, non-string type raises ValueError."""
        rows = [123]
        with pytest.raises(ValueError) as exc_info:
            _ensure_list_of_dicts(rows, stage="TEST")
        assert "must be dict or JSON-string" in str(exc_info.value)

    def test_list_of_lists_flattened(self):
        """Nested list-of-lists is flattened."""
        rows = [[{"a": 1}], [{"b": 2}]]
        result = _ensure_list_of_dicts(rows, stage="TEST")
        assert result == [{"a": 1}, {"b": 2}]

    def test_dict_wrapper_unwrapped(self):
        """Dict wrapper with 'rows' key is unwrapped."""
        rows = {"rows": [{"a": 1}]}
        result = _ensure_list_of_dicts(rows, stage="TEST")
        assert result == [{"a": 1}]


class TestNormalizeReviewerDecision:
    """Tests for _normalize_reviewer_decision() function."""

    def test_ok_decision_normalized(self):
        """OK decision is normalized correctly."""
        raw = {"decision": "ok", "scope": "l1", "issues": [], "suggestions": []}
        result = _normalize_reviewer_decision(raw, default_scope="L1")
        assert result["decision"] == "OK"
        assert result["scope"] == "L1"

    def test_return_decision_normalized(self):
        """RETURN decision is normalized correctly."""
        raw = {"decision": "return", "scope": "L2", "issues": ["error"], "suggestions": ["fix"]}
        result = _normalize_reviewer_decision(raw, default_scope="L1")
        assert result["decision"] == "RETURN"
        assert result["scope"] == "L2"
        assert result["issues"] == ["error"]
        assert result["suggestions"] == ["fix"]

    def test_escalate_decision_normalized(self):
        """ESCALATE decision is normalized correctly."""
        raw = {"decision": "escalate"}
        result = _normalize_reviewer_decision(raw, default_scope="L1")
        assert result["decision"] == "ESCALATE"

    def test_non_object_input(self):
        """Non-object input returns ESCALATE."""
        result = _normalize_reviewer_decision("not a dict", default_scope="L1")
        assert result["decision"] == "ESCALATE"
        assert "non-object" in result["issues"][0]

    def test_default_scope_used(self):
        """Default scope used when not provided."""
        raw = {"decision": "ok"}
        result = _normalize_reviewer_decision(raw, default_scope="L2")
        assert result["scope"] == "L2"

    def test_unknown_decision_defaults_to_return(self):
        """Unknown decision defaults to RETURN."""
        raw = {"decision": "unknown"}
        result = _normalize_reviewer_decision(raw, default_scope="L1")
        assert result["decision"] == "RETURN"

    def test_errors_key_accepted(self):
        """'errors' key is accepted as alias for 'issues'."""
        raw = {"decision": "return", "errors": ["error1"]}
        result = _normalize_reviewer_decision(raw, default_scope="L1")
        assert result["issues"] == ["error1"]

    def test_fix_hints_converted_to_suggestions(self):
        """'fix_hints' dict is converted to suggestions."""
        raw = {"decision": "return", "fix_hints": {"field": "fix this"}}
        result = _normalize_reviewer_decision(raw, default_scope="L1")
        assert any("field" in s for s in result["suggestions"])

    def test_suggestions_by_row_salvaged_from_rows_key(self):
        """suggestions_by_row salvaged from 'rows' key."""
        raw = {
            "decision": "return",
            "rows": [{"row_id": "L1-0", "fields": ["deviation"], "problem": "empty"}]
        }
        result = _normalize_reviewer_decision(raw, default_scope="L1")
        assert len(result["suggestions_by_row"]) == 1
        assert result["suggestions_by_row"][0]["row_id"] == "L1-0"

    def test_analysis_block_extracted_to_issues(self):
        """'analysis' block content is extracted to issues."""
        raw = {
            "decision": "return",
            "analysis": {"problem_area": ["item1", "item2"]}
        }
        result = _normalize_reviewer_decision(raw, default_scope="L1")
        assert any("problem_area" in issue for issue in result["issues"])


class TestL1InitFullCoverage:
    """Tests for l1_init_full_coverage() with mocked LLM."""

    def test_returns_valid_rows(self, mocker, sample_functions, sample_guideword_map, configured_llm):
        """Returns valid L1 rows from mocked LLM."""
        mock_response = [
            {"row_id": "L1-0", "function": "Braking", "guideword": "no", "deviation": "No braking"}
        ]
        mocker.patch("src.chains.chat_json", return_value=mock_response)

        rows = l1_init_full_coverage(sample_functions, sample_guideword_map, max_devs_per_gw=1)
        assert len(rows) == 1
        assert rows[0]["function"] == "Braking"

    def test_handles_wrapped_response(self, mocker, sample_functions, sample_guideword_map, configured_llm):
        """Handles LLM response wrapped in 'rows' key."""
        mock_response = {
            "rows": [{"row_id": "L1-0", "function": "Braking", "guideword": "no", "deviation": "d"}]
        }
        mocker.patch("src.chains.chat_json", return_value=mock_response)

        rows = l1_init_full_coverage(sample_functions, sample_guideword_map)
        assert len(rows) == 1


class TestL1ToReviewer:
    """Tests for l1_to_reviewer() with mocked LLM."""

    def test_returns_normalized_decision(self, mocker, configured_llm):
        """Returns normalized reviewer decision."""
        mock_response = {"decision": "ok", "issues": []}
        mocker.patch("src.chains.chat_json", return_value=mock_response)

        validator_report = {"stage": "L1", "errors": []}
        rows_l1 = [{"row_id": "L1-0", "function": "Test", "guideword": "no", "deviation": "d"}]

        decision = l1_to_reviewer(validator_report, rows_l1)
        assert decision["decision"] == "OK"


class TestL2InitFromL1:
    """Tests for l2_init_from_l1() with mocked LLM."""

    def test_adds_causes_to_rows(self, mocker, full_l1_coverage_rows, configured_llm):
        """Adds cause field to L1 rows."""
        mock_response = [
            {**row, "cause": f"Cause for {row['deviation']}"}
            for row in full_l1_coverage_rows
        ]
        response_by_id = {r["row_id"]: r for r in mock_response}

        def batch_aware_response(prompt, model=None):
            # Match json.dumps format ("row_id": "L1-0") to avoid false positives
            # from example row IDs in the prompt instructions (which use "row_id":"L1-0" without space)
            matched = [row for rid, row in response_by_id.items() if f'"row_id": "{rid}"' in prompt]
            return matched if matched else mock_response

        mocker.patch("src.chains.chat_json", side_effect=batch_aware_response)

        rows = l2_init_from_l1(full_l1_coverage_rows)
        assert len(rows) == len(full_l1_coverage_rows)
        assert all("cause" in r for r in rows)


class TestL3InitFromL2:
    """Tests for l3_init_from_l2() with mocked LLM."""

    def test_adds_effects_to_rows(self, mocker, full_l2_coverage_rows, configured_llm):
        """Adds effect and triage fields to L2 rows."""
        mock_response = [
            {**row, "effect": f"Effect for {row['cause']}", "potentially_dangerous": True}
            for row in full_l2_coverage_rows
        ]
        response_by_id = {r["row_id"]: r for r in mock_response}

        def batch_aware_response(prompt, model=None):
            # Match json.dumps format ("row_id": "L2-0") to avoid false positives
            # from example row IDs in prompt instructions
            matched = [row for rid, row in response_by_id.items() if f'"row_id": "{rid}"' in prompt]
            return matched if matched else mock_response

        mocker.patch("src.chains.chat_json", side_effect=batch_aware_response)

        rows = l3_init_from_l2(full_l2_coverage_rows)
        assert len(rows) == len(full_l2_coverage_rows)
        assert all("effect" in r for r in rows)
        assert all("potentially_dangerous" in r for r in rows)


class TestReviewerPatchFunctions:
    """Tests for reviewer_patch_l1/l2/l3() functions."""

    def test_patch_l1_returns_rows(self, mocker, configured_llm):
        """reviewer_patch_l1 returns patched rows."""
        mock_response = [
            {"row_id": "L1-0", "function": "Test", "guideword": "no", "deviation": "Fixed deviation"}
        ]
        mocker.patch("src.chains.chat_json", return_value=mock_response)

        rows = [{"row_id": "L1-0", "function": "Test", "guideword": "no", "deviation": ""}]
        hints = [{"row_id": "L1-0", "fields": ["deviation"], "suggestion": "Fix it"}]

        result = reviewer_patch_l1(rows, hints)
        assert len(result) == 1
        assert result[0]["deviation"] == "Fixed deviation"

    def test_patch_l2_returns_rows(self, mocker, configured_llm):
        """reviewer_patch_l2 returns patched rows."""
        mock_response = [
            {"row_id": "L2-0", "function": "Test", "guideword": "no",
             "deviation": "d", "cause": "Fixed cause"}
        ]
        mocker.patch("src.chains.chat_json", return_value=mock_response)

        rows = [{"row_id": "L2-0", "function": "Test", "guideword": "no",
                 "deviation": "d", "cause": ""}]
        hints = [{"row_id": "L2-0", "fields": ["cause"], "suggestion": "Fix it"}]

        result = reviewer_patch_l2(rows, hints)
        assert len(result) == 1
        assert result[0]["cause"] == "Fixed cause"

    def test_patch_l3_returns_rows(self, mocker, configured_llm):
        """reviewer_patch_l3 returns patched rows."""
        mock_response = [
            {"row_id": "L3-0", "function": "Test", "guideword": "no",
             "deviation": "d", "cause": "c", "effect": "Fixed effect",
             "potentially_dangerous": True}
        ]
        mocker.patch("src.chains.chat_json", return_value=mock_response)

        rows = [{"row_id": "L3-0", "function": "Test", "guideword": "no",
                 "deviation": "d", "cause": "c", "effect": "",
                 "potentially_dangerous": False}]
        hints = [{"row_id": "L3-0", "fields": ["effect"], "suggestion": "Fix it"}]

        result = reviewer_patch_l3(rows, hints)
        assert len(result) == 1
        assert result[0]["effect"] == "Fixed effect"


class TestHolisticReviewer:
    """Tests for l3_pass_to_reviewer_holistic() function."""

    def test_returns_normalized_decision(self, mocker, full_l3_coverage_rows, configured_llm):
        """Returns normalized holistic reviewer decision."""
        mock_response = {"decision": "ok", "scope": "ALL", "issues": []}
        mocker.patch("src.chains.chat_json", return_value=mock_response)

        decision = l3_pass_to_reviewer_holistic(full_l3_coverage_rows, [])
        assert decision["decision"] == "OK"
        assert decision["scope"] == "ALL"

    def test_returns_issues_when_problems_found(self, mocker, configured_llm):
        """Returns issues when holistic review finds problems."""
        mock_response = {
            "decision": "return",
            "scope": "L2",
            "issues": ["Inconsistency in row L2-0"]
        }
        mocker.patch("src.chains.chat_json", return_value=mock_response)

        rows = [{"row_id": "L3-0", "effect": "e", "potentially_dangerous": True}]
        decision = l3_pass_to_reviewer_holistic(rows, [])

        assert decision["decision"] == "RETURN"
        assert len(decision["issues"]) > 0


class TestVerifyAndRetryBatch:
    """Tests for _verify_and_retry_batch() helper."""

    def test_all_present_no_retry(self):
        """When all input rows are matched in output, returns output unchanged."""
        input_rows = [
            {"row_id": "L2-0", "function": "Braking", "guideword": "no", "deviation": "d0"},
            {"row_id": "L2-1", "function": "Braking", "guideword": "more", "deviation": "d1"},
            {"row_id": "L2-2", "function": "Steering", "guideword": "less", "deviation": "d2"},
        ]
        output_rows = [
            {**r, "cause": f"cause-{r['row_id']}"} for r in input_rows
        ]
        retry_fn = MagicMock()

        result = _verify_and_retry_batch(input_rows, output_rows, retry_fn, "TEST")

        assert result == output_rows
        retry_fn.assert_not_called()

    def test_missing_rows_retried(self):
        """Missing rows are retried individually and placed at their original positions."""
        input_rows = [
            {"row_id": f"L2-{i}", "function": "F", "guideword": "no", "deviation": f"d{i}"}
            for i in range(5)
        ]
        # Output missing L2-3 and L2-4
        output_rows = [
            {**r, "cause": f"cause-{r['row_id']}"} for r in input_rows[:3]
        ]

        def retry_side_effect(rows):
            return [{**rows[0], "cause": f"retried-{rows[0]['row_id']}"}]

        retry_fn = MagicMock(side_effect=retry_side_effect)

        result = _verify_and_retry_batch(input_rows, output_rows, retry_fn, "TEST")

        assert len(result) == 5
        assert retry_fn.call_count == 2
        # Recovered rows should be at their original positions (indices 3 and 4)
        assert result[3]["row_id"] == "L2-3"
        assert result[3]["cause"] == "retried-L2-3"
        assert result[4]["row_id"] == "L2-4"
        assert result[4]["cause"] == "retried-L2-4"

    def test_composite_key_fallback(self):
        """Rows matched by (function, guideword, deviation) when row_id differs."""
        input_rows = [
            {"row_id": "L2-0", "function": "Braking", "guideword": "no", "deviation": "No brake"},
            {"row_id": "L2-1", "function": "Steering", "guideword": "more", "deviation": "Over steer"},
        ]
        # Output has different row_ids but same composite keys
        output_rows = [
            {"row_id": "L2-99", "function": "Braking", "guideword": "no",
             "deviation": "No brake", "cause": "c0"},
            {"row_id": "L2-100", "function": "Steering", "guideword": "more",
             "deviation": "Over steer", "cause": "c1"},
        ]
        retry_fn = MagicMock()

        result = _verify_and_retry_batch(input_rows, output_rows, retry_fn, "TEST")

        assert len(result) == len(output_rows)
        retry_fn.assert_not_called()

    def test_retry_failure_skips_row(self):
        """When retry_fn raises, the missing row is excluded from results."""
        input_rows = [
            {"row_id": "L2-0", "function": "F", "guideword": "no", "deviation": "d0"},
            {"row_id": "L2-1", "function": "F", "guideword": "more", "deviation": "d1"},
        ]
        output_rows = [
            {**input_rows[0], "cause": "c0"},
        ]
        retry_fn = MagicMock(side_effect=Exception("LLM error"))

        result = _verify_and_retry_batch(input_rows, output_rows, retry_fn, "TEST")

        # Only the matched row survives; L2-1 is lost after retry failures
        assert len(result) == 1
        assert result[0]["row_id"] == "L2-0"

    def test_empty_input(self):
        """Empty input_rows returns output_rows unchanged."""
        output_rows = [{"row_id": "L2-0", "cause": "c"}]
        retry_fn = MagicMock()

        result = _verify_and_retry_batch([], output_rows, retry_fn, "TEST")

        assert result == output_rows
        retry_fn.assert_not_called()

    def test_empty_output_retries_all(self):
        """Empty output_rows triggers retry for every input row, preserving order."""
        input_rows = [
            {"row_id": "L2-0", "function": "F", "guideword": "no", "deviation": "d0"},
            {"row_id": "L2-1", "function": "F", "guideword": "more", "deviation": "d1"},
        ]

        def retry_side_effect(rows):
            return [{**rows[0], "cause": f"retried-{rows[0]['row_id']}"}]

        retry_fn = MagicMock(side_effect=retry_side_effect)

        result = _verify_and_retry_batch(input_rows, [], retry_fn, "TEST")

        assert len(result) == 2
        assert retry_fn.call_count == 2
        # Order must match input order
        assert result[0]["row_id"] == "L2-0"
        assert result[1]["row_id"] == "L2-1"


# ============================================================================
#                     PURE-LOGIC HELPER TESTS
# ============================================================================


class TestBuildPrompt:
    """Tests for _build_prompt()."""

    def test_minimal_instructions_and_payload(self):
        p = {"instructions": "Do the thing."}
        result = _build_prompt(p, "PAYLOAD")
        assert "Do the thing." in result
        assert result.endswith("PAYLOAD")

    def test_with_constraints(self):
        p = {"instructions": "Instr", "constraints": "Max 50 rows."}
        result = _build_prompt(p, "PAYLOAD")
        assert "Max 50 rows." in result

    def test_with_few_shot_examples(self):
        p = {
            "instructions": "Instr",
            "few_shot_examples": [
                {"role": "user", "content": "example input"},
                {"role": "assistant", "content": "example output"},
            ],
        }
        result = _build_prompt(p, "PAYLOAD")
        assert "EXAMPLES:" in result
        assert "[USER]" in result
        assert "[ASSISTANT]" in result
        assert "example input" in result

    def test_instr_override(self):
        p = {"instructions": "Original"}
        result = _build_prompt(p, "PAYLOAD", instr="Override")
        assert "Override" in result
        assert "Original" not in result

    def test_extra_section_appended(self):
        p = {"instructions": "Instr"}
        result = _build_prompt(p, "PAYLOAD", extra="EXTRA CONTEXT")
        parts = result.split("\n\n")
        assert "EXTRA CONTEXT" in parts

    def test_empty_extra_not_added(self):
        p = {"instructions": "Instr"}
        result = _build_prompt(p, "PAYLOAD", extra="")
        # No double blank lines between instructions and payload
        assert "Instr\n\nPAYLOAD" == result

    def test_all_sections_combined(self):
        p = {
            "instructions": "Instr",
            "constraints": "Constraints",
            "few_shot_examples": [{"role": "user", "content": "ex"}],
        }
        result = _build_prompt(p, "PAYLOAD", extra="Extra")
        parts = result.split("\n\n")
        assert parts[0] == "Instr"
        assert parts[1] == "Constraints"
        assert "EXAMPLES:" in parts[2]
        assert parts[3] == "Extra"
        assert parts[4] == "PAYLOAD"


class TestExtractAnalysisIssues:
    """Tests for _extract_analysis_issues()."""

    def test_no_analysis_key(self):
        assert _extract_analysis_issues({"decision": "OK"}) == []

    def test_analysis_not_dict(self):
        assert _extract_analysis_issues({"analysis": "string"}) == []
        assert _extract_analysis_issues({"analysis": 42}) == []

    def test_simple_key_value(self):
        raw = {"analysis": {"coverage": "good", "consistency": "poor"}}
        issues = _extract_analysis_issues(raw)
        assert "coverage: good" in issues
        assert "consistency: poor" in issues

    def test_list_value(self):
        raw = {"analysis": {"problems": ["p1", "p2"]}}
        issues = _extract_analysis_issues(raw)
        assert len(issues) == 1
        assert "problems:" in issues[0]

    def test_missing_guidewords_format(self):
        raw = {
            "analysis": {
                "coverage_check": [
                    {"function": "Braking", "missing_guidewords": ["no", "more"]},
                    {"function": "Steering", "missing_guidewords": ["less"]},
                ]
            }
        }
        issues = _extract_analysis_issues(raw)
        assert len(issues) == 2
        assert "Braking" in issues[0]
        assert "['no', 'more']" in issues[0]
        assert "Steering" in issues[1]


class TestFoldFallbackHints:
    """Tests for _fold_fallback_hints()."""

    def test_empty_raw(self):
        assert _fold_fallback_hints({}) == []

    def test_fix_hints_dict(self):
        raw = {"fix_hints": {"field": "fix this", "other": "that"}}
        hints = _fold_fallback_hints(raw)
        assert "field: fix this" in hints
        assert "other: that" in hints

    def test_fix_hints_list(self):
        raw = {"fix_hints": ["hint1", "hint2"]}
        hints = _fold_fallback_hints(raw)
        assert hints == ["hint1", "hint2"]

    def test_routing_decision_appended(self):
        raw = {"routing_decision": "RETURN to L1"}
        hints = _fold_fallback_hints(raw)
        assert "routing_decision: RETURN to L1" in hints

    def test_corrective_action_appended(self):
        raw = {"corrective_action": "fix deviation"}
        hints = _fold_fallback_hints(raw)
        assert "corrective_action: fix deviation" in hints

    def test_all_sources_combined(self):
        raw = {
            "fix_hints": {"a": "b"},
            "routing_decision": "RD",
            "corrective_action": "CA",
        }
        hints = _fold_fallback_hints(raw)
        assert len(hints) == 3


class TestSalvageSuggestionsByRow:
    """Tests for _salvage_suggestions_by_row()."""

    def test_direct_key_returned(self):
        raw = {"suggestions_by_row": [{"row_id": "L1-0", "fields": ["deviation"]}]}
        result = _salvage_suggestions_by_row(raw, [])
        assert len(result) == 1
        assert result[0]["row_id"] == "L1-0"

    def test_salvage_from_rows_key(self):
        raw = {"rows": [{"row_id": "L1-0", "fields": ["deviation"], "problem": "empty"}]}
        result = _salvage_suggestions_by_row(raw, [])
        assert len(result) == 1

    def test_salvage_from_response_root(self):
        raw = {"row_id": "L1-0", "fields": ["cause"], "problem": "p", "suggestion": "s"}
        result = _salvage_suggestions_by_row(raw, [])
        assert len(result) == 1
        assert result[0]["row_id"] == "L1-0"

    def test_salvage_from_suggestions_list(self):
        suggestions = [{"row_id": "L2-1", "fields": ["cause"]}]
        raw = {}
        result = _salvage_suggestions_by_row(raw, suggestions)
        assert len(result) == 1

    def test_no_valid_source_returns_empty(self):
        raw = {"decision": "OK"}
        result = _salvage_suggestions_by_row(raw, [])
        assert result == []

    def test_rows_key_without_fields_not_salvaged(self):
        raw = {"rows": [{"row_id": "L1-0", "value": "no fields key"}]}
        result = _salvage_suggestions_by_row(raw, [])
        assert result == []


class TestNormalizeSbrItem:
    """Tests for _normalize_sbr_item()."""

    def test_non_dict_returns_empty(self):
        assert _normalize_sbr_item("string") == {}
        assert _normalize_sbr_item(42) == {}
        assert _normalize_sbr_item(None) == {}

    def test_missing_row_id_returns_empty(self):
        assert _normalize_sbr_item({"fields": ["deviation"]}) == {}

    def test_missing_fields_returns_empty(self):
        assert _normalize_sbr_item({"row_id": "L1-0"}) == {}

    def test_empty_fields_returns_empty(self):
        assert _normalize_sbr_item({"row_id": "L1-0", "fields": []}) == {}

    def test_valid_item_normalized(self):
        item = {
            "row_id": "L1-0",
            "fields": ["deviation", "cause"],
            "problem": "empty field",
            "suggestion": "add content",
        }
        result = _normalize_sbr_item(item)
        assert result["row_id"] == "L1-0"
        assert result["fields"] == ["deviation", "cause"]
        assert result["problem"] == "empty field"
        assert result["suggestion"] == "add content"

    def test_fields_coerced_to_strings(self):
        item = {"row_id": "L1-0", "fields": [123, True]}
        result = _normalize_sbr_item(item)
        assert result["fields"] == ["123", "True"]

    def test_missing_problem_and_suggestion_default_empty(self):
        item = {"row_id": "L1-0", "fields": ["deviation"]}
        result = _normalize_sbr_item(item)
        assert result["problem"] == ""
        assert result["suggestion"] == ""


class TestDetectVerbatimCopy:
    """Tests for _detect_verbatim_copy()."""

    def test_no_hints_returns_empty(self):
        rows = [{"row_id": "L1-0", "deviation": "some text"}]
        result = _detect_verbatim_copy(rows, [], ["deviation"])
        assert result == []

    def test_no_matching_row_ids(self):
        rows = [{"row_id": "L1-0", "deviation": "some text"}]
        hints = [{"row_id": "L1-99", "suggestion": "some long suggestion text here"}]
        result = _detect_verbatim_copy(rows, hints, ["deviation"])
        assert result == []

    def test_exact_match_detected(self):
        rows = [{"row_id": "L1-0", "deviation": "The suggestion was copied verbatim"}]
        hints = [{"row_id": "L1-0", "suggestion": "The suggestion was copied verbatim"}]
        result = _detect_verbatim_copy(rows, hints, ["deviation"])
        assert result == ["L1-0"]

    def test_containment_detected(self):
        rows = [{"row_id": "L1-0", "deviation": "prefix The suggestion was copied verbatim suffix"}]
        hints = [{"row_id": "L1-0", "suggestion": "The suggestion was copied verbatim"}]
        result = _detect_verbatim_copy(rows, hints, ["deviation"])
        assert result == ["L1-0"]

    def test_case_insensitive_match(self):
        rows = [{"row_id": "L1-0", "deviation": "THE SUGGESTION WAS COPIED VERBATIM"}]
        hints = [{"row_id": "L1-0", "suggestion": "the suggestion was copied verbatim"}]
        result = _detect_verbatim_copy(rows, hints, ["deviation"])
        assert result == ["L1-0"]

    def test_whitespace_normalized(self):
        rows = [{"row_id": "L1-0", "deviation": "the  suggestion   was copied"}]
        hints = [{"row_id": "L1-0", "suggestion": "the suggestion was copied"}]
        result = _detect_verbatim_copy(rows, hints, ["deviation"])
        assert result == ["L1-0"]

    def test_short_suggestion_ignored(self):
        """Suggestions shorter than 10 chars are ignored."""
        rows = [{"row_id": "L1-0", "deviation": "short"}]
        hints = [{"row_id": "L1-0", "suggestion": "short"}]
        result = _detect_verbatim_copy(rows, hints, ["deviation"])
        assert result == []

    def test_multiple_target_fields_first_match_breaks(self):
        rows = [{"row_id": "L1-0", "deviation": "match this long suggestion text", "cause": "other"}]
        hints = [{"row_id": "L1-0", "suggestion": "match this long suggestion text"}]
        result = _detect_verbatim_copy(rows, hints, ["deviation", "cause"])
        assert result == ["L1-0"]

    def test_no_verbatim_copy(self):
        rows = [{"row_id": "L1-0", "deviation": "completely different technical content"}]
        hints = [{"row_id": "L1-0", "suggestion": "this is a long suggestion about something else"}]
        result = _detect_verbatim_copy(rows, hints, ["deviation"])
        assert result == []


class TestChunkRowsByFunction:
    """Tests for _chunk_rows_by_function()."""

    def test_empty_rows(self):
        assert _chunk_rows_by_function([], 3) == []

    def test_single_function_single_chunk(self):
        rows = [
            {"function": "F1", "row_id": "L1-0"},
            {"function": "F1", "row_id": "L1-1"},
        ]
        chunks = _chunk_rows_by_function(rows, 3)
        assert len(chunks) == 1
        assert len(chunks[0]) == 2

    def test_grouping_by_function(self):
        rows = [
            {"function": "F1", "row_id": "L1-0"},
            {"function": "F2", "row_id": "L1-1"},
            {"function": "F1", "row_id": "L1-2"},
        ]
        chunks = _chunk_rows_by_function(rows, 5)
        assert len(chunks) == 1
        # F1 rows grouped together, F2 follows
        funcs_in_chunk = [r["function"] for r in chunks[0]]
        assert funcs_in_chunk == ["F1", "F1", "F2"]

    def test_batching_by_funcs_per_chunk(self):
        rows = [
            {"function": "F1", "row_id": "L1-0"},
            {"function": "F2", "row_id": "L1-1"},
            {"function": "F3", "row_id": "L1-2"},
            {"function": "F4", "row_id": "L1-3"},
        ]
        chunks = _chunk_rows_by_function(rows, 2)
        assert len(chunks) == 2
        assert [r["function"] for r in chunks[0]] == ["F1", "F2"]
        assert [r["function"] for r in chunks[1]] == ["F3", "F4"]

    def test_function_not_split_across_chunks(self):
        rows = [
            {"function": "F1", "row_id": "L1-0"},
            {"function": "F1", "row_id": "L1-1"},
            {"function": "F2", "row_id": "L1-2"},
            {"function": "F2", "row_id": "L1-3"},
            {"function": "F3", "row_id": "L1-4"},
        ]
        chunks = _chunk_rows_by_function(rows, 2)
        # Chunk 1: F1 (2 rows) + F2 (2 rows), Chunk 2: F3 (1 row)
        assert len(chunks) == 2
        chunk1_funcs = {r["function"] for r in chunks[0]}
        chunk2_funcs = {r["function"] for r in chunks[1]}
        assert chunk1_funcs == {"F1", "F2"}
        assert chunk2_funcs == {"F3"}

    def test_order_preserved_within_groups(self):
        rows = [
            {"function": "F1", "row_id": "L1-0", "gw": "no"},
            {"function": "F1", "row_id": "L1-1", "gw": "more"},
            {"function": "F1", "row_id": "L1-2", "gw": "less"},
        ]
        chunks = _chunk_rows_by_function(rows, 1)
        assert len(chunks) == 1
        assert [r["gw"] for r in chunks[0]] == ["no", "more", "less"]

    def test_funcs_per_chunk_one(self):
        rows = [
            {"function": "F1", "row_id": "0"},
            {"function": "F2", "row_id": "1"},
            {"function": "F3", "row_id": "2"},
        ]
        chunks = _chunk_rows_by_function(rows, 1)
        assert len(chunks) == 3


class TestMergeHolisticDecisions:
    """Tests for _merge_holistic_decisions()."""

    def test_empty_list(self):
        result = _merge_holistic_decisions([])
        assert result["decision"] == "OK"
        assert result["scope"] == "ALL"
        assert result["issues"] == []
        assert result["suggestion"] == ""

    def test_single_ok(self):
        decisions = [{"decision": "OK", "scope": "ALL", "issues": [], "suggestion": ""}]
        result = _merge_holistic_decisions(decisions)
        assert result["decision"] == "OK"
        assert result["scope"] == "ALL"

    def test_single_return(self):
        decisions = [{"decision": "RETURN", "scope": "L2", "issues": ["iss"], "suggestion": "fix"}]
        result = _merge_holistic_decisions(decisions)
        assert result["decision"] == "RETURN"
        assert result["scope"] == "L2"
        assert result["suggestion"] == "fix"

    def test_any_return_wins(self):
        decisions = [
            {"decision": "OK", "scope": "ALL", "issues": [], "suggestion": ""},
            {"decision": "RETURN", "scope": "L3", "issues": ["problem"], "suggestion": "fix it"},
        ]
        result = _merge_holistic_decisions(decisions)
        assert result["decision"] == "RETURN"

    def test_highest_scope_wins(self):
        decisions = [
            {"decision": "RETURN", "scope": "L3", "issues": ["a"], "suggestion": "s1"},
            {"decision": "RETURN", "scope": "L1", "issues": ["b"], "suggestion": "s2"},
        ]
        result = _merge_holistic_decisions(decisions)
        assert result["scope"] == "L1"

    def test_scope_ranking_l1_gt_l2_gt_l3_gt_all(self):
        decisions = [
            {"decision": "OK", "scope": "ALL", "issues": [], "suggestion": ""},
            {"decision": "OK", "scope": "L3", "issues": [], "suggestion": ""},
        ]
        result = _merge_holistic_decisions(decisions)
        assert result["scope"] == "L3"

    def test_issues_deduplicated_order_preserved(self):
        decisions = [
            {"decision": "OK", "scope": "ALL", "issues": ["a", "b"], "suggestion": ""},
            {"decision": "OK", "scope": "ALL", "issues": ["b", "c"], "suggestion": ""},
        ]
        result = _merge_holistic_decisions(decisions)
        assert result["issues"] == ["a", "b", "c"]

    def test_only_return_decisions_contribute_suggestions(self):
        decisions = [
            {"decision": "OK", "scope": "ALL", "issues": [], "suggestion": "ok suggestion"},
            {"decision": "RETURN", "scope": "L2", "issues": ["iss"], "suggestion": "return suggestion"},
        ]
        result = _merge_holistic_decisions(decisions)
        assert result["suggestion"] == "return suggestion"
        assert "ok suggestion" not in result["suggestion"]

    def test_blank_suggestions_filtered(self):
        decisions = [
            {"decision": "RETURN", "scope": "L2", "issues": ["iss"], "suggestion": "   "},
            {"decision": "RETURN", "scope": "L3", "issues": ["iss2"], "suggestion": "real fix"},
        ]
        result = _merge_holistic_decisions(decisions)
        assert result["suggestion"] == "real fix"

    def test_multiple_return_suggestions_joined(self):
        decisions = [
            {"decision": "RETURN", "scope": "L2", "issues": [], "suggestion": "fix A"},
            {"decision": "RETURN", "scope": "L3", "issues": [], "suggestion": "fix B"},
        ]
        result = _merge_holistic_decisions(decisions)
        assert "fix A" in result["suggestion"]
        assert "fix B" in result["suggestion"]
        assert " | " in result["suggestion"]


class TestMergeReviewerDecisions:
    """Tests for _merge_reviewer_decisions()."""

    def test_empty_list(self):
        result = _merge_reviewer_decisions([], "L1")
        assert result["decision"] == "OK"
        assert result["scope"] == "L1"
        assert result["issues"] == []
        assert result["suggestions"] == []
        assert result["suggestions_by_row"] == []

    def test_any_return_wins(self):
        decisions = [
            {"decision": "OK", "issues": [], "suggestions": [], "suggestions_by_row": []},
            {"decision": "RETURN", "issues": ["err"], "suggestions": ["fix"], "suggestions_by_row": []},
        ]
        result = _merge_reviewer_decisions(decisions, "L2")
        assert result["decision"] == "RETURN"

    def test_issues_deduplicated(self):
        decisions = [
            {"decision": "OK", "issues": ["a", "b"], "suggestions": [], "suggestions_by_row": []},
            {"decision": "OK", "issues": ["b", "c"], "suggestions": [], "suggestions_by_row": []},
        ]
        result = _merge_reviewer_decisions(decisions, "L1")
        assert result["issues"] == ["a", "b", "c"]

    def test_suggestions_deduplicated(self):
        decisions = [
            {"decision": "OK", "issues": [], "suggestions": ["s1", "s2"], "suggestions_by_row": []},
            {"decision": "OK", "issues": [], "suggestions": ["s2", "s3"], "suggestions_by_row": []},
        ]
        result = _merge_reviewer_decisions(decisions, "L1")
        assert result["suggestions"] == ["s1", "s2", "s3"]

    def test_sbr_concatenated(self):
        sbr1 = [{"row_id": "L1-0", "fields": ["deviation"]}]
        sbr2 = [{"row_id": "L1-1", "fields": ["deviation"]}]
        decisions = [
            {"decision": "RETURN", "issues": [], "suggestions": [], "suggestions_by_row": sbr1},
            {"decision": "RETURN", "issues": [], "suggestions": [], "suggestions_by_row": sbr2},
        ]
        result = _merge_reviewer_decisions(decisions, "L1")
        assert len(result["suggestions_by_row"]) == 2

    def test_scope_is_default(self):
        decisions = [
            {"decision": "OK", "issues": [], "suggestions": [], "suggestions_by_row": []},
        ]
        result = _merge_reviewer_decisions(decisions, "L3")
        assert result["scope"] == "L3"


class TestL1CoverageRetry:
    """Tests for the post-batch function-level coverage check in l1_init_full_coverage."""

    def test_missing_function_recovered_on_retry(self, mocker, sample_guideword_map, configured_llm):
        """First call returns only last function; retry returns missing function."""
        functions = ["Braking", "Steering"]
        # First batch call: returns only Steering rows
        steering_rows = [
            {"row_id": f"L1-{i}", "function": "Steering", "guideword": f"gw{i}", "deviation": f"d{i}"}
            for i in range(11)
        ]
        # Retry call: returns Braking rows
        braking_rows = [
            {"row_id": f"L1-{i+11}", "function": "Braking", "guideword": f"gw{i}", "deviation": f"d{i}"}
            for i in range(11)
        ]
        mock_chat = mocker.patch("src.chains.chat_json", side_effect=[steering_rows, braking_rows])

        rows = l1_init_full_coverage(functions, sample_guideword_map, max_devs_per_gw=1)

        covered = {r["function"] for r in rows}
        assert "Braking" in covered
        assert "Steering" in covered
        assert mock_chat.call_count == 2  # 1 batch + 1 retry

    def test_no_retry_when_all_covered(self, mocker, sample_guideword_map, configured_llm):
        """All functions returned on first call. No retry needed."""
        functions = ["Braking", "Steering"]
        all_rows = [
            {"row_id": f"L1-{i}", "function": fn, "guideword": "no", "deviation": f"d{i}"}
            for i, fn in enumerate(functions)
        ]
        mock_chat = mocker.patch("src.chains.chat_json", return_value=all_rows)

        rows = l1_init_full_coverage(functions, sample_guideword_map, max_devs_per_gw=1)

        assert len(rows) == 2
        assert mock_chat.call_count == 1  # Only the batch call, no retries

    def test_retry_failure_logs_error(self, mocker, sample_guideword_map, configured_llm):
        """Retry raises Exception. Partial results returned, 3 total calls."""
        functions = ["Braking", "Steering"]
        # Batch returns only Steering
        steering_rows = [
            {"row_id": "L1-0", "function": "Steering", "guideword": "no", "deviation": "d0"}
        ]
        mock_chat = mocker.patch(
            "src.chains.chat_json",
            side_effect=[steering_rows, Exception("LLM error"), Exception("LLM error")],
        )

        rows = l1_init_full_coverage(functions, sample_guideword_map, max_devs_per_gw=1)

        # Only Steering rows survive
        assert all(r["function"] == "Steering" for r in rows)
        assert mock_chat.call_count == 3  # 1 batch + 2 retries

    def test_multiple_missing_functions(self, mocker, sample_guideword_map, configured_llm):
        """Two functions missing, each retried individually."""
        functions = ["Braking", "Steering", "Throttle"]
        # Batch returns only Throttle
        throttle_rows = [
            {"row_id": "L1-0", "function": "Throttle", "guideword": "no", "deviation": "d0"}
        ]
        braking_rows = [
            {"row_id": "L1-1", "function": "Braking", "guideword": "no", "deviation": "d1"}
        ]
        steering_rows = [
            {"row_id": "L1-2", "function": "Steering", "guideword": "no", "deviation": "d2"}
        ]
        mock_chat = mocker.patch(
            "src.chains.chat_json",
            side_effect=[throttle_rows, braking_rows, steering_rows],
        )

        rows = l1_init_full_coverage(functions, sample_guideword_map, max_devs_per_gw=1)

        covered = {r["function"] for r in rows}
        assert covered == {"Braking", "Steering", "Throttle"}
        assert mock_chat.call_count == 3  # 1 batch + 2 retries (one per missing fn)
