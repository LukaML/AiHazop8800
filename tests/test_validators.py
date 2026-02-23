# tests/test_validators.py
"""Tests for src/validators.py validation functions."""
import pytest
from typing import List, Dict, Any

from src.validators import (
    parse_rows_safely,
    validate_l1_payload,
    validate_l2_payload,
    validate_l3_payload,
    build_validator_report,
    _check_full_guideword_coverage,
    _check_duplicates,
    _check_non_empty,
)
from src.models import HazopL1Row, HazopL2Row, HazopL3Row


class TestParseRowsSafely:
    """Tests for parse_rows_safely() function."""

    def test_valid_rows_parsed(self):
        """Valid row dicts are parsed into Pydantic models."""
        payload = [
            {"row_id": "L1-0", "function": "Test", "guideword": "no", "deviation": "d1"}
        ]
        rows, issues = parse_rows_safely(payload, HazopL1Row)
        assert len(rows) == 1
        assert len(issues) == 0
        assert isinstance(rows[0], HazopL1Row)

    def test_invalid_rows_recorded(self):
        """Invalid rows generate issues but are preserved."""
        payload = [
            {"row_id": "L1-0", "function": "Test", "guideword": "no", "deviation": ""}
        ]
        rows, issues = parse_rows_safely(payload, HazopL1Row)
        assert len(rows) == 1  # Row preserved even if invalid
        assert len(issues) > 0  # Issue recorded

    def test_string_json_rows_parsed(self):
        """JSON string rows are parsed."""
        payload = ['{"row_id": "L1-0", "function": "Test", "guideword": "no", "deviation": "d1"}']
        rows, issues = parse_rows_safely(payload, HazopL1Row)
        assert len(rows) == 1
        assert len(issues) == 0

    def test_invalid_json_string_recorded(self):
        """Invalid JSON string generates issue."""
        payload = ['not valid json']
        rows, issues = parse_rows_safely(payload, HazopL1Row)
        assert len(rows) == 0
        assert len(issues) == 1
        assert "not JSON-decodable" in issues[0]

    def test_non_list_payload_recorded(self):
        """Non-list payload generates issue."""
        payload = "not a list"
        rows, issues = parse_rows_safely(payload, HazopL1Row)
        assert len(rows) == 0
        assert len(issues) == 1
        assert "not a list" in issues[0]

    def test_dict_wrapper_unwrapped(self):
        """Dict with 'rows' key is unwrapped."""
        payload = {"rows": [
            {"row_id": "L1-0", "function": "Test", "guideword": "no", "deviation": "d1"}
        ]}
        rows, issues = parse_rows_safely(payload, HazopL1Row)
        assert len(rows) == 1
        assert len(issues) == 0

    def test_missing_row_id_injected(self):
        """Missing row_id is auto-injected."""
        payload = [
            {"function": "Test", "guideword": "no", "deviation": "d1"}
        ]
        rows, issues = parse_rows_safely(payload, HazopL1Row)
        assert len(rows) == 1
        assert rows[0].row_id == "L1-0"


class TestValidateL1Payload:
    """Tests for validate_l1_payload() function."""

    def test_valid_full_coverage(self, full_l1_coverage_rows, sample_functions):
        """Valid L1 payload with full guideword coverage passes."""
        rows, ok, issues = validate_l1_payload(full_l1_coverage_rows, sample_functions)
        assert ok
        assert len(issues) == 0
        assert len(rows) == len(full_l1_coverage_rows)

    def test_empty_deviation_detected(self, sample_functions):
        """Empty deviation field is detected."""
        payload = [
            {"row_id": "L1-0", "function": "Braking", "guideword": "no", "deviation": ""}
        ]
        rows, ok, issues = validate_l1_payload(payload, sample_functions)
        assert not ok
        assert any("empty 'deviation'" in issue for issue in issues)

    def test_missing_guidewords_detected(self, sample_functions):
        """Missing guidewords for a function are detected."""
        # Only provide one guideword instead of all 11
        payload = [
            {"row_id": "L1-0", "function": "Braking", "guideword": "no", "deviation": "d1"}
        ]
        rows, ok, issues = validate_l1_payload(payload, sample_functions)
        assert not ok
        assert any("missing guidewords" in issue for issue in issues)

    def test_duplicate_detection(self, sample_functions):
        """Duplicate rows are detected."""
        payload = [
            {"row_id": "L1-0", "function": "Braking", "guideword": "no", "deviation": "same"},
            {"row_id": "L1-1", "function": "Braking", "guideword": "no", "deviation": "same"},
        ]
        rows, ok, issues = validate_l1_payload(payload, sample_functions)
        assert not ok
        assert any("duplicate" in issue for issue in issues)

    def test_unknown_function_detected(self):
        """Function not in expected list is detected."""
        payload = [
            {"row_id": "L1-0", "function": "Unknown", "guideword": "no", "deviation": "d1"}
        ]
        rows, ok, issues = validate_l1_payload(payload, ["Braking"])
        assert not ok
        assert any("not in expected set" in issue for issue in issues)


class TestValidateL2Payload:
    """Tests for validate_l2_payload() function."""

    def test_valid_l2_payload(self, full_l2_coverage_rows):
        """Valid L2 payload passes validation."""
        rows, ok, issues = validate_l2_payload(full_l2_coverage_rows)
        assert ok
        assert len(issues) == 0

    def test_empty_cause_detected(self):
        """Empty cause field is detected."""
        payload = [
            {"row_id": "L2-0", "function": "Braking", "guideword": "no",
             "deviation": "d1", "cause": ""}
        ]
        rows, ok, issues = validate_l2_payload(payload)
        assert not ok
        assert any("empty 'cause'" in issue for issue in issues)

    def test_duplicate_l2_rows_detected(self):
        """Duplicate L2 rows are detected."""
        payload = [
            {"row_id": "L2-0", "function": "Braking", "guideword": "no",
             "deviation": "d1", "cause": "same cause"},
            {"row_id": "L2-1", "function": "Braking", "guideword": "no",
             "deviation": "d1", "cause": "same cause"},
        ]
        rows, ok, issues = validate_l2_payload(payload)
        assert not ok
        assert any("duplicate" in issue for issue in issues)


class TestValidateL3Payload:
    """Tests for validate_l3_payload() function."""

    def test_valid_l3_payload(self, full_l3_coverage_rows):
        """Valid L3 payload passes validation."""
        rows, ok, issues = validate_l3_payload(full_l3_coverage_rows)
        assert ok
        assert len(issues) == 0

    def test_empty_effect_detected(self):
        """Empty effect field is detected."""
        payload = [
            {"row_id": "L3-0", "function": "Braking", "guideword": "no",
             "deviation": "d1", "cause": "c1", "effect": "",
             "potentially_dangerous": True}
        ]
        rows, ok, issues = validate_l3_payload(payload)
        assert not ok
        assert any("empty 'effect'" in issue for issue in issues)

    def test_missing_triage_detected(self):
        """Missing potentially_dangerous field is detected."""
        payload = [
            {"row_id": "L3-0", "function": "Braking", "guideword": "no",
             "deviation": "d1", "cause": "c1", "effect": "e1"}
            # Missing potentially_dangerous
        ]
        rows, ok, issues = validate_l3_payload(payload)
        assert not ok
        # The issue comes from validation error or empty check
        assert len(issues) > 0


class TestCheckFullGuidewordCoverage:
    """Tests for _check_full_guideword_coverage() helper."""

    def test_full_coverage_no_issues(self, full_l1_coverage_rows, sample_functions):
        """Full coverage returns no issues."""
        issues = _check_full_guideword_coverage(full_l1_coverage_rows, sample_functions)
        assert len(issues) == 0

    def test_missing_guidewords_reported(self, sample_functions):
        """Missing guidewords are reported."""
        rows = [
            {"function": "Braking", "guideword": "no"},  # Only one guideword
        ]
        issues = _check_full_guideword_coverage(rows, sample_functions)
        assert len(issues) > 0
        assert any("Braking" in issue and "missing guidewords" in issue for issue in issues)

    def test_handles_pydantic_models(self, sample_l1_pydantic, sample_functions):
        """Works with Pydantic model rows."""
        # This will have missing guidewords since it's only one row
        issues = _check_full_guideword_coverage([sample_l1_pydantic], sample_functions)
        assert len(issues) > 0


class TestCheckDuplicates:
    """Tests for _check_duplicates() helper."""

    def test_no_duplicates(self):
        """No duplicates returns empty list."""
        rows = [
            {"function": "A", "guideword": "no"},
            {"function": "A", "guideword": "more"},
        ]
        issues = _check_duplicates(rows, ["function", "guideword"])
        assert len(issues) == 0

    def test_duplicates_detected(self):
        """Duplicates are detected."""
        rows = [
            {"function": "A", "guideword": "no"},
            {"function": "A", "guideword": "no"},  # Duplicate
        ]
        issues = _check_duplicates(rows, ["function", "guideword"])
        assert len(issues) == 1
        assert "duplicate" in issues[0]


class TestCheckNonEmpty:
    """Tests for _check_non_empty() helper."""

    def test_non_empty_no_issues(self):
        """Non-empty fields return no issues."""
        rows = [{"row_id": "L1-0", "field": "value"}]
        issues = _check_non_empty(rows, "field")
        assert len(issues) == 0

    def test_empty_field_detected(self):
        """Empty field is detected."""
        rows = [{"row_id": "L1-0", "field": ""}]
        issues = _check_non_empty(rows, "field")
        assert len(issues) == 1
        assert "empty 'field'" in issues[0]

    def test_none_field_detected(self):
        """None field is detected."""
        rows = [{"row_id": "L1-0", "field": None}]
        issues = _check_non_empty(rows, "field")
        assert len(issues) == 1

    def test_includes_row_id_in_message(self):
        """Issue message includes row_id."""
        rows = [{"row_id": "L1-5", "field": ""}]
        issues = _check_non_empty(rows, "field")
        assert "L1-5" in issues[0]


class TestBuildValidatorReport:
    """Tests for build_validator_report() function."""

    def test_basic_report(self):
        """Basic report structure is correct."""
        report = build_validator_report("L1", ["issue1", "issue2"])
        assert report["stage"] == "L1"
        assert report["errors"] == ["issue1", "issue2"]
        assert report["hint"] == ""

    def test_report_with_hint(self):
        """Report includes hint when provided."""
        report = build_validator_report("L2", ["issue"], "Fix the cause")
        assert report["hint"] == "Fix the cause"

    def test_empty_issues(self):
        """Report handles empty issues list."""
        report = build_validator_report("L3", [])
        assert report["errors"] == []
