# tests/test_models.py
"""Tests for src/models.py Pydantic models and enums."""
import pytest
from pydantic import ValidationError

from src.models import (
    Guideword,
    HazopL1Row,
    HazopL2Row,
    HazopL3Row,
    DictLikeMixin,
)


class TestGuidewordEnum:
    """Tests for Guideword enum."""

    def test_all_11_guidewords_present(self):
        """All 11 HAZOP guidewords are defined."""
        expected = {
            "no", "more", "less", "as well as", "part of",
            "reverse", "other than", "early", "late", "before", "after"
        }
        actual = {g.value for g in Guideword}
        assert actual == expected

    def test_guideword_string_value(self):
        """Guideword enum values are strings."""
        assert Guideword.NO.value == "no"
        assert Guideword.AS_WELL_AS.value == "as well as"


class TestHazopL1Row:
    """Tests for HazopL1Row model."""

    def test_valid_l1_row(self):
        """Valid L1 row is accepted."""
        row = HazopL1Row(
            row_id="L1-0",
            function="Braking",
            guideword=Guideword.NO,
            deviation="No braking action",
        )
        assert row.function == "Braking"
        assert row.guideword == Guideword.NO

    def test_l1_row_with_string_guideword(self):
        """L1 row accepts string guideword value."""
        row = HazopL1Row(
            row_id="L1-0",
            function="Braking",
            guideword="no",
            deviation="No braking action",
        )
        assert row.guideword == Guideword.NO

    def test_empty_row_id_rejected(self):
        """Empty row_id is rejected."""
        with pytest.raises(ValidationError):
            HazopL1Row(
                row_id="",
                function="Braking",
                guideword=Guideword.NO,
                deviation="Test",
            )

    def test_empty_function_rejected(self):
        """Empty function is rejected."""
        with pytest.raises(ValidationError):
            HazopL1Row(
                row_id="L1-0",
                function="",
                guideword=Guideword.NO,
                deviation="Test",
            )

    def test_empty_deviation_rejected(self):
        """Empty deviation is rejected."""
        with pytest.raises(ValidationError):
            HazopL1Row(
                row_id="L1-0",
                function="Braking",
                guideword=Guideword.NO,
                deviation="",
            )

    def test_optional_applicability_fields(self):
        """Optional applicability fields are accepted."""
        row = HazopL1Row(
            row_id="L1-0",
            function="Braking",
            guideword=Guideword.NO,
            deviation="Test",
            applicability_score=0.8,
            applicability_note="High relevance",
        )
        assert row.applicability_score == 0.8
        assert row.applicability_note == "High relevance"

    def test_applicability_score_bounds(self):
        """Applicability score must be between 0 and 1."""
        with pytest.raises(ValidationError):
            HazopL1Row(
                row_id="L1-0",
                function="Braking",
                guideword=Guideword.NO,
                deviation="Test",
                applicability_score=1.5,
            )


class TestHazopL2Row:
    """Tests for HazopL2Row model."""

    def test_valid_l2_row(self):
        """Valid L2 row is accepted."""
        row = HazopL2Row(
            row_id="L2-0",
            function="Braking",
            guideword=Guideword.NO,
            deviation="No braking action",
            cause="Hydraulic failure",
        )
        assert row.cause == "Hydraulic failure"

    def test_l2_inherits_l1_fields(self):
        """L2 row has all L1 fields."""
        row = HazopL2Row(
            row_id="L2-0",
            function="Braking",
            guideword=Guideword.NO,
            deviation="No braking action",
            cause="Test cause",
        )
        assert row.row_id == "L2-0"
        assert row.function == "Braking"
        assert row.deviation == "No braking action"

    def test_empty_cause_rejected(self):
        """Empty cause is rejected."""
        with pytest.raises(ValidationError):
            HazopL2Row(
                row_id="L2-0",
                function="Braking",
                guideword=Guideword.NO,
                deviation="Test",
                cause="",
            )


class TestHazopL3Row:
    """Tests for HazopL3Row model."""

    def test_valid_l3_row(self):
        """Valid L3 row is accepted."""
        row = HazopL3Row(
            row_id="L3-0",
            function="Braking",
            guideword=Guideword.NO,
            deviation="No braking action",
            cause="Hydraulic failure",
            effect="Vehicle cannot stop",
            potentially_dangerous=True,
        )
        assert row.effect == "Vehicle cannot stop"
        assert row.potentially_dangerous is True

    def test_l3_inherits_l2_fields(self):
        """L3 row has all L2 fields."""
        row = HazopL3Row(
            row_id="L3-0",
            function="Braking",
            guideword=Guideword.NO,
            deviation="No braking action",
            cause="Test cause",
            effect="Test effect",
            potentially_dangerous=False,
        )
        assert row.cause == "Test cause"

    def test_empty_effect_rejected(self):
        """Empty effect is rejected."""
        with pytest.raises(ValidationError):
            HazopL3Row(
                row_id="L3-0",
                function="Braking",
                guideword=Guideword.NO,
                deviation="Test",
                cause="Test cause",
                effect="",
                potentially_dangerous=True,
            )

    def test_missing_potentially_dangerous_rejected(self):
        """Missing potentially_dangerous field is rejected."""
        with pytest.raises(ValidationError):
            HazopL3Row(
                row_id="L3-0",
                function="Braking",
                guideword=Guideword.NO,
                deviation="Test",
                cause="Test cause",
                effect="Test effect",
                # Missing potentially_dangerous
            )

    def test_potentially_dangerous_false(self):
        """potentially_dangerous can be False."""
        row = HazopL3Row(
            row_id="L3-0",
            function="Braking",
            guideword=Guideword.NO,
            deviation="Test",
            cause="Test cause",
            effect="Minor inconvenience",
            potentially_dangerous=False,
        )
        assert row.potentially_dangerous is False

class TestDictLikeMixin:
    """Tests for DictLikeMixin functionality."""

    def test_get_method(self, sample_l1_pydantic):
        """get() method returns attribute value."""
        assert sample_l1_pydantic.get("function") == "Braking"
        assert sample_l1_pydantic.get("nonexistent", "default") == "default"

    def test_getitem_method(self, sample_l1_pydantic):
        """__getitem__ returns attribute value."""
        assert sample_l1_pydantic["function"] == "Braking"

    def test_setitem_method(self, sample_l1_pydantic):
        """__setitem__ sets attribute value."""
        sample_l1_pydantic["function"] = "Steering"
        assert sample_l1_pydantic.function == "Steering"

    def test_to_dict_method(self, sample_l1_pydantic):
        """to_dict() returns model as dict."""
        result = sample_l1_pydantic.to_dict()
        assert isinstance(result, dict)
        assert result["function"] == "Braking"


