# tests/test_models_extended.py
"""Tests for DictLikeMixin __contains__ and keys() methods added in Phase 4."""
import pytest

from src.models import HazopL1Row, HazopL2Row, HazopL3Row, Guideword


class TestDictLikeContains:
    """Test __contains__ (the 'in' operator) on row models."""

    def test_contains_existing_field(self, sample_l1_pydantic):
        assert "function" in sample_l1_pydantic
        assert "guideword" in sample_l1_pydantic
        assert "deviation" in sample_l1_pydantic
        assert "row_id" in sample_l1_pydantic

    def test_contains_nonexistent_field(self, sample_l1_pydantic):
        assert "bogus" not in sample_l1_pydantic
        assert "nonexistent" not in sample_l1_pydantic

    def test_contains_l2_fields(self, sample_l2_pydantic):
        assert "cause" in sample_l2_pydantic
        assert "function" in sample_l2_pydantic  # inherited from L1

    def test_contains_l3_fields(self, sample_l3_pydantic):
        assert "effect" in sample_l3_pydantic
        assert "potentially_dangerous" in sample_l3_pydantic
        assert "cause" in sample_l3_pydantic  # inherited from L2


class TestDictLikeKeys:
    """Test keys() method on row models."""

    def test_keys_returns_model_fields_l1(self, sample_l1_pydantic):
        keys = set(sample_l1_pydantic.keys())
        assert "row_id" in keys
        assert "function" in keys
        assert "guideword" in keys
        assert "deviation" in keys

    def test_keys_returns_model_fields_l2(self, sample_l2_pydantic):
        keys = set(sample_l2_pydantic.keys())
        assert "cause" in keys
        assert "function" in keys

    def test_keys_returns_model_fields_l3(self, sample_l3_pydantic):
        keys = set(sample_l3_pydantic.keys())
        assert "effect" in keys
        assert "potentially_dangerous" in keys
        assert "cause" in keys
        assert "function" in keys

    def test_keys_on_each_row_type(self):
        """Verify keys() returns correct field counts for each model."""
        l1 = HazopL1Row(row_id="L1-0", function="F", guideword=Guideword.NO, deviation="D")
        l2 = HazopL2Row(row_id="L2-0", function="F", guideword=Guideword.NO, deviation="D", cause="C")
        l3 = HazopL3Row(row_id="L3-0", function="F", guideword=Guideword.NO, deviation="D",
                        cause="C", effect="E", potentially_dangerous=True)

        l1_keys = set(l1.keys())
        l2_keys = set(l2.keys())
        l3_keys = set(l3.keys())

        # L2 has all L1 keys plus "cause"
        assert l1_keys < l2_keys
        assert "cause" in l2_keys - l1_keys

        # L3 has all L2 keys plus "effect" and "potentially_dangerous"
        assert l2_keys < l3_keys
        assert {"effect", "potentially_dangerous"} <= l3_keys - l2_keys
