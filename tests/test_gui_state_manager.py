# tests/test_gui_state_manager.py
"""Tests for StateManager and RowState — in-memory row tracking, canonical ID
assignment, import, editing, rating, and regeneration updates."""
import pytest

from src.gui.services.state_manager import StateManager, RowState, _prettify
from src.gui.models import AnalysisStatus, Rating


# ============================================================================
#                         _prettify helper
# ============================================================================

class TestPrettify:
    def test_colon_cause(self):
        result = _prettify("Leak: fluid loss", kind="cause")
        assert "that causes" in result
        assert ":" not in result

    def test_colon_effect(self):
        result = _prettify("Impact: vehicle stops", kind="effect")
        assert "which results in" in result
        assert ":" not in result

    def test_colon_other(self):
        result = _prettify("Label: value", kind="deviation")
        assert "Label - value" in result
        assert ":" not in result

    def test_empty_string(self):
        assert _prettify("", kind="cause") == ""

    def test_none_value(self):
        assert _prettify(None, kind="cause") == ""

    def test_adds_period(self):
        result = _prettify("No braking", kind="deviation")
        assert result.endswith(".")

    def test_does_not_double_period(self):
        result = _prettify("Already ends.", kind="deviation")
        assert result == "Already ends."
        assert not result.endswith("..")

    def test_preserves_exclamation(self):
        result = _prettify("Critical failure!", kind="effect")
        assert result.endswith("!")

    def test_collapses_whitespace(self):
        result = _prettify("  too   many   spaces  ", kind="deviation")
        assert "  " not in result


# ============================================================================
#                         RowState
# ============================================================================

class TestRowState:
    def _make_row(self, **overrides):
        defaults = dict(
            row_id="L3-0", function="Braking", guideword="no",
            deviation_original="D", deviation_final="D",
            cause_original="C", cause_final="C",
            effect_original="E", effect_final="E",
            potentially_dangerous_ai=True,
        )
        defaults.update(overrides)
        return RowState(**defaults)

    def test_to_row_data_returns_correct_type(self):
        row = self._make_row()
        rd = row.to_row_data()
        assert rd.row_id == "L3-0"
        assert rd.function == "Braking"
        assert rd.potentially_dangerous_final is True

    def test_to_dict_for_export_includes_all_fields(self):
        row = self._make_row()
        d = row.to_dict_for_export()
        expected_keys = {
            "row_id", "function", "guideword",
            "deviation_original", "deviation_final",
            "cause_original", "cause_final",
            "effect_original", "effect_final",
            "potentially_dangerous_ai", "potentially_dangerous_human",
            "potentially_dangerous_final",
            "rating", "edited_flag", "regenerated_flag",
        }
        assert set(d.keys()) == expected_keys

    def test_dangerous_final_uses_ai_by_default(self):
        row = self._make_row(potentially_dangerous_ai=True)
        assert row.potentially_dangerous_final is True

    def test_dangerous_final_human_override_true(self):
        row = self._make_row(potentially_dangerous_ai=False, potentially_dangerous_human=True)
        assert row.potentially_dangerous_final is True

    def test_dangerous_final_human_override_false(self):
        row = self._make_row(potentially_dangerous_ai=True, potentially_dangerous_human=False)
        assert row.potentially_dangerous_final is False


# ============================================================================
#                         StateManager.create_run / get_run
# ============================================================================

class TestCreateAndGetRun:
    def test_create_run_returns_id(self):
        sm = StateManager()
        run_id = sm.create_run(
            provider="openai", model="gpt-4", functions=["Braking"],
            notes="test", max_devs_per_gw=2,
        )
        assert isinstance(run_id, str)
        assert len(run_id) == 8

    def test_get_run_returns_run(self):
        sm = StateManager()
        run_id = sm.create_run(
            provider="openai", model="gpt-4", functions=["Braking"],
            notes="", max_devs_per_gw=1,
        )
        run = sm.get_run(run_id)
        assert run is not None
        assert run.provider == "openai"

    def test_get_run_unknown_returns_none(self):
        sm = StateManager()
        assert sm.get_run("nonexistent") is None


# ============================================================================
#                         StateManager.update_status
# ============================================================================

class TestUpdateStatus:
    def test_updates_status(self):
        sm = StateManager()
        run_id = sm.create_run(provider="openai", model="m", functions=["F"],
                               notes="", max_devs_per_gw=1)
        sm.update_status(run_id, AnalysisStatus.RUNNING, "Step 1")
        run = sm.get_run(run_id)
        assert run.status == AnalysisStatus.RUNNING
        assert run.progress == "Step 1"

    def test_sets_completed_at(self):
        sm = StateManager()
        run_id = sm.create_run(provider="openai", model="m", functions=["F"],
                               notes="", max_devs_per_gw=1)
        assert sm.get_run(run_id).completed_at is None
        sm.update_status(run_id, AnalysisStatus.COMPLETED)
        assert sm.get_run(run_id).completed_at is not None


# ============================================================================
#                  StateManager.initialize_rows_from_pipeline
# ============================================================================

class TestInitializeFromPipeline:
    def _sample_pipeline_data(self):
        l1 = [
            {"row_id": "L1-0", "function": "Braking", "guideword": "no",
             "deviation": "No braking"},
            {"row_id": "L1-1", "function": "Braking", "guideword": "more",
             "deviation": "Excessive braking"},
        ]
        l2 = [
            {"row_id": "L2-0", "function": "Braking", "guideword": "no",
             "deviation": "No braking", "cause": "Hydraulic leak"},
            {"row_id": "L2-1", "function": "Braking", "guideword": "more",
             "deviation": "Excessive braking", "cause": "Sensor fault"},
        ]
        l3 = [
            {"row_id": "L3-0", "function": "Braking", "guideword": "no",
             "deviation": "No braking", "cause": "Hydraulic leak",
             "effect": "Cannot stop", "potentially_dangerous": True},
            {"row_id": "L3-1", "function": "Braking", "guideword": "more",
             "deviation": "Excessive braking", "cause": "Sensor fault",
             "effect": "Wheel lockup", "potentially_dangerous": False},
        ]
        return l1, l2, l3

    def test_creates_row_states(self):
        sm = StateManager()
        run_id = sm.create_run(provider="openai", model="m", functions=["Braking"],
                               notes="", max_devs_per_gw=1)
        l1, l2, l3 = self._sample_pipeline_data()
        sm.initialize_rows_from_pipeline(run_id, l1, l2, l3)
        rows = sm.get_all_rows(run_id)
        assert len(rows) == 2

    def test_canonical_ids_from_l3(self):
        sm = StateManager()
        run_id = sm.create_run(provider="openai", model="m", functions=["Braking"],
                               notes="", max_devs_per_gw=1)
        l1, l2, l3 = self._sample_pipeline_data()
        sm.initialize_rows_from_pipeline(run_id, l1, l2, l3)
        rows = sm.get_all_rows(run_id)
        row_ids = [r.row_id for r in rows]
        assert "L3-0" in row_ids
        assert "L3-1" in row_ids

    def test_stores_raw_rows(self):
        sm = StateManager()
        run_id = sm.create_run(provider="openai", model="m", functions=["Braking"],
                               notes="", max_devs_per_gw=1)
        l1, l2, l3 = self._sample_pipeline_data()
        sm.initialize_rows_from_pipeline(run_id, l1, l2, l3)
        raw_l1, raw_l2, raw_l3 = sm.get_raw_rows(run_id)
        assert len(raw_l1) == 2
        assert len(raw_l2) == 2
        assert len(raw_l3) == 2


# ============================================================================
#                  StateManager.initialize_rows_from_import
# ============================================================================

class TestInitializeFromImport:
    def test_creates_rows_from_import(self):
        sm = StateManager()
        run_id = sm.create_run(provider="imported", model="imported",
                               functions=["F"], notes="", max_devs_per_gw=1)
        imported = [
            {"function": "Braking", "guideword": "no", "deviation": "D",
             "cause": "C", "effect": "E", "potentially_dangerous": True},
        ]
        sm.initialize_rows_from_import(run_id, imported)
        rows = sm.get_all_rows(run_id)
        assert len(rows) == 1
        assert rows[0].row_id == "L3-0"

    def test_synthesizes_raw_l1_l2_l3(self):
        sm = StateManager()
        run_id = sm.create_run(provider="imported", model="imported",
                               functions=["F"], notes="", max_devs_per_gw=1)
        imported = [{"function": "F", "guideword": "no", "deviation": "D",
                     "cause": "C", "effect": "E", "potentially_dangerous": False}]
        sm.initialize_rows_from_import(run_id, imported)
        raw_l1, raw_l2, raw_l3 = sm.get_raw_rows(run_id)
        assert len(raw_l1) == 1
        assert raw_l1[0]["row_id"] == "L1-0"
        assert raw_l2[0]["row_id"] == "L2-0"
        assert raw_l3[0]["row_id"] == "L3-0"

    def test_sets_completed_status(self):
        sm = StateManager()
        run_id = sm.create_run(provider="imported", model="imported",
                               functions=["F"], notes="", max_devs_per_gw=1)
        sm.initialize_rows_from_import(run_id, [
            {"function": "F", "guideword": "no", "deviation": "D",
             "cause": "C", "effect": "E", "potentially_dangerous": False}
        ])
        run = sm.get_run(run_id)
        assert run.status == AnalysisStatus.COMPLETED

    def test_preserves_rating(self):
        sm = StateManager()
        run_id = sm.create_run(provider="imported", model="imported",
                               functions=["F"], notes="", max_devs_per_gw=1)
        sm.initialize_rows_from_import(run_id, [
            {"function": "F", "guideword": "no", "deviation": "D",
             "cause": "C", "effect": "E", "potentially_dangerous": False,
             "rating": "correct"}
        ])
        rows = sm.get_all_rows(run_id)
        assert rows[0].rating == Rating.CORRECT


# ============================================================================
#                  StateManager.update_row_fields
# ============================================================================

class TestUpdateRowFields:
    def _setup(self):
        sm = StateManager()
        run_id = sm.create_run(provider="openai", model="m", functions=["F"],
                               notes="", max_devs_per_gw=1)
        run = sm.get_run(run_id)
        run.rows["L3-0"] = RowState(
            row_id="L3-0", function="F", guideword="no",
            deviation_original="D", deviation_final="D",
            cause_original="C", cause_final="C",
            effect_original="E", effect_final="E",
            potentially_dangerous_ai=False,
        )
        return sm, run_id

    def test_updates_final_values(self):
        sm, run_id = self._setup()
        row = sm.update_row_fields(run_id, "L3-0", deviation="New D")
        assert row.deviation_final == "New D"
        assert row.deviation_original == "D"  # original unchanged

    def test_sets_edited_flag(self):
        sm, run_id = self._setup()
        row = sm.update_row_fields(run_id, "L3-0", cause="New C")
        assert row.edited_flag is True

    def test_no_change_no_flag(self):
        sm, run_id = self._setup()
        row = sm.update_row_fields(run_id, "L3-0", deviation="D")  # same value
        assert row.edited_flag is False

    def test_dangerous_override(self):
        sm, run_id = self._setup()
        row = sm.update_row_fields(run_id, "L3-0", potentially_dangerous=True)
        assert row.potentially_dangerous_human is True
        assert row.potentially_dangerous_final is True

    def test_nonexistent_row_returns_none(self):
        sm, run_id = self._setup()
        assert sm.update_row_fields(run_id, "L3-99") is None


# ============================================================================
#                  StateManager.update_row_rating
# ============================================================================

class TestUpdateRowRating:
    def test_updates_rating(self):
        sm = StateManager()
        run_id = sm.create_run(provider="p", model="m", functions=["F"],
                               notes="", max_devs_per_gw=1)
        run = sm.get_run(run_id)
        run.rows["L3-0"] = RowState(
            row_id="L3-0", function="F", guideword="no",
            deviation_original="D", deviation_final="D",
            cause_original="C", cause_final="C",
            effect_original="E", effect_final="E",
            potentially_dangerous_ai=False,
        )
        row = sm.update_row_rating(run_id, "L3-0", Rating.CORRECT)
        assert row.rating == Rating.CORRECT

    def test_nonexistent_row_returns_none(self):
        sm = StateManager()
        run_id = sm.create_run(provider="p", model="m", functions=["F"],
                               notes="", max_devs_per_gw=1)
        assert sm.update_row_rating(run_id, "L3-99", Rating.CORRECT) is None


# ============================================================================
#                  StateManager.update_rows_after_regeneration
# ============================================================================

class TestUpdateRowsAfterRegeneration:
    def _setup(self):
        sm = StateManager()
        run_id = sm.create_run(provider="p", model="m", functions=["F"],
                               notes="", max_devs_per_gw=1)
        run = sm.get_run(run_id)
        run.rows["L3-0"] = RowState(
            row_id="L3-0", function="F", guideword="no",
            deviation_original="D", deviation_final="D",
            cause_original="C", cause_final="C",
            effect_original="E", effect_final="E",
            potentially_dangerous_ai=False,
        )
        return sm, run_id

    def test_updates_final_values(self):
        sm, run_id = self._setup()
        regen = [{"row_id": "L3-0", "deviation": "New D", "cause": "New C",
                  "effect": "New E", "potentially_dangerous": True}]
        updated = sm.update_rows_after_regeneration(run_id, regen, ["L3-0"])
        assert len(updated) == 1
        assert updated[0].regenerated_flag is True

    def test_sets_regenerated_flag(self):
        sm, run_id = self._setup()
        regen = [{"row_id": "L3-0", "effect": "Changed"}]
        updated = sm.update_rows_after_regeneration(run_id, regen, ["L3-0"])
        assert updated[0].regenerated_flag is True


# ============================================================================
#                  StateManager.get_all_rows / get_raw_rows
# ============================================================================

class TestGetAllRows:
    def test_returns_sorted_by_suffix(self):
        sm = StateManager()
        run_id = sm.create_run(provider="p", model="m", functions=["F"],
                               notes="", max_devs_per_gw=1)
        run = sm.get_run(run_id)
        for i in [2, 0, 1]:
            run.rows[f"L3-{i}"] = RowState(
                row_id=f"L3-{i}", function="F", guideword="no",
                deviation_original="D", deviation_final="D",
                cause_original="C", cause_final="C",
                effect_original="E", effect_final="E",
                potentially_dangerous_ai=False,
            )
        rows = sm.get_all_rows(run_id)
        assert [r.row_id for r in rows] == ["L3-0", "L3-1", "L3-2"]

    def test_empty_run_returns_empty(self):
        sm = StateManager()
        assert sm.get_all_rows("nonexistent") == []


class TestGetRawRows:
    def test_returns_empty_for_unknown_run(self):
        sm = StateManager()
        assert sm.get_raw_rows("nonexistent") == ([], [], [])
