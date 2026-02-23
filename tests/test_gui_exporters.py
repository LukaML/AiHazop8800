# tests/test_gui_exporters.py
"""Tests for CSV and HTML GUI exporters."""
import csv
import io
import pytest

from src.gui.services.state_manager import StateManager, RowState, _prettify
from src.gui.models import AnalysisStatus, Rating
from src.gui.services.csv_exporter import export_to_csv, CSV_COLUMNS
from src.gui.services.html_exporter import export_to_html, _get_rating_display


# ============================================================================
#                         Fixtures
# ============================================================================

@pytest.fixture
def sm_with_run():
    """StateManager with a completed run containing 2 rows."""
    sm = StateManager()

    run_id = sm.create_run(
        provider="openai", model="gpt-4", functions=["Braking"],
        notes="test", max_devs_per_gw=1,
    )
    sm.update_status(run_id, AnalysisStatus.COMPLETED)

    # Manually add rows
    run = sm.get_run(run_id)
    run.rows["L3-0"] = RowState(
        row_id="L3-0", function="Braking", guideword="no",
        deviation_original="No braking.", deviation_final="No braking.",
        cause_original="Hydraulic leak.", cause_final="Hydraulic leak.",
        effect_original="Cannot stop.", effect_final="Cannot stop.",
        potentially_dangerous_ai=True, rating=Rating.CORRECT,
    )
    run.rows["L3-1"] = RowState(
        row_id="L3-1", function="Braking", guideword="more",
        deviation_original="Excessive force.", deviation_final="Excessive force.",
        cause_original="Sensor fault.", cause_final="Sensor fault.",
        effect_original="Wheel lockup.", effect_final="Wheel lockup.",
        potentially_dangerous_ai=False, rating=Rating.UNRATED,
    )

    # Patch the global state_manager used by exporters
    return sm, run_id


# ============================================================================
#                         CSV Export Tests
# ============================================================================

class TestCsvExport:
    def test_header_row(self, sm_with_run, monkeypatch):
        sm, run_id = sm_with_run
        monkeypatch.setattr("src.gui.services.csv_exporter.state_manager", sm)
        content = export_to_csv(run_id)
        reader = csv.DictReader(io.StringIO(content))
        assert set(reader.fieldnames) == set(CSV_COLUMNS)

    def test_row_count(self, sm_with_run, monkeypatch):
        sm, run_id = sm_with_run
        monkeypatch.setattr("src.gui.services.csv_exporter.state_manager", sm)
        content = export_to_csv(run_id)
        reader = list(csv.DictReader(io.StringIO(content)))
        assert len(reader) == 2

    def test_contains_all_columns(self, sm_with_run, monkeypatch):
        sm, run_id = sm_with_run
        monkeypatch.setattr("src.gui.services.csv_exporter.state_manager", sm)
        content = export_to_csv(run_id)
        reader = list(csv.DictReader(io.StringIO(content)))
        row = reader[0]
        assert row["function"] == "Braking"
        assert row["provider"] == "openai"
        assert row["model"] == "gpt-4"
        assert row["rating"] == "correct"

    def test_unknown_run_raises(self, monkeypatch):
        sm = StateManager()
        monkeypatch.setattr("src.gui.services.csv_exporter.state_manager", sm)
        with pytest.raises(ValueError, match="Run not found"):
            export_to_csv("nonexistent")


# ============================================================================
#                         HTML Export Tests
# ============================================================================

class TestHtmlExport:
    def test_contains_table(self, sm_with_run, monkeypatch):
        sm, run_id = sm_with_run
        monkeypatch.setattr("src.gui.services.html_exporter.state_manager", sm)
        html = export_to_html(run_id)
        assert "<table>" in html
        assert "<tbody>" in html

    def test_dangerous_highlighted(self, sm_with_run, monkeypatch):
        sm, run_id = sm_with_run
        monkeypatch.setattr("src.gui.services.html_exporter.state_manager", sm)
        html = export_to_html(run_id)
        assert "Dangerous" in html
        assert "Not dangerous" in html

    def test_rating_displayed(self, sm_with_run, monkeypatch):
        sm, run_id = sm_with_run
        monkeypatch.setattr("src.gui.services.html_exporter.state_manager", sm)
        html = export_to_html(run_id)
        assert "Correct" in html

    def test_unknown_run_raises(self, monkeypatch):
        sm = StateManager()
        monkeypatch.setattr("src.gui.services.html_exporter.state_manager", sm)
        with pytest.raises(ValueError, match="Run not found"):
            export_to_html("nonexistent")


# ============================================================================
#                         _get_rating_display
# ============================================================================

class TestGetRatingDisplay:
    def test_correct(self):
        text, cls = _get_rating_display(Rating.CORRECT)
        assert text == "Correct"
        assert cls == "rating-correct"

    def test_partial(self):
        text, cls = _get_rating_display(Rating.PARTIALLY_CORRECT)
        assert text == "Partial"
        assert cls == "rating-partial"

    def test_incorrect(self):
        text, cls = _get_rating_display(Rating.INCORRECT)
        assert text == "Incorrect"
        assert cls == "rating-incorrect"

    def test_unrated(self):
        text, cls = _get_rating_display(Rating.UNRATED)
        assert text == "-"
        assert cls == "rating-unrated"

    def test_none(self):
        text, cls = _get_rating_display(None)
        assert text == "-"
        assert cls == "rating-unrated"
