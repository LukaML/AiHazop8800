# tests/test_gui_routes.py
"""Tests for FastAPI endpoints — settings, analysis, rows, export.

All LLM calls and pipeline execution are mocked. Uses httpx.AsyncClient
via pytest-asyncio for async endpoint testing.
"""
import io
import json
import pytest
from unittest.mock import patch, MagicMock

from fastapi.testclient import TestClient

from src.gui.app import app
from src.gui.config import RuntimeConfig
from src.gui.services.state_manager import StateManager, RowState
from src.gui.models import AnalysisStatus, Rating


@pytest.fixture(autouse=True)
def _clean_keys():
    RuntimeConfig.clear_all_keys()
    yield
    RuntimeConfig.clear_all_keys()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def sm_with_completed_run():
    """Create a StateManager with a completed run and return (sm, run_id)."""
    sm = StateManager()
    run_id = sm.create_run(
        provider="openai", model="gpt-4", functions=["Braking"],
        notes="test", max_devs_per_gw=1,
    )
    sm.update_status(run_id, AnalysisStatus.COMPLETED)

    run = sm.get_run(run_id)
    run.rows["L3-0"] = RowState(
        row_id="L3-0", function="Braking", guideword="no",
        deviation_original="D orig", deviation_final="D final",
        cause_original="C orig", cause_final="C final",
        effect_original="E orig", effect_final="E final",
        potentially_dangerous_ai=True, rating=Rating.UNRATED,
    )
    return sm, run_id


# ============================================================================
#                         SETTINGS ROUTES
# ============================================================================

class TestSettingsRoutes:
    def test_set_api_key(self, client):
        resp = client.post("/api/settings/api-key", json={
            "provider": "openai", "api_key": "sk-test-123"
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["provider"] == "openai"

    def test_get_providers(self, client):
        resp = client.get("/api/settings/providers")
        assert resp.status_code == 200
        data = resp.json()
        providers = {p["provider"] for p in data["providers"]}
        assert providers == {"openai", "gemini", "groq"}

    def test_set_invalid_provider(self, client):
        resp = client.post("/api/settings/api-key", json={
            "provider": "invalid_provider", "api_key": "key"
        })
        assert resp.status_code == 422  # Pydantic validation error


# ============================================================================
#                         ANALYSIS ROUTES
# ============================================================================

class TestAnalysisRoutes:
    def test_start_analysis_no_key_fails(self, client):
        resp = client.post("/api/analysis/start", json={
            "provider": "openai",
            "functions": ["Braking"],
            "notes": "test",
        })
        assert resp.status_code == 400
        assert "API key not configured" in resp.json()["detail"]

    @patch("src.gui.routes.analysis.pipeline_adapter")
    def test_start_analysis_success(self, mock_adapter, client):
        RuntimeConfig.set_api_key("openai", "sk-test")
        mock_adapter.run_analysis_async = MagicMock()

        resp = client.post("/api/analysis/start", json={
            "provider": "openai",
            "functions": ["Braking"],
            "notes": "test",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "pending"
        assert "run_id" in data

    @patch("src.gui.routes.analysis.state_manager")
    def test_get_status(self, mock_sm, client, sm_with_completed_run):
        sm, run_id = sm_with_completed_run
        mock_sm.get_run = sm.get_run

        resp = client.get(f"/api/analysis/{run_id}/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "completed"

    @patch("src.gui.routes.analysis.state_manager")
    def test_get_status_not_found(self, mock_sm, client):
        mock_sm.get_run.return_value = None
        resp = client.get("/api/analysis/nonexistent/status")
        assert resp.status_code == 404

    @patch("src.gui.routes.analysis.state_manager")
    def test_get_results_completed(self, mock_sm, client, sm_with_completed_run):
        sm, run_id = sm_with_completed_run
        mock_sm.get_run = sm.get_run
        mock_sm.get_all_rows = sm.get_all_rows

        resp = client.get(f"/api/analysis/{run_id}/results")
        assert resp.status_code == 200
        data = resp.json()
        assert data["run_id"] == run_id
        assert len(data["rows"]) == 1

    def test_upload_functions_txt(self, client):
        content = b"Braking\nSteering\n"
        resp = client.post(
            "/api/analysis/upload-functions",
            files={"file": ("funcs.txt", io.BytesIO(content), "text/plain")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["functions"] == ["Braking", "Steering"]

    def test_upload_functions_json(self, client):
        content = json.dumps(["Braking", "Steering"]).encode()
        resp = client.post(
            "/api/analysis/upload-functions",
            files={"file": ("funcs.json", io.BytesIO(content), "application/json")},
        )
        assert resp.status_code == 200
        assert resp.json()["functions"] == ["Braking", "Steering"]

    @patch("src.gui.routes.analysis.state_manager")
    def test_import_html(self, mock_sm, client):
        sm = StateManager()

        # Wire mock to real state manager
        mock_sm.create_run = sm.create_run
        mock_sm.initialize_rows_from_import = sm.initialize_rows_from_import

        html = """<html><body><table><thead><tr>
        <th>#</th><th>Function</th><th>Guideword</th><th>Deviation</th>
        <th>Cause</th><th>Effect</th><th>Dangerous</th></tr></thead>
        <tbody>
        <tr><td>1</td><td>Braking</td><td>no</td><td>D</td><td>C</td>
        <td>E</td><td>Dangerous</td></tr>
        </tbody></table></body></html>"""

        resp = client.post(
            "/api/analysis/import-html",
            files={"file": ("report.html", io.BytesIO(html.encode()), "text/html")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["row_count"] == 1


# ============================================================================
#                         ROW ROUTES
# ============================================================================

class TestRowRoutes:
    @patch("src.gui.routes.rows.state_manager")
    def test_edit_row(self, mock_sm, client, sm_with_completed_run):
        sm, run_id = sm_with_completed_run
        mock_sm.get_run = sm.get_run
        mock_sm.update_row_fields = sm.update_row_fields
        mock_sm.get_row = sm.get_row

        resp = client.put(
            f"/api/rows/{run_id}/L3-0/edit",
            json={"deviation": "Updated deviation"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["row"]["deviation_final"] == "Updated deviation"

    @patch("src.gui.routes.rows.state_manager")
    def test_rate_row(self, mock_sm, client, sm_with_completed_run):
        sm, run_id = sm_with_completed_run
        mock_sm.get_run = sm.get_run
        mock_sm.update_row_rating = sm.update_row_rating
        mock_sm.get_row = sm.get_row

        resp = client.put(
            f"/api/rows/{run_id}/L3-0/rate",
            json={"rating": "correct"},
        )
        assert resp.status_code == 200
        assert resp.json()["rating"] == "correct"

    @patch("src.gui.routes.rows.state_manager")
    def test_edit_nonexistent_run(self, mock_sm, client):
        mock_sm.get_run.return_value = None
        resp = client.put(
            "/api/rows/nope/L3-0/edit",
            json={"deviation": "X"},
        )
        assert resp.status_code == 404


# ============================================================================
#                         EXPORT ROUTES
# ============================================================================

class TestExportRoutes:
    @patch("src.gui.routes.export.state_manager")
    @patch("src.gui.routes.export.export_to_csv")
    def test_export_csv(self, mock_csv, mock_sm, client, sm_with_completed_run):
        sm, run_id = sm_with_completed_run
        mock_sm.get_run = sm.get_run
        mock_csv.return_value = "col1,col2\nval1,val2\n"

        resp = client.get(f"/api/export/{run_id}/csv")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")

    @patch("src.gui.routes.export.state_manager")
    @patch("src.gui.routes.export.export_to_html")
    def test_export_html(self, mock_html, mock_sm, client, sm_with_completed_run):
        sm, run_id = sm_with_completed_run
        mock_sm.get_run = sm.get_run
        mock_html.return_value = "<html><body>test</body></html>"

        resp = client.get(f"/api/export/{run_id}/html")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")

    @patch("src.gui.routes.export.state_manager")
    def test_export_csv_unknown_run(self, mock_sm, client):
        mock_sm.get_run.return_value = None
        resp = client.get("/api/export/nonexistent/csv")
        assert resp.status_code == 404
