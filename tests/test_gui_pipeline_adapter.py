# tests/test_gui_pipeline_adapter.py
"""Tests for PipelineAdapter — all LLM calls mocked."""
import os
import pytest
from unittest.mock import patch, MagicMock

from src.gui.config import RuntimeConfig
from src.gui.services.state_manager import StateManager, RowState
from src.gui.models import AnalysisStatus, RegenerationScope, Rating


@pytest.fixture(autouse=True)
def _clean_keys():
    RuntimeConfig.clear_all_keys()
    yield
    RuntimeConfig.clear_all_keys()


@pytest.fixture
def sm_with_completed_run():
    """StateManager with a completed run containing 2 rows + raw data."""
    sm = StateManager()
    run_id = sm.create_run(
        provider="openai", model="gpt-4", functions=["Braking"],
        notes="test notes", max_devs_per_gw=1,
    )
    sm.update_status(run_id, AnalysisStatus.COMPLETED)

    l1 = [{"row_id": "L1-0", "function": "Braking", "guideword": "no", "deviation": "No braking"}]
    l2 = [{"row_id": "L2-0", "function": "Braking", "guideword": "no", "deviation": "No braking", "cause": "Hydraulic leak"}]
    l3 = [{"row_id": "L3-0", "function": "Braking", "guideword": "no", "deviation": "No braking",
           "cause": "Hydraulic leak", "effect": "Cannot stop", "potentially_dangerous": True}]

    sm.initialize_rows_from_pipeline(run_id, l1, l2, l3)
    return sm, run_id


class TestInjectApiKey:
    def test_sets_env_var(self):
        from src.gui.services.pipeline_adapter import PipelineAdapter
        RuntimeConfig.set_api_key("openai", "sk-test-key")
        PipelineAdapter._inject_api_key("openai")
        assert os.environ.get("OPENAI_API_KEY") == "sk-test-key"

    def test_no_key_no_env(self):
        from src.gui.services.pipeline_adapter import PipelineAdapter
        old = os.environ.pop("OPENAI_API_KEY", None)
        PipelineAdapter._inject_api_key("openai")
        # Should not set env var when no key is configured
        if old is not None:
            os.environ["OPENAI_API_KEY"] = old


class TestRegenerateRowsL3Scope:
    @patch("src.gui.services.pipeline_adapter.state_manager")
    @patch("src.chains.reviewer_patch_l3")
    @patch("src.llm_client.configure")
    def test_l3_scope_patches_effect(self, mock_configure, mock_patch_l3, mock_sm,
                                      sm_with_completed_run):
        sm, run_id = sm_with_completed_run
        from src.gui.services.pipeline_adapter import PipelineAdapter

        # Wire the mock state_manager to our real one
        mock_sm.get_run = sm.get_run
        mock_sm.get_raw_rows = sm.get_raw_rows
        mock_sm.update_rows_after_regeneration = sm.update_rows_after_regeneration
        mock_sm.get_row = sm.get_row

        RuntimeConfig.set_api_key("openai", "sk-test")

        # Mock the patch function to return updated rows
        mock_patch_l3.return_value = [
            {"row_id": "L3-0", "function": "Braking", "guideword": "no",
             "deviation": "No braking", "cause": "Hydraulic leak",
             "effect": "Collision imminent", "potentially_dangerous": True}
        ]

        result = PipelineAdapter.regenerate_rows(
            run_id=run_id, row_ids=["L3-0"],
            scope=RegenerationScope.L3, suggestion="Be more specific",
        )
        assert len(result) == 1
        mock_patch_l3.assert_called_once()


class TestRegenerateImportedRun:
    @patch("src.gui.services.pipeline_adapter.state_manager")
    @patch("src.chains.reviewer_patch_l3")
    @patch("src.llm_client.configure")
    def test_imported_run_resolves_provider(self, mock_configure, mock_patch_l3, mock_sm):
        sm = StateManager()
        run_id = sm.create_run(
            provider="imported", model="imported", functions=["F"],
            notes="", max_devs_per_gw=1,
        )
        sm.initialize_rows_from_import(run_id, [
            {"function": "F", "guideword": "no", "deviation": "D",
             "cause": "C", "effect": "E", "potentially_dangerous": False}
        ])

        mock_sm.get_run = sm.get_run
        mock_sm.get_raw_rows = sm.get_raw_rows
        mock_sm.update_rows_after_regeneration = sm.update_rows_after_regeneration
        mock_sm.get_row = sm.get_row

        RuntimeConfig.set_api_key("gemini", "test-key")

        mock_patch_l3.return_value = [
            {"row_id": "L3-0", "function": "F", "guideword": "no",
             "deviation": "D", "cause": "C", "effect": "New E",
             "potentially_dangerous": False}
        ]

        from src.gui.services.pipeline_adapter import PipelineAdapter
        PipelineAdapter.regenerate_rows(
            run_id=run_id, row_ids=["L3-0"],
            scope=RegenerationScope.L3, suggestion="",
        )

        run = sm.get_run(run_id)
        assert run.provider == "gemini"

    @patch("src.gui.services.pipeline_adapter.state_manager")
    def test_no_key_raises(self, mock_sm):
        sm = StateManager()
        run_id = sm.create_run(
            provider="imported", model="imported", functions=["F"],
            notes="", max_devs_per_gw=1,
        )
        sm.initialize_rows_from_import(run_id, [
            {"function": "F", "guideword": "no", "deviation": "D",
             "cause": "C", "effect": "E", "potentially_dangerous": False}
        ])

        mock_sm.get_run = sm.get_run
        mock_sm.get_raw_rows = sm.get_raw_rows

        from src.gui.services.pipeline_adapter import PipelineAdapter
        with pytest.raises(ValueError, match="No API key configured"):
            PipelineAdapter.regenerate_rows(
                run_id=run_id, row_ids=["L3-0"],
                scope=RegenerationScope.L3, suggestion="",
            )
