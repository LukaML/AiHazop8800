# src/gui/services/pipeline_adapter.py
"""Adapter bridging the GUI to the AI-HAZOP-8800 LangGraph pipeline.

Handles API-key injection, async pipeline execution (thread pool), and per-row
regeneration. A run analyses one or more *contexts* (component+aspect units); the
full LangGraph output state for each is kept so regeneration can patch a phase and
cascade downstream via ``graph_full.patch_and_cascade`` — the same helper the
holistic reviewer uses. Row matching is suffix-based; canonical GUI ids are
``c{component_index}__{within-context row_id}``.
"""
import os
import logging
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading

from ..config import RuntimeConfig
from ..models import AnalysisStatus, RegenerationScope
from .state_manager import state_manager

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=2)
_run_locks: Dict[str, threading.Lock] = {}
_run_locks_guard = threading.Lock()

_PHASE_KEYS = [f"rows_l{i}" for i in range(1, 9)]


def _get_run_lock(run_id: str) -> threading.Lock:
    with _run_locks_guard:
        if run_id not in _run_locks:
            _run_locks[run_id] = threading.Lock()
        return _run_locks[run_id]


def _merge_or_append(full: List[Dict[str, Any]], subset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Replace rows in ``full`` by suffix from ``subset``; append suffixes not present."""
    from src.row_utils import _merge_subset_by_suffix, _suffix
    merged = _merge_subset_by_suffix(full, subset)
    have = {_suffix(str(r.get("row_id"))) for r in merged}
    for r in subset:
        if _suffix(str(r.get("row_id"))) not in have:
            merged.append(deepcopy(r))
    return merged


class PipelineAdapter:
    """Adapter for the AI-HAZOP-8800 pipeline."""

    @staticmethod
    def _inject_api_key(provider: str) -> None:
        env_var = RuntimeConfig.get_env_var_name(provider)
        if env_var:
            api_key = RuntimeConfig.get_api_key(provider)
            if api_key:
                os.environ[env_var] = api_key

    @staticmethod
    def _configure_llm(provider: str, model: Optional[str], model_review: Optional[str]) -> None:
        from src.llm_client import configure as configure_llm
        configure_llm(provider=provider, model=model, model_review=model_review)

    @staticmethod
    def run_analysis_sync(
        run_id: str,
        provider: str,
        model: Optional[str],
        model_review: Optional[str],
        contexts: List[Dict[str, str]],
        notes: str,
        guidewords: Optional[List[str]] = None,
    ) -> None:
        """Run the full L1–L8 pipeline for every context, then build row state."""
        lock = _get_run_lock(run_id)
        with lock:
            try:
                state_manager.update_status(run_id, AnalysisStatus.RUNNING, "Configuring LLM...")
                PipelineAdapter._inject_api_key(provider)
                PipelineAdapter._configure_llm(provider, model, model_review)

                from src.graph_full import build_full_graph
                from src.run_pipeline import _build_state

                graph = build_full_graph().compile()
                states: List[Dict[str, Any]] = []
                total = len(contexts)
                for ci, ctx in enumerate(contexts):
                    state_manager.update_status(
                        run_id, AnalysisStatus.RUNNING,
                        f"Analysing component {ci + 1}/{total}: {ctx.get('component', '')}",
                    )
                    state_in = _build_state(ctx, notes, guidewords)
                    states.append(graph.invoke(state_in))

                state_manager.set_states(run_id, states)
                state_manager.update_status(run_id, AnalysisStatus.COMPLETED)
                run = state_manager.get_run(run_id)
                logger.info("Analysis %s completed with %d rows", run_id, len(run.rows) if run else 0)

            except Exception as e:
                logger.exception("Analysis %s failed: %s", run_id, e)
                state_manager.update_status(run_id, AnalysisStatus.FAILED, error=str(e))

    @staticmethod
    def run_analysis_async(
        run_id: str,
        provider: str,
        model: Optional[str],
        model_review: Optional[str],
        contexts: List[Dict[str, str]],
        notes: str,
        guidewords: Optional[List[str]] = None,
    ) -> None:
        _executor.submit(
            PipelineAdapter.run_analysis_sync,
            run_id, provider, model, model_review, contexts, notes, guidewords,
        )

    @staticmethod
    def regenerate_rows(
        run_id: str,
        row_ids: List[str],
        scope: RegenerationScope,
        suggestion: str,
    ) -> List[str]:
        """Patch the targeted rows at ``scope`` and cascade downstream per component."""
        run = state_manager.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        lock = _get_run_lock(run_id)
        with lock:
            provider = run.provider
            model = run.model

            # Imported runs have no real provider — resolve from configured keys.
            if provider == "imported":
                from ..routes.settings import PROVIDER_DEFAULTS
                configured = [p for p, has_key in RuntimeConfig.get_key_status().items() if has_key]
                if not configured:
                    raise ValueError(
                        "No API key configured. Set an API key for a provider before regenerating."
                    )
                provider = configured[0]
                model = PROVIDER_DEFAULTS.get(provider)
                run.provider = provider
                run.model = model

            PipelineAdapter._inject_api_key(provider)
            PipelineAdapter._configure_llm(provider, model, None)

            from src.graph_full import patch_and_cascade
            from src.row_utils import _suffix, _pick_rows_by_suffix

            scope_str = scope.value if hasattr(scope, "value") else str(scope)
            if scope_str == "ALL":
                scope_str = "L1"
            notes = run.notes

            # Group targets by component.
            by_comp: Dict[int, List[str]] = {}
            for rid in row_ids:
                row = run.rows.get(rid)
                if row is not None:
                    by_comp.setdefault(row.component_index, []).append(rid)

            for ci, rids in by_comp.items():
                if ci >= len(run.states):
                    continue
                state = run.states[ci]
                target_sfx = sorted({_suffix(rid) for rid in rids})

                # Surgical sub-state: only the targeted rows at each phase, so the
                # cascade regenerates downstream for just those rows (preserving
                # other rows' edits in the full state).
                sub = dict(state)
                for key in _PHASE_KEYS:
                    sub[key] = _pick_rows_by_suffix(state.get(key) or [], target_sfx)

                updates = patch_and_cascade(sub, scope_str, target_sfx, suggestion, notes=notes)

                for key, rows in updates.items():
                    state[key] = _merge_or_append(state.get(key) or [], rows)

            state_manager.refresh_rows_from_state(run_id, row_ids)
            return row_ids


# Global adapter instance
pipeline_adapter = PipelineAdapter()
