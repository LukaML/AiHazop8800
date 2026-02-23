# src/gui/services/pipeline_adapter.py
"""Adapter bridging GUI row edits/regeneration to the LLM pipeline.

Handles API key injection, async pipeline execution (via thread pool),
and per-scope regeneration with cascade (L1→L2→L3, L2→L3, or L3 only).
Row ID mapping uses suffix-based lookup to handle prefix mismatches between
the GUI's canonical L3 IDs and the raw pipeline stage prefixes.
"""
import os
import logging
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import threading

from ..config import RuntimeConfig
from ..models import AnalysisStatus, RegenerationScope
from .state_manager import state_manager
from src.row_utils import _merge_rag_notes

logger = logging.getLogger(__name__)

# Thread pool for async pipeline execution
_executor = ThreadPoolExecutor(max_workers=2)
_run_locks: Dict[str, threading.Lock] = {}
_run_locks_guard = threading.Lock()


def _get_run_lock(run_id: str) -> threading.Lock:
    """Get or create a lock for a specific run."""
    with _run_locks_guard:
        if run_id not in _run_locks:
            _run_locks[run_id] = threading.Lock()
        return _run_locks[run_id]


def _regen_notes(run, stage: str) -> str:
    """Merge user notes + stored RAG context for a given stage."""
    rag_key = {"L1": "_rag_notes_l1", "L2": "_rag_notes_l2", "L3": "_rag_notes_l3"}
    rag = getattr(run, rag_key.get(stage, "_rag_notes_l3"), "")
    return _merge_rag_notes(run.notes, rag)


class PipelineAdapter:
    """Adapter for the HAZOP LLM pipeline."""

    @staticmethod
    def _inject_api_key(provider: str) -> None:
        """Temporarily inject API key into environment for pipeline."""
        env_var = RuntimeConfig.get_env_var_name(provider)
        if env_var:
            api_key = RuntimeConfig.get_api_key(provider)
            if api_key:
                os.environ[env_var] = api_key

    @staticmethod
    def _configure_llm(provider: str, model: Optional[str], model_review: Optional[str]) -> None:
        """Configure the LLM client."""
        from src.llm_client import configure as configure_llm
        configure_llm(
            provider=provider,
            model=model,
            model_review=model_review,
        )

    @staticmethod
    def run_analysis_sync(
        run_id: str,
        provider: str,
        model: Optional[str],
        model_review: Optional[str],
        functions: List[str],
        notes: str,
        max_devs_per_gw: int,
        rag_enabled: bool,
        rag_paths: List[str],
        rag_embedder: str,
        rag_min_sim: Optional[float] = None,
    ) -> None:
        """
        Run the HAZOP analysis synchronously.
        Updates state manager with progress and results.
        """
        lock = _get_run_lock(run_id)

        with lock:
            try:
                # Update status to running
                state_manager.update_status(run_id, AnalysisStatus.RUNNING, "Configuring LLM...")

                # Inject API key
                PipelineAdapter._inject_api_key(provider)

                # Configure LLM
                PipelineAdapter._configure_llm(provider, model, model_review)

                state_manager.update_status(run_id, AnalysisStatus.RUNNING, "Running pipeline...")

                # Import and run the pipeline
                from src.graph_full import build_full_graph, HazopGraphState
                from src.models import Guideword

                guideword_map = {g.value: g.value for g in Guideword}

                state: HazopGraphState = {
                    "functions": functions,
                    "guideword_map": guideword_map,
                    "max_devs_per_gw": max_devs_per_gw,
                    "notes": notes,
                    "rag_enabled": rag_enabled,
                    "rag_paths": rag_paths,
                    "rag_embedder": rag_embedder,
                    "rag_min_sim": rag_min_sim if rag_min_sim is not None else (0.1 if rag_embedder == "local" else 0.25),
                }

                graph = build_full_graph().compile()
                result = graph.invoke(state)

                # Extract rows from result
                rows_l1 = result.get("rows_l1", [])
                rows_l2 = result.get("rows_l2", [])
                rows_l3 = result.get("rows_l3", [])

                # Persist computed RAG notes for later regeneration
                run = state_manager.get_run(run_id)
                if run:
                    run._rag_notes_l1 = result.get("rag_notes_l1", "")
                    run._rag_notes_l2 = result.get("rag_notes_l2", "")
                    run._rag_notes_l3 = result.get("rag_notes_l3", "")

                # Initialize row states
                state_manager.initialize_rows_from_pipeline(run_id, rows_l1, rows_l2, rows_l3)

                # Update status to completed
                state_manager.update_status(run_id, AnalysisStatus.COMPLETED)

                logger.info("Analysis %s completed with %d rows", run_id, len(rows_l3))

            except Exception as e:
                logger.exception("Analysis %s failed: %s", run_id, e)
                state_manager.update_status(
                    run_id,
                    AnalysisStatus.FAILED,
                    error=str(e),
                )

    @staticmethod
    def run_analysis_async(
        run_id: str,
        provider: str,
        model: Optional[str],
        model_review: Optional[str],
        functions: List[str],
        notes: str,
        max_devs_per_gw: int,
        rag_enabled: bool,
        rag_paths: List[str],
        rag_embedder: str,
        rag_min_sim: Optional[float] = None,
    ) -> None:
        """Submit analysis to thread pool for async execution."""
        _executor.submit(
            PipelineAdapter.run_analysis_sync,
            run_id,
            provider,
            model,
            model_review,
            functions,
            notes,
            max_devs_per_gw,
            rag_enabled,
            rag_paths,
            rag_embedder,
            rag_min_sim,
        )

    @staticmethod
    def regenerate_rows(
        run_id: str,
        row_ids: List[str],
        scope: RegenerationScope,
        suggestion: str,
    ) -> List[Dict[str, Any]]:
        """
        Regenerate specific rows using the patch functions.
        Applies cascade based on scope (L1→L2→L3, L2→L3, or L3 only).
        """
        run = state_manager.get_run(run_id)
        if not run:
            raise ValueError(f"Run not found: {run_id}")

        lock = _get_run_lock(run_id)

        with lock:
            try:
                provider = run.provider
                model = run.model

                # Imported runs have no real provider — resolve from configured keys
                if provider == "imported":
                    from ..config import RuntimeConfig
                    from ..routes.settings import PROVIDER_DEFAULTS
                    key_status = RuntimeConfig.get_key_status()
                    configured = [p for p, has_key in key_status.items() if has_key]
                    if not configured:
                        raise ValueError(
                            "No API key configured. Set an API key for a provider before regenerating."
                        )
                    provider = configured[0]
                    model = PROVIDER_DEFAULTS.get(provider)
                    # Update the run so future regenerations + info bar reflect the real provider
                    run.provider = provider
                    run.model = model

                # Inject API key for regeneration
                PipelineAdapter._inject_api_key(provider)
                PipelineAdapter._configure_llm(provider, model, None)

                # Get raw rows
                raw_l1, raw_l2, raw_l3 = state_manager.get_raw_rows(run_id)

                # Import patch functions
                from src.chains import reviewer_patch_l1, reviewer_patch_l2, reviewer_patch_l3
                from src.chains import l2_init_from_l1, l3_init_from_l2
                from src.row_utils import (
                    _ensure_rows_list,
                    _ensure_row_ids,
                    _merge_subset_by_suffix,
                    _pick_rows_by_suffix,
                    _suffix,
                )

                # Extract suffixes from row_ids
                target_suffixes = [_suffix(rid) for rid in row_ids]

                regenerated = []

                # --- Regeneration dispatch by scope ---
                # L1/ALL: patch deviation → cascade to L2 → cascade to L3
                # L2: patch cause → cascade to L3
                # L3: patch effect only (no cascade)
                if scope == RegenerationScope.L1 or scope == RegenerationScope.ALL:
                    # Patch L1
                    l1_subset = _pick_rows_by_suffix(raw_l1, target_suffixes)
                    row_hints = [
                        {"row_id": rid, "fields": ["deviation"], "suggestion": suggestion}
                        for rid in row_ids
                    ]
                    patched_l1 = reviewer_patch_l1(l1_subset, row_hints, source="human", notes=_regen_notes(run, "L1")) or []
                    patched_l1 = _ensure_rows_list(patched_l1)
                    _ensure_row_ids("L1", patched_l1)
                    raw_l1 = _merge_subset_by_suffix(raw_l1, patched_l1)

                    # Cascade to L2
                    l2_new = l2_init_from_l1(raw_l1, notes=_regen_notes(run, "L2")) or []
                    l2_new = _ensure_rows_list(l2_new)
                    _ensure_row_ids("L2", l2_new)
                    l2_subset = _pick_rows_by_suffix(l2_new, target_suffixes)
                    raw_l2 = _merge_subset_by_suffix(raw_l2, l2_subset)

                    # Cascade to L3
                    l3_new = l3_init_from_l2(raw_l2, notes=_regen_notes(run, "L3")) or []
                    l3_new = _ensure_rows_list(l3_new)
                    _ensure_row_ids("L3", l3_new)
                    l3_subset = _pick_rows_by_suffix(l3_new, target_suffixes)
                    raw_l3 = _merge_subset_by_suffix(raw_l3, l3_subset)

                    regenerated = l3_subset

                elif scope == RegenerationScope.L2:
                    # Check for missing L2 rows and generate from L1 if needed
                    l2_subset = _pick_rows_by_suffix(raw_l2, target_suffixes)
                    l2_found_suffixes = {_suffix(r.get("row_id", "")) for r in l2_subset}
                    missing_l2_suffixes = [s for s in target_suffixes if s not in l2_found_suffixes]

                    if missing_l2_suffixes:
                        logger.warning(
                            "L2 regeneration: %d rows missing from raw_l2, generating from L1: %s",
                            len(missing_l2_suffixes), missing_l2_suffixes,
                        )
                        l1_for_missing = _pick_rows_by_suffix(raw_l1, missing_l2_suffixes)
                        if l1_for_missing:
                            generated_l2 = l2_init_from_l1(l1_for_missing, notes=_regen_notes(run, "L2")) or []
                            generated_l2 = _ensure_rows_list(generated_l2)
                            _ensure_row_ids("L2", generated_l2)
                            raw_l2 = raw_l2 + generated_l2
                            # Refresh subset with newly generated rows
                            l2_subset = _pick_rows_by_suffix(raw_l2, target_suffixes)

                    # Patch L2 — use suffix-based lookup to build hint row_ids
                    # from actual subset rows (handles prefix mismatches)
                    ids_by_suffix = {_suffix(r.get("row_id", "")): r.get("row_id", "") for r in l2_subset}
                    row_hints = [
                        {"row_id": ids_by_suffix.get(_suffix(rid), rid), "fields": ["cause"], "suggestion": suggestion}
                        for rid in row_ids
                    ]
                    patched_l2 = reviewer_patch_l2(l2_subset, row_hints, source="human", notes=_regen_notes(run, "L2")) or []
                    patched_l2 = _ensure_rows_list(patched_l2)
                    _ensure_row_ids("L2", patched_l2)
                    raw_l2 = _merge_subset_by_suffix(raw_l2, patched_l2)

                    # Cascade to L3
                    l3_new = l3_init_from_l2(raw_l2, notes=_regen_notes(run, "L3")) or []
                    l3_new = _ensure_rows_list(l3_new)
                    _ensure_row_ids("L3", l3_new)
                    l3_subset = _pick_rows_by_suffix(l3_new, target_suffixes)
                    raw_l3 = _merge_subset_by_suffix(raw_l3, l3_subset)

                    regenerated = l3_subset

                elif scope == RegenerationScope.L3:
                    # Check for missing L3 rows and generate from L2 (or L1→L2→L3) if needed
                    l3_subset = _pick_rows_by_suffix(raw_l3, target_suffixes)
                    l3_found_suffixes = {_suffix(r.get("row_id", "")) for r in l3_subset}
                    missing_l3_suffixes = [s for s in target_suffixes if s not in l3_found_suffixes]

                    if missing_l3_suffixes:
                        logger.warning(
                            "L3 regeneration: %d rows missing from raw_l3, attempting generation from L2: %s",
                            len(missing_l3_suffixes), missing_l3_suffixes,
                        )
                        # Check if the missing rows also lack L2 data
                        l2_for_missing = _pick_rows_by_suffix(raw_l2, missing_l3_suffixes)
                        l2_found_for_missing = {_suffix(r.get("row_id", "")) for r in l2_for_missing}
                        missing_l2_suffixes = [s for s in missing_l3_suffixes if s not in l2_found_for_missing]

                        if missing_l2_suffixes:
                            logger.warning(
                                "L3 regeneration: %d rows also missing from raw_l2, generating L2 from L1: %s",
                                len(missing_l2_suffixes), missing_l2_suffixes,
                            )
                            l1_for_missing = _pick_rows_by_suffix(raw_l1, missing_l2_suffixes)
                            if l1_for_missing:
                                generated_l2 = l2_init_from_l1(l1_for_missing, notes=_regen_notes(run, "L2")) or []
                                generated_l2 = _ensure_rows_list(generated_l2)
                                _ensure_row_ids("L2", generated_l2)
                                raw_l2 = raw_l2 + generated_l2
                                l2_for_missing = _pick_rows_by_suffix(raw_l2, missing_l3_suffixes)

                        # Generate L3 from L2 for missing rows
                        if l2_for_missing:
                            generated_l3 = l3_init_from_l2(l2_for_missing, notes=_regen_notes(run, "L3")) or []
                            generated_l3 = _ensure_rows_list(generated_l3)
                            _ensure_row_ids("L3", generated_l3)
                            raw_l3 = raw_l3 + generated_l3
                            # Refresh subset with newly generated rows
                            l3_subset = _pick_rows_by_suffix(raw_l3, target_suffixes)

                    # Patch L3 only
                    row_hints = [
                        {"row_id": rid, "fields": ["effect", "potentially_dangerous"], "suggestion": suggestion}
                        for rid in row_ids
                    ]
                    patched_l3 = reviewer_patch_l3(l3_subset, row_hints, source="human", notes=_regen_notes(run, "L3")) or []
                    patched_l3 = _ensure_rows_list(patched_l3)
                    _ensure_row_ids("L3", patched_l3)
                    raw_l3 = _merge_subset_by_suffix(raw_l3, patched_l3)

                    regenerated = patched_l3

                # Update raw rows in state
                run._raw_l1 = raw_l1
                run._raw_l2 = raw_l2
                run._raw_l3 = raw_l3

                # Merge regenerated data into RowState
                # Build merged row data for the updated rows
                merged_rows = []
                l1_by_suffix = {_suffix(r.get("row_id", "")): r for r in raw_l1}
                l2_by_suffix = {_suffix(r.get("row_id", "")): r for r in raw_l2}
                l3_by_suffix = {_suffix(r.get("row_id", "")): r for r in raw_l3}

                for rid in row_ids:
                    sfx = _suffix(rid)
                    r1 = l1_by_suffix.get(sfx, {})
                    r2 = l2_by_suffix.get(sfx, {})
                    r3 = l3_by_suffix.get(sfx, {})

                    merged_rows.append({
                        "row_id": rid,
                        "deviation": r1.get("deviation", ""),
                        "cause": r2.get("cause") or r2.get("causes", ""),
                        "effect": r3.get("effect") or r3.get("effects", ""),
                        "potentially_dangerous": r3.get("potentially_dangerous", False),
                    })

                state_manager.update_rows_after_regeneration(run_id, merged_rows, row_ids)

                return merged_rows

            except Exception as e:
                logger.exception("Regeneration failed for run %s: %s", run_id, e)
                raise


# Global adapter instance
pipeline_adapter = PipelineAdapter()
