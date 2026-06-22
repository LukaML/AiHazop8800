# src/gui/routes/analysis.py
"""API endpoints for starting and monitoring AI-HAZOP-8800 analysis runs.

A run analyses one or more *contexts* (component+aspect units). Supports a context
file upload (.yaml/.json), loading a bundled example, and HTML import of a
previously exported worksheet for review/editing.
"""
import os
import tempfile

from fastapi import APIRouter, HTTPException, UploadFile, File

from ..config import RuntimeConfig
from ..models import (
    StartAnalysisRequest,
    StartAnalysisResponse,
    AnalysisStatusResponse,
    AnalysisResultsResponse,
    AnalysisStatus,
    ContextInput,
    UploadContextResponse,
    LoadExampleResponse,
    ImportHtmlResponse,
)
from ..services.state_manager import state_manager
from ..services.pipeline_adapter import pipeline_adapter
from ..services.html_importer import parse_hazop_html, build_states_from_rows
from .catalogues import _EXAMPLES_DIR

router = APIRouter(prefix="/api/analysis", tags=["analysis"])


def _load_context_file(content_str: str, ext: str):
    """Parse a context file's text via the shared CLI loader. Returns context dicts."""
    from src.run_pipeline import load_contexts
    suffix = ext if ext in (".yaml", ".yml", ".json") else ".yaml"
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=suffix, encoding="utf-8") as tmp:
            tmp.write(content_str)
            tmp_path = tmp.name
        return load_contexts(tmp_path)
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


@router.post("/start", response_model=StartAnalysisResponse)
async def start_analysis(request: StartAnalysisRequest):
    """Start a new analysis. Runs asynchronously — poll /status for completion."""
    provider = request.provider.value
    if not RuntimeConfig.has_api_key(provider):
        raise HTTPException(
            status_code=400,
            detail=f"API key not configured for {provider}. Set it first via /api/settings/api-key",
        )

    from .settings import PROVIDER_DEFAULTS
    model = request.model or PROVIDER_DEFAULTS.get(provider)

    contexts = [c.model_dump() for c in request.contexts]

    run_id = state_manager.create_run(
        provider=provider, model=model or "", contexts=contexts, notes=request.notes,
    )

    pipeline_adapter.run_analysis_async(
        run_id=run_id,
        provider=provider,
        model=model,
        model_review=request.model_review,
        contexts=contexts,
        notes=request.notes,
        guidewords=request.guidewords,
    )

    return StartAnalysisResponse(
        run_id=run_id, status=AnalysisStatus.PENDING, message="Analysis started",
    )


@router.get("/{run_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(run_id: str):
    """Poll until status is COMPLETED or FAILED."""
    run = state_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    return AnalysisStatusResponse(
        run_id=run_id, status=run.status, progress=run.progress, error=run.error,
    )


@router.get("/{run_id}/results", response_model=AnalysisResultsResponse)
async def get_analysis_results(run_id: str):
    """Get the full results of a completed analysis."""
    run = state_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")
    if run.status == AnalysisStatus.PENDING:
        raise HTTPException(status_code=400, detail="Analysis not yet started")
    if run.status == AnalysisStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Analysis still running")
    if run.status == AnalysisStatus.FAILED:
        raise HTTPException(status_code=400, detail=f"Analysis failed: {run.error}")

    rows = state_manager.get_all_rows(run_id)
    return AnalysisResultsResponse(
        run_id=run_id,
        status=run.status,
        provider=run.provider,
        model=run.model,
        components=len(run.contexts),
        rows=[row.to_row_data() for row in rows],
        created_at=run.created_at.isoformat() + "Z",
        completed_at=run.completed_at.isoformat() + "Z" if run.completed_at else None,
    )


@router.post("/upload-context", response_model=UploadContextResponse)
async def upload_context(file: UploadFile = File(...)):
    """Upload a context file (.yaml/.json): a single context or {components: [...]}."""
    try:
        content_str = (await file.read()).decode("utf-8")
        ext = os.path.splitext(file.filename or "context.yaml")[1].lower()
        contexts = _load_context_file(content_str, ext)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return UploadContextResponse(
        success=True,
        contexts=[ContextInput(**c) for c in contexts],
        message=f"Parsed {len(contexts)} context(s)",
    )


@router.get("/example/{name}", response_model=LoadExampleResponse)
async def load_example(name: str):
    """Load a bundled example context by file stem (see /api/catalogues)."""
    base = (name or "").strip().replace("/", "").replace("\\", "")
    path = None
    for ext in (".yaml", ".yml", ".json"):
        cand = _EXAMPLES_DIR / f"{base}{ext}"
        if cand.exists():
            path = cand
            break
    if path is None:
        raise HTTPException(status_code=404, detail=f"Example not found: {name}")

    try:
        contexts = _load_context_file(path.read_text(encoding="utf-8"), path.suffix.lower())
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return LoadExampleResponse(
        success=True,
        contexts=[ContextInput(**c) for c in contexts],
        message=f"Loaded example '{base}'",
    )


@router.post("/import-html", response_model=ImportHtmlResponse)
async def import_html(file: UploadFile = File(...)):
    """Import a previously exported worksheet (CLI or GUI HTML) for editing."""
    try:
        content_str = (await file.read()).decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    try:
        rows = parse_hazop_html(content_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    contexts, states = build_states_from_rows(rows)

    run_id = state_manager.create_run(
        provider="imported", model="imported", contexts=contexts, notes="Imported from HTML",
    )
    state_manager.set_states(run_id, states)
    run = state_manager.get_run(run_id)
    if run:
        run.status = AnalysisStatus.COMPLETED
        from datetime import datetime
        run.completed_at = datetime.utcnow()

    return ImportHtmlResponse(
        success=True, run_id=run_id, row_count=len(rows),
        message=f"Imported {len(rows)} rows from HTML",
    )
