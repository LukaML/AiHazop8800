# src/gui/routes/analysis.py
"""API endpoints for starting and monitoring HAZOP analysis runs.

Includes file upload endpoints for functions (txt/yaml/json) and RAG
documents (txt/csv/xlsx/pdf), plus HTML import for resuming previous runs.
"""
import os
import tempfile
import json
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File

from ..config import RuntimeConfig
from ..models import (
    StartAnalysisRequest,
    StartAnalysisResponse,
    AnalysisStatusResponse,
    AnalysisResultsResponse,
    AnalysisStatus,
    UploadFunctionsResponse,
    UploadRagResponse,
    ImportHtmlResponse,
)
from ..services.state_manager import state_manager
from ..services.pipeline_adapter import pipeline_adapter
from ..services.html_importer import parse_hazop_html

router = APIRouter(prefix="/api/analysis", tags=["analysis"])

# Temporary storage for uploaded files
_uploaded_rag_files: List[str] = []


@router.post("/start", response_model=StartAnalysisResponse)
async def start_analysis(request: StartAnalysisRequest):
    """
    Start a new HAZOP analysis.
    Runs asynchronously - use /status endpoint to poll for completion.
    """
    # Verify API key is configured
    provider = request.provider.value
    if not RuntimeConfig.has_api_key(provider):
        raise HTTPException(
            status_code=400,
            detail=f"API key not configured for {provider}. Set it first via /api/settings/api-key",
        )

    # Determine model
    from .settings import PROVIDER_DEFAULTS
    model = request.model or PROVIDER_DEFAULTS.get(provider)

    # Get RAG paths
    rag_paths = list(request.rag_paths) if request.rag_enabled else []
    if request.rag_enabled and _uploaded_rag_files:
        rag_paths.extend(_uploaded_rag_files)

    # Create run
    run_id = state_manager.create_run(
        provider=provider,
        model=model or "",
        functions=request.functions,
        notes=request.notes,
        max_devs_per_gw=request.max_devs_per_gw,
        rag_enabled=request.rag_enabled,
        rag_embedder=request.rag_embedder.value if request.rag_enabled else None,
        rag_paths=rag_paths,
    )

    if request.rag_enabled and not rag_paths:
        raise HTTPException(
            status_code=400,
            detail="RAG is enabled but no documents were uploaded. Please upload RAG files first.",
        )

    # Start async analysis
    pipeline_adapter.run_analysis_async(
        run_id=run_id,
        provider=provider,
        model=model,
        model_review=request.model_review,
        functions=request.functions,
        notes=request.notes,
        max_devs_per_gw=request.max_devs_per_gw,
        rag_enabled=request.rag_enabled,
        rag_paths=rag_paths,
        rag_embedder=request.rag_embedder.value,
        rag_min_sim=request.rag_min_sim,
    )

    return StartAnalysisResponse(
        run_id=run_id,
        status=AnalysisStatus.PENDING,
        message="Analysis started",
    )


@router.get("/{run_id}/status", response_model=AnalysisStatusResponse)
async def get_analysis_status(run_id: str):
    """
    Get the status of an analysis run.
    Poll this endpoint until status is COMPLETED or FAILED.
    """
    run = state_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    return AnalysisStatusResponse(
        run_id=run_id,
        status=run.status,
        progress=run.progress,
        error=run.error,
    )


@router.get("/{run_id}/results", response_model=AnalysisResultsResponse)
async def get_analysis_results(run_id: str):
    """
    Get the full results of a completed analysis.
    """
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
    row_data = [row.to_row_data() for row in rows]

    return AnalysisResultsResponse(
        run_id=run_id,
        status=run.status,
        provider=run.provider,
        model=run.model,
        rag_enabled=run.rag_enabled,
        rag_embedder=run.rag_embedder,
        rows=row_data,
        created_at=run.created_at.isoformat() + "Z",
        completed_at=run.completed_at.isoformat() + "Z" if run.completed_at else None,
    )


@router.post("/upload-functions", response_model=UploadFunctionsResponse)
async def upload_functions(file: UploadFile = File(...)):
    """
    Upload a functions file (txt, yaml, json).
    Returns parsed list of functions.
    """
    try:
        content = await file.read()
        content_str = content.decode("utf-8")

        filename = file.filename or "functions.txt"
        ext = os.path.splitext(filename)[1].lower()

        functions = []

        if ext in (".txt", ".list", ""):
            functions = [ln.strip() for ln in content_str.splitlines() if ln.strip()]

        elif ext in (".yaml", ".yml"):
            try:
                import yaml
                obj = yaml.safe_load(content_str)
                if isinstance(obj, dict) and "functions" in obj:
                    functions = [str(f) for f in obj["functions"]]
                elif isinstance(obj, list):
                    functions = [str(f) for f in obj]
                else:
                    raise ValueError("YAML must contain 'functions' key or be a list")
            except ImportError:
                raise HTTPException(status_code=400, detail="YAML support not available")

        elif ext == ".json":
            obj = json.loads(content_str)
            if isinstance(obj, dict) and "functions" in obj:
                functions = [str(f) for f in obj["functions"]]
            elif isinstance(obj, list):
                functions = [str(f) for f in obj]
            else:
                raise ValueError("JSON must contain 'functions' key or be an array")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

        if not functions:
            raise HTTPException(status_code=400, detail="No functions found in file")

        return UploadFunctionsResponse(
            success=True,
            functions=functions,
            message=f"Parsed {len(functions)} functions",
        )

    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/upload-rag", response_model=UploadRagResponse)
async def upload_rag(files: List[UploadFile] = File(...)):
    """
    Upload RAG documents (txt, csv, xlsx, pdf).
    Files are stored temporarily for the analysis.
    """
    global _uploaded_rag_files

    saved_paths = []

    try:
        for file in files:
            # Create temp file
            suffix = os.path.splitext(file.filename or "doc.txt")[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                content = await file.read()
                tmp.write(content)
                saved_paths.append(tmp.name)

        _uploaded_rag_files = saved_paths

        return UploadRagResponse(
            success=True,
            file_paths=saved_paths,
            message=f"Uploaded {len(saved_paths)} RAG documents",
        )

    except Exception as e:
        # Clean up on error
        for path in saved_paths:
            try:
                os.unlink(path)
            except Exception:
                pass
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/import-html", response_model=ImportHtmlResponse)
async def import_html(file: UploadFile = File(...)):
    """Import a previously exported HAZOP HTML file into the GUI for editing."""
    try:
        content = await file.read()
        content_str = content.decode("utf-8")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {e}")

    try:
        rows = parse_hazop_html(content_str)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Extract unique functions from parsed rows
    functions = list(dict.fromkeys(r["function"] for r in rows))

    # Create run in COMPLETED state
    run_id = state_manager.create_run(
        provider="imported",
        model="imported",
        functions=functions,
        notes="Imported from HTML",
        max_devs_per_gw=1,
    )
    state_manager.initialize_rows_from_import(run_id, rows)

    return ImportHtmlResponse(
        success=True,
        run_id=run_id,
        row_count=len(rows),
        message=f"Imported {len(rows)} rows from HTML",
    )
