# src/gui/routes/rows.py
"""API endpoints for row editing, rating, and regeneration.

Edit/rate update in-memory state directly (edits also flow into the authoritative
component state). Regeneration patches the chosen phase (L1–L8) and cascades
downstream via the pipeline adapter.
"""
from fastapi import APIRouter, HTTPException

from ..models import (
    EditRowRequest,
    EditRowResponse,
    RateRowRequest,
    RateRowResponse,
    RegenerateRowsRequest,
    RegenerateRowsResponse,
)
from ..services.state_manager import state_manager
from ..services.pipeline_adapter import pipeline_adapter

router = APIRouter(prefix="/api/rows", tags=["rows"])


@router.put("/{run_id}/{row_id}/edit", response_model=EditRowResponse)
async def edit_row(run_id: str, row_id: str, request: EditRowRequest):
    """Edit a row's editable L1–L8 fields. Human edits override LLM values."""
    if not state_manager.get_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    row = state_manager.update_row_fields(run_id, row_id, request.fields)
    if not row:
        raise HTTPException(status_code=404, detail=f"Row not found: {row_id}")

    return EditRowResponse(success=True, row=row.to_row_data(), message="Row updated")


@router.put("/{run_id}/{row_id}/rate", response_model=RateRowResponse)
async def rate_row(run_id: str, row_id: str, request: RateRowRequest):
    """Set human rating for a row."""
    if not state_manager.get_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    row = state_manager.update_row_rating(run_id, row_id, request.rating)
    if not row:
        raise HTTPException(status_code=404, detail=f"Row not found: {row_id}")

    return RateRowResponse(
        success=True, row_id=row_id, rating=request.rating, message="Rating saved",
    )


@router.post("/{run_id}/regenerate", response_model=RegenerateRowsResponse)
async def regenerate_rows(run_id: str, request: RegenerateRowsRequest):
    """Regenerate rows at a phase scope (L1–L8 / ALL) and cascade downstream."""
    if not state_manager.get_run(run_id):
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    for row_id in request.row_ids:
        if not state_manager.get_row(run_id, row_id):
            raise HTTPException(status_code=404, detail=f"Row not found: {row_id}")

    try:
        pipeline_adapter.regenerate_rows(
            run_id=run_id,
            row_ids=request.row_ids,
            scope=request.scope,
            suggestion=request.suggestion,
        )

        updated_rows = []
        for row_id in request.row_ids:
            row = state_manager.get_row(run_id, row_id)
            if row:
                updated_rows.append(row.to_row_data())

        return RegenerateRowsResponse(
            success=True,
            regenerated_row_ids=request.row_ids,
            rows=updated_rows,
            message=f"Regenerated {len(request.row_ids)} rows",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
