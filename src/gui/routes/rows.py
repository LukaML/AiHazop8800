# src/gui/routes/rows.py
"""API endpoints for row editing, rating, and regeneration.

Edit/rate operations update in-memory state directly.  Regeneration invokes
the LLM pipeline adapter with cascade based on scope (L1→L2→L3, L2→L3, L3).
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
    """
    Edit a row's fields (deviation, cause, effect, potentially_dangerous).
    Human edits override original values.
    """
    run = state_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    row = state_manager.update_row_fields(
        run_id=run_id,
        row_id=row_id,
        deviation=request.deviation,
        cause=request.cause,
        effect=request.effect,
        potentially_dangerous=request.potentially_dangerous,
    )

    if not row:
        raise HTTPException(status_code=404, detail=f"Row not found: {row_id}")

    return EditRowResponse(
        success=True,
        row=row.to_row_data(),
        message="Row updated",
    )


@router.put("/{run_id}/{row_id}/rate", response_model=RateRowResponse)
async def rate_row(run_id: str, row_id: str, request: RateRowRequest):
    """
    Set human rating for a row.
    """
    run = state_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    row = state_manager.update_row_rating(
        run_id=run_id,
        row_id=row_id,
        rating=request.rating,
    )

    if not row:
        raise HTTPException(status_code=404, detail=f"Row not found: {row_id}")

    return RateRowResponse(
        success=True,
        row_id=row_id,
        rating=request.rating,
        message="Rating saved",
    )


@router.post("/{run_id}/regenerate", response_model=RegenerateRowsResponse)
async def regenerate_rows(run_id: str, request: RegenerateRowsRequest):
    """
    Regenerate specific rows using LLM.
    Applies cascade based on scope:
    - L1: Regenerate deviation → cascade to L2 → L3
    - L2: Regenerate cause → cascade to L3
    - L3: Regenerate effect only
    - ALL: Full cascade from L1
    """
    run = state_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    # Verify all row_ids exist
    for row_id in request.row_ids:
        if not state_manager.get_row(run_id, row_id):
            raise HTTPException(status_code=404, detail=f"Row not found: {row_id}")

    try:
        regenerated = pipeline_adapter.regenerate_rows(
            run_id=run_id,
            row_ids=request.row_ids,
            scope=request.scope,
            suggestion=request.suggestion,
        )

        # Get updated rows
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
