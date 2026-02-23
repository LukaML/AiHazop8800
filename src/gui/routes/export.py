# src/gui/routes/export.py
"""API endpoints for CSV and HTML export of analysis results."""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse, Response
import io

from ..services.state_manager import state_manager
from ..services.csv_exporter import export_to_csv
from ..services.html_exporter import export_to_html

router = APIRouter(prefix="/api/export", tags=["export"])


@router.get("/{run_id}/csv")
async def export_csv(run_id: str):
    """
    Export analysis results as research-grade CSV.
    Includes original/final values and all tracking metadata.
    """
    run = state_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    try:
        csv_content = export_to_csv(run_id)

        # Return as downloadable file
        return StreamingResponse(
            io.BytesIO(csv_content.encode("utf-8")),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=hazop_{run_id}.csv"
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{run_id}/html")
async def export_html_endpoint(run_id: str):
    """
    Export analysis results as HTML file.
    Shows only final values plus rating column.
    """
    run = state_manager.get_run(run_id)
    if not run:
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    try:
        html_content = export_to_html(run_id)

        return Response(
            content=html_content,
            media_type="text/html",
            headers={
                "Content-Disposition": f'attachment; filename="hazop_{run_id}.html"'
            },
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
