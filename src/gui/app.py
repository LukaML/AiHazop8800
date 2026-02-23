# src/gui/app.py
"""FastAPI web application for interactive HAZOP analysis.

Serves a single-page frontend (static/index.html) and exposes REST API
endpoints for settings, analysis runs, row editing, and export.
Run with: python -m src.gui.app
"""
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from .routes import settings_router, analysis_router, rows_router, export_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="HAZOP LLM GUI",
    description="Human-in-the-loop HAZOP analysis interface",
    version="0.1.0",
)

# Include API routers
app.include_router(settings_router)
app.include_router(analysis_router)
app.include_router(rows_router)
app.include_router(export_router)

# Static files directory
STATIC_DIR = Path(__file__).parent / "static"


# Mount static files if directory exists
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "HAZOP GUI API", "docs": "/docs"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}


def main():
    """Run the server."""
    import uvicorn

    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))

    logger.info("Starting HAZOP GUI server at http://%s:%d", host, port)
    logger.info("API docs available at http://%s:%d/docs", host, port)

    uvicorn.run(
        "src.gui.app:app",
        host=host,
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
