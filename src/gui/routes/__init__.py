# src/gui/routes/__init__.py
"""
API routes package.
"""
from .settings import router as settings_router
from .analysis import router as analysis_router
from .rows import router as rows_router
from .export import router as export_router

__all__ = [
    "settings_router",
    "analysis_router",
    "rows_router",
    "export_router",
]
