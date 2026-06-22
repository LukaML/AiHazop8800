# src/gui/routes/catalogues.py
"""Read-only catalogue data that drives the setup form.

Exposes the taxonomy component classes + aspects, the working guideword set, and
the names of bundled example contexts, all sourced from src/catalogues and
src/examples via catalogue_loader.
"""
from pathlib import Path

from fastapi import APIRouter

from ..models import CataloguesResponse
from src.catalogue_loader import get_component_classes, get_aspects, get_guidewords

router = APIRouter(prefix="/api/catalogues", tags=["catalogues"])

_EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "examples"


def _example_names() -> list:
    if not _EXAMPLES_DIR.exists():
        return []
    return sorted(
        p.stem for p in _EXAMPLES_DIR.iterdir()
        if p.suffix.lower() in (".yaml", ".yml", ".json")
    )


@router.get("", response_model=CataloguesResponse)
@router.get("/", response_model=CataloguesResponse)
async def get_catalogues():
    """Return catalogue data for the setup form dropdowns and example picker."""
    return CataloguesResponse(
        component_classes=get_component_classes(),
        aspects=get_aspects(),
        guidewords=get_guidewords(enabled_only=False),
        examples=_example_names(),
    )
