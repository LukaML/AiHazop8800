"""Load the fixed AI-HAZOP-8800 methodology catalogues from src/catalogues/*.yaml.

These catalogues are the "system inputs" of the AI-HAZOP-8800 workflow
(architecture doc §3): guidewords, taxonomy, class questions, risk model,
acceptance criterion, mitigation taxonomy, and evidence catalogue. They are
loaded once and injected into the L1–L8 prompts and risk math.
"""
from __future__ import annotations

import functools
import pathlib
from typing import Any, Dict, List

import yaml

CATALOGUES_DIR = pathlib.Path(__file__).parent / "catalogues"

_CATALOGUE_NAMES = (
    "guidewords",
    "taxonomy",
    "class_questions",
    "risk_model",
    "acceptance_criterion",
    "mitigation_taxonomy",
    "evidence_catalogue",
    "component_levels",
)


@functools.lru_cache(maxsize=None)
def load_catalogue(name: str) -> Dict[str, Any]:
    """Load a single catalogue YAML by name (cached)."""
    if name not in _CATALOGUE_NAMES:
        raise ValueError(f"Unknown catalogue '{name}'. Known: {_CATALOGUE_NAMES}")
    path = CATALOGUES_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Catalogue file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_all_catalogues() -> Dict[str, Dict[str, Any]]:
    """Load every catalogue into a single dict keyed by catalogue name."""
    return {name: load_catalogue(name) for name in _CATALOGUE_NAMES}


# ---------------------------------------------------------------------------
# Convenience accessors
# ---------------------------------------------------------------------------

def get_guidewords(enabled_only: bool = True, override: List[str] | None = None) -> List[Dict[str, Any]]:
    """Return the working guideword set.

    Args:
        enabled_only: if True, restrict to the catalogue's ``enabled`` ids.
        override: explicit list of guideword ids (or the literal ['all']) that
                  takes precedence over the catalogue default.
    """
    cat = load_catalogue("guidewords")
    all_gws: List[Dict[str, Any]] = cat.get("guidewords", [])
    by_id = {g["id"]: g for g in all_gws}

    if override:
        ids = [s.strip() for s in override if str(s).strip()]
        if len(ids) == 1 and ids[0].lower() == "all":
            return all_gws
        picked = [by_id[i] for i in ids if i in by_id]
        return picked or all_gws

    if enabled_only:
        enabled_ids = cat.get("enabled") or [g["id"] for g in all_gws]
        return [by_id[i] for i in enabled_ids if i in by_id]
    return all_gws


def get_component_classes() -> List[str]:
    """Return all taxonomy component-class ids (for GUI dropdowns)."""
    cat = load_catalogue("taxonomy")
    return [c["id"] for c in cat.get("classes", []) if c.get("id")]


def get_aspects() -> List[str]:
    """Return all taxonomy aspect ids (for GUI dropdowns)."""
    cat = load_catalogue("taxonomy")
    return [a["id"] for a in cat.get("aspects", []) if a.get("id")]


def get_class_questions(component_class: str) -> List[str]:
    """Return class-specific questions for a taxonomy class (empty if unknown)."""
    cat = load_catalogue("class_questions")
    return (cat.get("questions") or {}).get(component_class, [])


def _class_family(component_class: str) -> str:
    """Map a taxonomy class id (e.g. 'Camera-ObjectDetection') to its family."""
    return str(component_class or "").split("-", 1)[0].strip()


def get_component_level(component_class: str) -> Dict[str, Any]:
    """Return the component-level scoping rule for a class (empty dict if none).

    Result keys: ``level`` (str), ``forbidden_terms`` (List[str]),
    ``required_any`` (List[str]). Lookup is by class *family* (the part before
    the first '-'), so all 'Camera-*' classes share the perception rule.
    """
    cat = load_catalogue("component_levels")
    families = cat.get("families") or {}
    rule = families.get(_class_family(component_class))
    if not isinstance(rule, dict):
        return {}
    return {
        "level": rule.get("level", ""),
        "forbidden_terms": [str(t).lower() for t in rule.get("forbidden_terms", []) if str(t).strip()],
        "required_any": [str(t).lower() for t in rule.get("required_any", []) if str(t).strip()],
    }


def get_aspect_hint(aspect: str) -> str:
    """Return the typical hazard-contribution hint for an aspect id (best-effort)."""
    cat = load_catalogue("taxonomy")
    for a in cat.get("aspects", []):
        if a.get("id") == aspect:
            return a.get("hazard_contribution", "")
    return ""
