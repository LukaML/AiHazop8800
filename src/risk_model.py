"""Code-side risk math for AI-HAZOP-8800 (methodology §7, architecture doc §L3/§L7).

The LLM picks an ordinal LEVEL for each risk factor (E, PF, PND, PNM, S) from the
allowed scales in catalogues/risk_model.yaml. This module — NOT the LLM — maps each
level to its numeric value, computes R = E·PF·PND·PNM·S, and compares R against the
acceptance target to produce a status. Used for both initial risk (L3) and residual
risk (L7).
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

from .catalogue_loader import load_catalogue

logger = logging.getLogger(__name__)

FACTORS = ("E", "PF", "PND", "PNM", "S")


def _levels(factor: str) -> Dict[str, float]:
    rm = load_catalogue("risk_model")
    return (rm.get("factors", {}).get(factor, {}) or {}).get("levels", {}) or {}


def factor_value(factor: str, level: Any) -> Optional[float]:
    """Resolve a factor level to its numeric value.

    Accepts an ordinal label (e.g. "medium") from the catalogue scale, or a raw
    numeric value (the LLM occasionally returns numbers directly). Returns None
    if the value cannot be resolved.
    """
    if level is None:
        return None
    levels = _levels(factor)
    # Label lookup (case-insensitive)
    if isinstance(level, str):
        key = level.strip().lower()
        if key in levels:
            return float(levels[key])
        # maybe a numeric string
        try:
            return float(level)
        except ValueError:
            logger.warning("risk_model: unknown level %r for factor %s", level, factor)
            return None
    # Numeric passthrough
    try:
        return float(level)
    except (TypeError, ValueError):
        return None


def acceptance() -> Dict[str, Any]:
    return load_catalogue("acceptance_criterion")


def compute_risk(factors: Dict[str, Any]) -> Tuple[Optional[float], str, Optional[float]]:
    """Compute R from factor levels and classify against the acceptance target.

    Args:
        factors: mapping with keys E, PF, PND, PNM, S whose values are ordinal
                 labels or numbers.

    Returns:
        (risk, status, target). ``risk`` is None if any factor is unresolvable.
        ``status`` is one of the catalogue status labels, or "INCOMPLETE".
    """
    acc = acceptance()
    target = acc.get("mem_target")
    values = {}
    for f in FACTORS:
        v = factor_value(f, factors.get(f))
        if v is None:
            logger.warning("risk_model: missing/invalid factor %s=%r", f, factors.get(f))
            return None, "INCOMPLETE", target
        values[f] = v

    risk = 1.0
    for f in FACTORS:
        risk *= values[f]

    status = classify(risk, target)
    return risk, status, target


def classify(risk: Optional[float], target: Optional[float] = None) -> str:
    """Classify a risk value against the MEM target."""
    acc = acceptance()
    if target is None:
        target = acc.get("mem_target")
    if risk is None or target is None:
        return "INCOMPLETE"
    if risk <= float(target):
        return acc.get("status_acceptable", "ACCEPTABLE")
    return acc.get("status_above", "ABOVE MEM TARGET")


def format_risk(risk: Optional[float]) -> str:
    """Human-readable risk value for worksheet cells."""
    if risk is None:
        return ""
    return f"{risk:.2e}"


def acceptance_criterion_text(rationale: str = "") -> str:
    """Methodology 'Acceptance criterion' cell: just the criterion in use (§12).

    e.g. "MEM target R ≤ 1.0e-05". ``rationale`` is accepted but ignored (the decision
    rationale lives in Safety decision)."""
    acc = acceptance()
    mode = acc.get("mode", "MEM")
    target = acc.get("mem_target")
    if mode == "MEM" and isinstance(target, (int, float)):
        return f"MEM target R ≤ {target:.1e}"
    return acc.get("game_reference") or "GAME reference"
