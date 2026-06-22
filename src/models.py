"""Pydantic data models for the AI-HAZOP-8800 pipeline phases (L1–L8).

Domain model (architecture doc §1, §5): the analysis unit is an AI *component +
aspect* evaluated under an ODD and scenario against one guideword at a time. Each
phase extends the previous one with new fields:

  L1 (AIHazopL1Row): + failure_mode
  L2 (AIHazopL2Row): + hazardous_behavior, potential_harm, potentially_dangerous
  L3 (AIHazopL3Row): + risk factors E/PF/PND/PNM/S (+ code-computed initial_risk)
  L4 (AIHazopL4Row): + safety_decision, acceptance_rationale
  L5 (AIHazopL5Row): + ai_safety_goals
  L6 (AIHazopL6Row): + respecifications, safety_functions, passive_operational_measures
  L7 (AIHazopL7Row): + residual risk factors (+ code-computed residual_risk)
  L8 (AIHazopL8Row): + evidence, open_assumptions (extends L4 so ACCEPT rows are evidenced)

All row models inherit DictLikeMixin so they interoperate with the plain dicts
that flow through the LangGraph state.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Dict compatibility mixin — lets Pydantic models behave like dicts in
# LangGraph state and helper functions that expect row["field"] access.
# ---------------------------------------------------------------------------

class DictLikeMixin:
    """Allow Pydantic models to be used like dicts (row["field"], row.get(...))."""
    def get(self, key, default=None):
        return getattr(self, key, default)
    def __getitem__(self, key):
        return getattr(self, key)
    def __setitem__(self, key, value):
        setattr(self, key, value)
    def to_dict(self):
        return self.model_dump()
    def __contains__(self, key):
        return hasattr(self, key)
    def keys(self):
        return type(self).model_fields.keys()


# Allowed L4 safety decisions (mirrors catalogues/acceptance_criterion.yaml).
SAFETY_DECISIONS = ("ACCEPT", "IMPROVE", "RESTRICT", "INVESTIGATE")


# ---------------------------------------------------------------------------
# Analysis context — replaces the old single "function" input. The user
# provides one of these per analysis unit (architecture doc §2).
# ---------------------------------------------------------------------------

class AIHazopInput(BaseModel):
    component: str = Field(description="AI component under analysis, e.g. 'Camera Object Detection'.")
    component_class: str = Field(description="Taxonomy class id, e.g. 'Camera-ObjectDetection'.")
    aspect: str = Field(description="AI aspect analysed, e.g. 'Outputs'.")
    odd: str = Field(description="Operational Design Domain slice (road type, speed, weather, ...).")
    scenario: str = Field(description="Concrete operating scenario, e.g. 'Cyclist behind parked van'.")

    @field_validator("component", "component_class", "aspect", "odd", "scenario")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# L1 row — failure mode for one (component+aspect × guideword) combination.
# Context fields (component..scenario) are stamped by the pipeline from the
# user input, not invented by the LLM; the LLM supplies guideword + failure_mode.
# ---------------------------------------------------------------------------

class AIHazopL1Row(DictLikeMixin, BaseModel):
    row_id: str = Field(description="Stable row id, e.g. 'L1-0'.")
    component: str
    component_class: str
    aspect: str
    odd: str
    scenario: str
    guideword: str = Field(description="Guideword id from the catalogue, e.g. 'no'.")
    failure_mode: str = Field(description="Component-level deviation triggered by the guideword.")

    @field_validator("row_id", "component", "guideword", "failure_mode")
    @classmethod
    def _not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()

    model_config = {"extra": "allow"}


# ---------------------------------------------------------------------------
# L2 row — vehicle-level hazard + safety-relevance triage. potentially_dangerous
# gates the downstream phases (L3–L8).
# ---------------------------------------------------------------------------

class AIHazopL2Row(AIHazopL1Row):
    hazardous_behavior: str = Field(description="Vehicle-level unsafe behavior the failure mode can cause.")
    potential_harm: str = Field(description="Who/what may be harmed and how severely.")
    potentially_dangerous: bool = Field(description="Safety-relevance triage (true if it must continue through risk analysis).")

    @field_validator("hazardous_behavior", "potential_harm")
    @classmethod
    def _hz_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()

    @field_validator("potentially_dangerous")
    @classmethod
    def _danger_not_none(cls, v: bool) -> bool:
        if v is None:
            raise ValueError("potentially_dangerous must not be null.")
        return bool(v)


# ---------------------------------------------------------------------------
# L3 row — initial risk. The LLM selects an ordinal level for each factor; the
# CODE computes initial_risk and risk_status (see src/risk_model.py).
# ---------------------------------------------------------------------------

class AIHazopL3Row(AIHazopL2Row):
    E: str = Field(description="Exposure level (catalogue scale label).")
    PF: str = Field(description="Failure-probability level.")
    PND: str = Field(description="Probability-not-detected level.")
    PNM: str = Field(description="Probability-no-mitigation level.")
    S: str = Field(description="Severity level.")
    initial_risk: float = Field(description="Code-computed R = E·PF·PND·PNM·S.")
    risk_status: str = Field(description="Code-computed acceptance status of initial_risk.")
    risk_rationale: str = Field(default="", description="Short justification for the chosen factor levels.")

    @field_validator("E", "PF", "PND", "PNM", "S")
    @classmethod
    def _factor_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Risk factor level must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# L4 row — acceptance decision against the criterion.
# ---------------------------------------------------------------------------

class AIHazopL4Row(AIHazopL3Row):
    safety_decision: str = Field(description="One of ACCEPT / IMPROVE / RESTRICT / INVESTIGATE.")
    acceptance_rationale: str = Field(default="", description="Why this decision given the initial risk and criterion.")

    @field_validator("safety_decision")
    @classmethod
    def _decision_valid(cls, v: str) -> str:
        if not v or v.strip().upper() not in SAFETY_DECISIONS:
            raise ValueError(f"safety_decision must be one of {SAFETY_DECISIONS}")
        return v.strip().upper()


# ---------------------------------------------------------------------------
# L5 row — AI Safety Goals. Generated only for rows whose L4 decision is not
# ACCEPT (the ACCEPT-skip is handled by row filtering in the pipeline).
# ---------------------------------------------------------------------------

class AIHazopL5Row(AIHazopL4Row):
    ai_safety_goals: List[str] = Field(description="High-level safety goals addressing the hazard.")

    @field_validator("ai_safety_goals")
    @classmethod
    def _goals_non_empty(cls, v: List[str]) -> List[str]:
        items = [str(g).strip() for g in (v or []) if str(g).strip()]
        if not items:
            raise ValueError("ai_safety_goals must contain at least one goal")
        return items


# ---------------------------------------------------------------------------
# L6 row — Requirements / Measures, written in requirement style (R/SF/P).
# ---------------------------------------------------------------------------

class AIHazopL6Row(AIHazopL5Row):
    respecifications: List[str] = Field(default_factory=list)
    safety_functions: List[str] = Field(default_factory=list)
    passive_operational_measures: List[str] = Field(default_factory=list)

    @field_validator("respecifications", "safety_functions", "passive_operational_measures")
    @classmethod
    def _clean_list(cls, v: List[str]) -> List[str]:
        return [str(x).strip() for x in (v or []) if str(x).strip()]


# ---------------------------------------------------------------------------
# L7 row — Residual Risk after the proposed measures. The LLM selects residual
# factor levels; the CODE recomputes residual_risk and residual_status.
# ---------------------------------------------------------------------------

class AIHazopL7Row(AIHazopL6Row):
    residual_E: str = Field(description="Residual exposure level.")
    residual_PF: str = Field(description="Residual failure-probability level.")
    residual_PND: str = Field(description="Residual probability-not-detected level.")
    residual_PNM: str = Field(description="Residual probability-no-mitigation level.")
    residual_S: str = Field(description="Residual severity level.")
    residual_risk: float = Field(description="Code-computed residual R.")
    residual_status: str = Field(description="Code-computed acceptance status of residual_risk.")
    residual_rationale: str = Field(default="", description="Why the measures lower the factors.")

    @field_validator("residual_E", "residual_PF", "residual_PND", "residual_PNM", "residual_S")
    @classmethod
    def _res_factor_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Residual risk factor level must not be empty")
        return v.strip()


# ---------------------------------------------------------------------------
# L8 row — Evidence and open assumptions. Extends L4 (NOT L5–L7) so that ACCEPT
# rows, which skip L5–L7, can still be evidenced. For improved rows the goals /
# measures / residual fields ride along as permitted extras.
# ---------------------------------------------------------------------------

class AIHazopL8Row(AIHazopL4Row):
    evidence: List[str] = Field(description="Evidence/test activities supporting the decision and measures.")
    open_assumptions: List[str] = Field(default_factory=list)

    @field_validator("evidence")
    @classmethod
    def _evidence_non_empty(cls, v: List[str]) -> List[str]:
        items = [str(e).strip() for e in (v or []) if str(e).strip()]
        if not items:
            raise ValueError("evidence must contain at least one item")
        return items

    @field_validator("open_assumptions")
    @classmethod
    def _clean_assumptions(cls, v: List[str]) -> List[str]:
        return [str(x).strip() for x in (v or []) if str(x).strip()]


__all__ = [
    "DictLikeMixin",
    "SAFETY_DECISIONS",
    "AIHazopInput",
    "AIHazopL1Row",
    "AIHazopL2Row",
    "AIHazopL3Row",
    "AIHazopL4Row",
    "AIHazopL5Row",
    "AIHazopL6Row",
    "AIHazopL7Row",
    "AIHazopL8Row",
]
