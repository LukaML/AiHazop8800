"""Pydantic data models for the three HAZOP pipeline stages (L1, L2, L3).

Each stage extends the previous one with additional fields:
  L1 (HazopL1Row): function + guideword + deviation
  L2 (HazopL2Row): adds root cause
  L3 (HazopL3Row): adds system-level effect and safety triage boolean

All row models inherit DictLikeMixin so they can be used interchangeably with
plain dicts throughout the pipeline (e.g. row["field"], row.get("field")).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Dict compatibility mixin — lets Pydantic models behave like dicts in
# LangGraph state and helper functions that expect row["field"] access.
# ---------------------------------------------------------------------------

class DictLikeMixin:
    """Allow Pydantic models to be used like dicts in the LangGraph state and
    helper functions (e.g. row["field"], row.get("field"))."""
    # Dict-style read access — used throughout the pipeline where rows may be dicts or Pydantic models
    def get(self, key, default=None):
        return getattr(self, key, default)
    def __getitem__(self, key):
        return getattr(self, key)
    # Dict-style write access — safe because our models are not frozen
    def __setitem__(self, key, value):
        setattr(self, key, value)
    # Conversion helper
    def to_dict(self):
        # Pydantic v2 model_dump(); enum values stay as strings when use_enum_values is enabled
        return self.model_dump()
    # Membership test — allows "field" in row to work like a dict
    def __contains__(self, key):
        return hasattr(self, key)
    # Key listing — allows row.keys() to work like a dict
    def keys(self):
        return type(self).model_fields.keys()


# ---------------------------------------------------------------------------
# The 11 standard HAZOP guidewords cover temporal, quantitative, and
# qualitative deviations from intended function.  Every function in the
# analysis is evaluated against all 11 guidewords.
# ---------------------------------------------------------------------------

class Guideword(str, Enum):
    NO = "no"                    # complete negation of the intended function
    MORE = "more"                # quantitative increase
    LESS = "less"                # quantitative decrease
    AS_WELL_AS = "as well as"    # qualitative modification increase
    PART_OF = "part of"          # qualitative modification decrease
    REVERSE = "reverse"          # logical opposite of the intended function
    OTHER_THAN = "other than"    # completely different, other operation modes
    EARLY = "early"              # early reaction – time
    LATE = "late"                # late reaction – time
    BEFORE = "before"            # earlier than planned/expected order
    AFTER = "after"              # later than planned/expected order

# ---------------------------------------------------------------------------
# L1 row — produced by the first LLM stage (deviation generation).
# Contains the function, guideword, deviation text, and optional
# applicability scoring fields.
# ---------------------------------------------------------------------------

class HazopL1Row(DictLikeMixin, BaseModel):
    row_id: str = Field(
        description="Id of row"
    )
    function: str
    guideword: Guideword
    deviation: str

    applicability_score: Optional[float] = Field(
        default=None, ge=0.0, le=1.0,
        description="0..1 subjective fit for guideword→function (optional)."
    )
    applicability_note: Optional[str] = Field(
        default=None,
        description="Short reason if applicability is weak (optional)."
    )

    @field_validator("row_id", "function", "guideword", "deviation")
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Field must not be empty")
        return v.strip()
    
    model_config = {
        "use_enum_values": True  # serialize Guideword as its string value
    }
    

# ---------------------------------------------------------------------------
# L2 row — produced by the second LLM stage (cause generation).
# Extends L1 by adding a root cause field.
# ---------------------------------------------------------------------------

class HazopL2Row(HazopL1Row):
    cause: str = Field(description="Root cause of the deviation.")

    @field_validator("cause")
    @classmethod
    def _cause_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Cause must not be empty.")
        return v.strip()
    

# ---------------------------------------------------------------------------
# L3 row — produced by the third LLM stage (effect generation + safety triage).
# Extends L2 by adding the system-level effect and a boolean flag indicating
# whether the combination of deviation, cause, and effect is potentially dangerous.
# ---------------------------------------------------------------------------

class HazopL3Row(HazopL2Row):
    effect: str = Field(description="System-level effect if the cause occurs.")

    # Phase 1 addition: safety relevance triage indicator.  This boolean
    # specifies whether the combination of deviation, cause and effect is
    # potentially dangerous.  It is required at L3 and must be either
    # True or False.  If it cannot be determined, it should default
    # to False.
    potentially_dangerous: bool = Field(
        description="Safety relevance triage indicator (true if potentially dangerous, false otherwise)."
    )

    @field_validator("effect")
    @classmethod
    def _effect_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Effect must not be empty.")
        return v.strip()

    @field_validator("potentially_dangerous")
    @classmethod
    def _danger_not_none(cls, v: bool) -> bool:
        # Ensure the boolean is provided (bool or truthy/falsy) and not None.
        if v is None:
            raise ValueError("Potentially dangerous must not be null.")
        return bool(v)
    

__all__ = [
    "Guideword",
    "HazopL1Row",
    "HazopL2Row",
    "HazopL3Row",
]
