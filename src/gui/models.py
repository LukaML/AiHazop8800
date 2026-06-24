# src/gui/models.py
"""Request/response Pydantic models for the GUI REST API.

Defines typed schemas for all API endpoints: settings, analysis lifecycle,
row editing/rating/regeneration, and export. The GUI drives the eight-phase
AI-HAZOP-8800 pipeline (L1–L8); the analysis unit is a structured *context*
(component / component_class / aspect / odd / scenario), not a flat function list.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"


class RegenerationScope(str, Enum):
    """Scope for row regeneration — the phase to patch before cascading downstream."""
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
    L4 = "L4"
    L5 = "L5"
    L6 = "L6"
    L7 = "L7"
    L8 = "L8"
    ALL = "ALL"


class Rating(str, Enum):
    """Human rating options for rows."""
    CORRECT = "correct"
    PARTIALLY_CORRECT = "partially_correct"
    INCORRECT = "incorrect"
    UNRATED = "unrated"


# ============================================================================
#                         SETTINGS MODELS
# ============================================================================

class SetApiKeyRequest(BaseModel):
    """Request to set an API key."""
    provider: Provider
    api_key: str = Field(..., min_length=1)


class SetApiKeyResponse(BaseModel):
    """Response after setting an API key."""
    success: bool
    provider: str
    message: str


class ProviderStatus(BaseModel):
    """Status of a single provider."""
    provider: str
    key_configured: bool
    default_model: Optional[str]


class ProvidersResponse(BaseModel):
    """Response with all providers and their status."""
    providers: List[ProviderStatus]


# ============================================================================
#                         ANALYSIS MODELS
# ============================================================================

class ContextInput(BaseModel):
    """One AI-HAZOP-8800 analysis context (one component+aspect to analyse)."""
    component: str = Field(..., min_length=1)
    component_class: str = Field(..., min_length=1)
    aspect: str = Field(..., min_length=1)
    odd: str = Field(..., min_length=1)
    scenario: str = Field(..., min_length=1)


class StartAnalysisRequest(BaseModel):
    """Request to start an AI-HAZOP-8800 analysis."""
    provider: Provider = Provider.OPENAI
    model: Optional[str] = None
    model_review: Optional[str] = None
    contexts: List[ContextInput] = Field(..., min_length=1)
    notes: str = ""
    guidewords: Optional[List[str]] = None  # override the catalogue's enabled set


class AnalysisStatus(str, Enum):
    """Status of an analysis run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class AnalysisStatusResponse(BaseModel):
    """Response with analysis status."""
    run_id: str
    status: AnalysisStatus
    progress: Optional[str] = None
    error: Optional[str] = None


class StartAnalysisResponse(BaseModel):
    """Response after starting an analysis."""
    run_id: str
    status: AnalysisStatus
    message: str


# ============================================================================
#                         ROW MODELS
# ============================================================================

class RowData(BaseModel):
    """Data for a single worksheet row.

    ``original`` and ``final`` are the full merged L1–L8 field maps (lists kept as
    lists, code-computed risk kept numeric); ``final`` reflects human edits and
    regeneration. The frontend formats/diffs these via its field metadata.
    """
    row_id: str
    display_id: str = ""
    component_index: int = 0
    component: str = ""
    guideword: str = ""
    original: Dict[str, Any] = Field(default_factory=dict)
    final: Dict[str, Any] = Field(default_factory=dict)
    dangerous_final: bool = False
    complete: bool = True
    rating: Rating = Rating.UNRATED
    edited_flag: bool = False
    regenerated_flag: bool = False


class AnalysisResultsResponse(BaseModel):
    """Response with full analysis results."""
    run_id: str
    status: AnalysisStatus
    provider: str
    model: str
    components: int = 0
    rows: List[RowData]
    created_at: str
    completed_at: Optional[str] = None


class EditRowRequest(BaseModel):
    """Request to edit a row's editable fields.

    ``fields`` maps editable field keys (e.g. ``failure_mode``, ``E``,
    ``ai_safety_goals``) to new values (strings, bools, or lists). ``rating`` is
    optional and handled separately if present.
    """
    fields: Dict[str, Any] = Field(default_factory=dict)


class EditRowResponse(BaseModel):
    """Response after editing a row."""
    success: bool
    row: RowData
    message: str


class RateRowRequest(BaseModel):
    """Request to rate a row."""
    rating: Rating


class RateRowResponse(BaseModel):
    """Response after rating a row."""
    success: bool
    row_id: str
    rating: Rating
    message: str


class RegenerateRowsRequest(BaseModel):
    """Request to regenerate specific rows."""
    row_ids: List[str] = Field(..., min_length=1)
    scope: RegenerationScope
    suggestion: str = ""


class RegenerateRowsResponse(BaseModel):
    """Response after regenerating rows."""
    success: bool
    regenerated_row_ids: List[str]
    rows: List[RowData]
    message: str


# ============================================================================
#                         CATALOGUE / FORM MODELS
# ============================================================================

class CataloguesResponse(BaseModel):
    """Catalogue data that drives the setup form (classes, aspects, guidewords)."""
    component_classes: List[str]
    aspects: List[str]
    guidewords: List[Dict[str, Any]]
    examples: List[str]


# ============================================================================
#                         EXPORT / UPLOAD MODELS
# ============================================================================

class ExportFormat(str, Enum):
    """Export file formats."""
    CSV = "csv"


class UploadContextResponse(BaseModel):
    """Response after uploading a context file (.yaml/.json)."""
    success: bool
    contexts: List[ContextInput]
    message: str


class LoadExampleResponse(BaseModel):
    """Response with the contexts loaded from a named example file."""
    success: bool
    contexts: List[ContextInput]
    message: str


class ImportHtmlResponse(BaseModel):
    """Response after importing HTML results."""
    success: bool
    run_id: str
    row_count: int
    message: str
