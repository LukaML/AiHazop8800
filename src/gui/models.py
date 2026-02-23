# src/gui/models.py
"""Request/response Pydantic models for the GUI REST API.

Defines typed schemas for all API endpoints: settings, analysis lifecycle,
row editing/rating/regeneration, and export.
"""
from typing import List, Optional
from pydantic import BaseModel, Field
from enum import Enum


class Provider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    GROQ = "groq"


class RagEmbedder(str, Enum):
    """RAG embedding backends."""
    LOCAL = "local"
    OPENAI = "openai"
    GEMINI = "gemini"


class RegenerationScope(str, Enum):
    """Scope for row regeneration."""
    L1 = "L1"
    L2 = "L2"
    L3 = "L3"
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

class StartAnalysisRequest(BaseModel):
    """Request to start a HAZOP analysis."""
    provider: Provider = Provider.OPENAI
    model: Optional[str] = None
    model_review: Optional[str] = None
    functions: List[str] = Field(..., min_length=1)
    notes: str = ""
    max_devs_per_gw: int = Field(default=2, ge=1, le=5)
    rag_enabled: bool = False
    rag_paths: List[str] = Field(default_factory=list)
    rag_embedder: RagEmbedder = RagEmbedder.LOCAL
    rag_min_sim: Optional[float] = Field(default=None, ge=0.0, le=1.0)


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
    """Data for a single HAZOP row with tracking fields."""
    row_id: str
    function: str
    guideword: str
    deviation_original: str
    deviation_final: str
    cause_original: str
    cause_final: str
    effect_original: str
    effect_final: str
    potentially_dangerous_ai: bool
    potentially_dangerous_human: Optional[bool] = None
    potentially_dangerous_final: bool
    rating: Rating = Rating.UNRATED
    edited_flag: bool = False
    regenerated_flag: bool = False


class AnalysisResultsResponse(BaseModel):
    """Response with full analysis results."""
    run_id: str
    status: AnalysisStatus
    provider: str
    model: str
    rag_enabled: bool
    rag_embedder: Optional[str] = None
    rows: List[RowData]
    created_at: str
    completed_at: Optional[str] = None


class EditRowRequest(BaseModel):
    """Request to edit a row's fields."""
    deviation: Optional[str] = None
    cause: Optional[str] = None
    effect: Optional[str] = None
    potentially_dangerous: Optional[bool] = None


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
#                         EXPORT MODELS
# ============================================================================

class ExportFormat(str, Enum):
    """Export file formats."""
    CSV = "csv"


# ============================================================================
#                         FILE UPLOAD MODELS
# ============================================================================

class UploadFunctionsResponse(BaseModel):
    """Response after uploading functions file."""
    success: bool
    functions: List[str]
    message: str


class UploadRagResponse(BaseModel):
    """Response after uploading RAG documents."""
    success: bool
    file_paths: List[str]
    message: str


class ImportHtmlResponse(BaseModel):
    """Response after importing HTML results."""
    success: bool
    run_id: str
    row_count: int
    message: str
