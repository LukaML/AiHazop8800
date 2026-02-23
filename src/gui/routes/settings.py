# src/gui/routes/settings.py
"""API endpoints for LLM provider configuration (API key management)."""
from fastapi import APIRouter, HTTPException

from ..config import RuntimeConfig
from ..models import (
    SetApiKeyRequest,
    SetApiKeyResponse,
    ProviderStatus,
    ProvidersResponse,
)

# Provider default models (from llm_client.py)
PROVIDER_DEFAULTS = {
    "openai": "gpt-3.5-turbo",
    "gemini": "gemini-2.5-flash",
    "groq": "llama-3.1-8b-instant",
}

router = APIRouter(prefix="/api/settings", tags=["settings"])


@router.post("/api-key", response_model=SetApiKeyResponse)
async def set_api_key(request: SetApiKeyRequest):
    """
    Set API key for a provider.
    Key is stored in RAM only - never logged or returned.
    """
    try:
        RuntimeConfig.set_api_key(request.provider.value, request.api_key)
        return SetApiKeyResponse(
            success=True,
            provider=request.provider.value,
            message=f"API key configured for {request.provider.value}",
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/providers", response_model=ProvidersResponse)
async def get_providers():
    """
    Get all providers and their configuration status.
    Does NOT return actual API keys.
    """
    key_status = RuntimeConfig.get_key_status()

    providers = []
    for provider, configured in key_status.items():
        providers.append(
            ProviderStatus(
                provider=provider,
                key_configured=configured,
                default_model=PROVIDER_DEFAULTS.get(provider),
            )
        )

    return ProvidersResponse(providers=providers)
