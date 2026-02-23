# src/gui/config.py
"""Runtime configuration — API keys are stored in-memory only for security.

Keys are never logged, persisted to disk, or returned to the frontend.
The RuntimeConfig class provides a class-level dict that all routes share.
"""
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class RuntimeConfig:
    """Singleton-like class for managing API keys in RAM only."""

    _api_keys: Dict[str, str] = {}

    # Mapping from provider to environment variable name
    _ENV_VAR_MAP = {
        "openai": "OPENAI_API_KEY",
        "gemini": "GEMINI_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    @classmethod
    def set_api_key(cls, provider: str, key: str) -> None:
        """
        Store API key in RAM only.
        NEVER log, write to disk, or return the actual key to frontend.
        """
        provider = provider.lower()
        if provider not in cls._ENV_VAR_MAP:
            raise ValueError(f"Unknown provider: {provider}")
        cls._api_keys[provider] = key.strip()
        logger.info("API key configured for provider: %s", provider)

    @classmethod
    def get_api_key(cls, provider: str) -> Optional[str]:
        """Get API key for a provider (internal use only)."""
        return cls._api_keys.get(provider.lower())

    @classmethod
    def has_api_key(cls, provider: str) -> bool:
        """Check if API key is configured for a provider."""
        provider = provider.lower()
        return bool(cls._api_keys.get(provider))

    @classmethod
    def get_env_var_name(cls, provider: str) -> Optional[str]:
        """Get the environment variable name for a provider."""
        return cls._ENV_VAR_MAP.get(provider.lower())

    @classmethod
    def clear_all_keys(cls) -> None:
        """Clear all stored API keys."""
        cls._api_keys.clear()
        logger.info("All API keys cleared from RAM")

    @classmethod
    def get_key_status(cls) -> Dict[str, bool]:
        """Get status of API key configuration for all providers."""
        return {
            provider: cls.has_api_key(provider)
            for provider in cls._ENV_VAR_MAP.keys()
        }
