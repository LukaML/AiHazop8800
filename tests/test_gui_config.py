# tests/test_gui_config.py
"""Tests for RuntimeConfig — in-memory API key management."""
import pytest

from src.gui.config import RuntimeConfig


@pytest.fixture(autouse=True)
def _clean_config():
    """Ensure a clean config for each test."""
    RuntimeConfig.clear_all_keys()
    yield
    RuntimeConfig.clear_all_keys()


class TestSetAndGetApiKey:
    def test_round_trip(self):
        RuntimeConfig.set_api_key("openai", "sk-test-123")
        assert RuntimeConfig.get_api_key("openai") == "sk-test-123"

    def test_strips_whitespace(self):
        RuntimeConfig.set_api_key("gemini", "  key-with-spaces  ")
        assert RuntimeConfig.get_api_key("gemini") == "key-with-spaces"

    def test_overwrite_key(self):
        RuntimeConfig.set_api_key("openai", "old-key")
        RuntimeConfig.set_api_key("openai", "new-key")
        assert RuntimeConfig.get_api_key("openai") == "new-key"


class TestHasApiKey:
    def test_false_initially(self):
        assert RuntimeConfig.has_api_key("openai") is False

    def test_true_after_set(self):
        RuntimeConfig.set_api_key("openai", "sk-key")
        assert RuntimeConfig.has_api_key("openai") is True

    def test_case_insensitive(self):
        RuntimeConfig.set_api_key("OpenAI", "sk-key")
        assert RuntimeConfig.has_api_key("OPENAI") is True


class TestUnknownProvider:
    def test_set_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            RuntimeConfig.set_api_key("unknown_provider", "key")


class TestGetEnvVarName:
    def test_openai(self):
        assert RuntimeConfig.get_env_var_name("openai") == "OPENAI_API_KEY"

    def test_gemini(self):
        assert RuntimeConfig.get_env_var_name("gemini") == "GEMINI_API_KEY"

    def test_groq(self):
        assert RuntimeConfig.get_env_var_name("groq") == "GROQ_API_KEY"

    def test_unknown_returns_none(self):
        assert RuntimeConfig.get_env_var_name("unknown") is None


class TestClearAllKeys:
    def test_clears_everything(self):
        RuntimeConfig.set_api_key("openai", "k1")
        RuntimeConfig.set_api_key("gemini", "k2")
        RuntimeConfig.clear_all_keys()
        assert RuntimeConfig.has_api_key("openai") is False
        assert RuntimeConfig.has_api_key("gemini") is False


class TestGetKeyStatus:
    def test_all_false_initially(self):
        status = RuntimeConfig.get_key_status()
        assert set(status.keys()) == {"openai", "gemini", "groq"}
        assert all(v is False for v in status.values())

    def test_reflects_configured(self):
        RuntimeConfig.set_api_key("openai", "sk-x")
        status = RuntimeConfig.get_key_status()
        assert status["openai"] is True
        assert status["gemini"] is False
        assert status["groq"] is False
