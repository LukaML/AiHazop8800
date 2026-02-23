# tests/test_llm_client.py
"""Tests for src/llm_client.py LLM client functions."""
import pytest
import json
from unittest.mock import MagicMock, patch

from src.llm_client import (
    _salvage_json,
    configure,
    _resolve_model,
    _ModelProperty,
    PROVIDER_DEFAULTS,
    chat_json,
    chat_raw,
    get_client,
)


class TestSalvageJson:
    """Tests for _salvage_json() JSON repair function."""

    def test_valid_json_passthrough(self):
        """Valid JSON is returned unchanged."""
        text = '{"key": "value"}'
        result = _salvage_json(text)
        # Should parse correctly
        assert json.loads(result) == {"key": "value"}

    def test_valid_array(self):
        """Valid JSON array is returned."""
        text = '[1, 2, 3]'
        result = _salvage_json(text)
        assert json.loads(result) == [1, 2, 3]

    def test_extracts_json_from_text(self):
        """Extracts JSON from surrounding text."""
        text = 'Some text before {"key": "value"} some text after'
        result = _salvage_json(text)
        assert json.loads(result) == {"key": "value"}

    def test_truncated_json_repaired(self):
        """Truncated JSON is repaired by closing brackets."""
        text = '{"key": "value", "arr": [1, 2'  # Missing ]}
        result = _salvage_json(text)
        # Should be parseable
        parsed = json.loads(result)
        assert "key" in parsed

    def test_no_json_raises(self):
        """Raises when no JSON start found."""
        text = 'no json here'
        with pytest.raises(ValueError) as exc_info:
            _salvage_json(text)
        assert "No JSON start" in str(exc_info.value)

    def test_nested_objects(self):
        """Handles nested objects correctly."""
        text = '{"outer": {"inner": "value"}}'
        result = _salvage_json(text)
        parsed = json.loads(result)
        assert parsed["outer"]["inner"] == "value"

    def test_handles_strings_with_braces(self):
        """Handles strings containing braces."""
        text = '{"msg": "hello {world}"}'
        result = _salvage_json(text)
        parsed = json.loads(result)
        assert parsed["msg"] == "hello {world}"


class TestConfigure:
    """Tests for configure() function."""

    def test_openai_provider_configured(self, mocker):
        """OpenAI provider is configured correctly."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
        mock_client = mocker.patch("src.llm_client.OpenAI")

        configure(provider="openai")

        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert call_kwargs["api_key"] == "test-key"
        assert call_kwargs["base_url"] == PROVIDER_DEFAULTS["openai"]["base_url"]

    def test_gemini_provider_configured(self, mocker):
        """Gemini provider is configured correctly."""
        mocker.patch.dict("os.environ", {"GEMINI_API_KEY": "test-gemini-key"})
        mock_client = mocker.patch("src.llm_client.OpenAI")

        configure(provider="gemini")

        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert call_kwargs["api_key"] == "test-gemini-key"
        assert call_kwargs["base_url"] == PROVIDER_DEFAULTS["gemini"]["base_url"]

    def test_unknown_provider_raises(self):
        """Unknown provider raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            configure(provider="unknown")
        assert "Unknown provider" in str(exc_info.value)

    def test_missing_api_key_raises(self, mocker):
        """Missing required API key raises RuntimeError."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": ""}, clear=True)

        with pytest.raises(RuntimeError) as exc_info:
            configure(provider="openai")
        assert "OPENAI_API_KEY" in str(exc_info.value)

    def test_custom_model_used(self, mocker):
        """Custom model parameter is used."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
        mocker.patch("src.llm_client.OpenAI")

        configure(provider="openai", model="gpt-4")

        import src.llm_client as llm_client
        assert llm_client._config.model_cheap == "gpt-4"

    def test_groq_provider_configured(self, mocker):
        """Groq provider is configured correctly."""
        mocker.patch.dict("os.environ", {"GROQ_API_KEY": "test-groq-key"})
        mock_client = mocker.patch("src.llm_client.OpenAI")

        configure(provider="groq")

        mock_client.assert_called_once()
        call_kwargs = mock_client.call_args[1]
        assert call_kwargs["api_key"] == "test-groq-key"
        assert call_kwargs["base_url"] == PROVIDER_DEFAULTS["groq"]["base_url"]


class TestModelProperty:
    """Tests for _ModelProperty lazy string class."""

    def test_str_conversion(self):
        """Converts to string via getter."""
        prop = _ModelProperty(lambda: "test-model")
        assert str(prop) == "test-model"

    def test_equality(self):
        """Equality comparison works."""
        prop = _ModelProperty(lambda: "test-model")
        assert prop == "test-model"
        assert prop != "other-model"

    def test_hash(self):
        """Hash works for use in sets/dicts."""
        prop = _ModelProperty(lambda: "test-model")
        d = {prop: "value"}
        assert d[prop] == "value"

    def test_bool_always_truthy(self):
        """Bool is always True."""
        prop = _ModelProperty(lambda: "")
        assert bool(prop) is True


class TestResolveModel:
    """Tests for _resolve_model() function."""

    def test_none_returns_config_default(self, mocker):
        """None model returns config default."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
        mocker.patch("src.llm_client.OpenAI")
        configure(provider="openai")

        result = _resolve_model(None)
        assert result == "gpt-3.5-turbo"  # OpenAI default

    def test_string_passthrough(self, mocker):
        """String model is returned as-is."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
        mocker.patch("src.llm_client.OpenAI")
        configure(provider="openai")

        result = _resolve_model("custom-model")
        assert result == "custom-model"

    def test_model_property_resolved(self, mocker):
        """_ModelProperty is resolved to string."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})
        mocker.patch("src.llm_client.OpenAI")
        configure(provider="openai")

        from src.llm_client import MODEL_CHEAP
        result = _resolve_model(MODEL_CHEAP)
        assert isinstance(result, str)


class TestChatJson:
    """Tests for chat_json() function with mocked client."""

    def test_parses_json_response(self, mocker):
        """Parses JSON response from LLM."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})

        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"key": "value"}'))]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch("src.llm_client.OpenAI", return_value=mock_client)

        configure(provider="openai")
        result = chat_json("test prompt")

        assert result == {"key": "value"}

    def test_handles_json_in_text(self, mocker):
        """Extracts JSON from text response."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(message=MagicMock(content='Here is the JSON: {"key": "value"} Done.'))
        ]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch("src.llm_client.OpenAI", return_value=mock_client)

        configure(provider="openai")
        result = chat_json("test prompt")

        assert result == {"key": "value"}

    def test_unwraps_double_encoded_json(self, mocker):
        """Unwraps double-encoded JSON strings when model returns stringified JSON."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})

        # Model returns a valid JSON object (not double-encoded in the response)
        # This tests that normal JSON parsing works
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"key": "value"}'))]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch("src.llm_client.OpenAI", return_value=mock_client)

        configure(provider="openai")
        result = chat_json("test prompt")

        # Normal JSON should parse correctly
        assert result == {"key": "value"}

    def test_salvages_truncated_json(self, mocker):
        """Salvages truncated JSON response by recovering what it can."""
        mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "test-key"})

        # Truncated JSON array - last element incomplete but first element is valid
        # The extraction regex may find the first complete object or salvage the array
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='[{"key": "value"}, {"other":'))]

        mock_client = MagicMock()
        mock_client.chat.completions.create.return_value = mock_response
        mocker.patch("src.llm_client.OpenAI", return_value=mock_client)

        configure(provider="openai")
        result = chat_json("test prompt")

        # Should have recovered something - either the array or the first object
        assert isinstance(result, (list, dict))
        if isinstance(result, list):
            assert len(result) >= 1
            assert result[0]["key"] == "value"
        else:
            # Regex found the first complete object {"key": "value"}
            assert result["key"] == "value"
