"""Multi-provider LLM client abstraction (OpenAI, Gemini, Groq).

All providers use the OpenAI-compatible chat completions API.  This module
handles:
  - Provider configuration and model resolution (configure / PROVIDER_DEFAULTS)
  - Raw chat completion calls (chat_raw)
  - Robust JSON parsing pipeline with fallback chain:
    parse → strip code fences → fix quotes → extract JSON block → salvage
  - Unwrapping common LLM output wrappers ({"rows": [...]})
"""
import logging
import os
import json
import random
import re
import time
from dataclasses import dataclass
from typing import Any, Optional
from openai import (
    OpenAI,
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
)

from .row_utils import WRAPPER_KEYS

logger = logging.getLogger(__name__)

# Retry policy for transient LLM errors (rate limits, timeouts, connection resets,
# provider 5xx). A single transient failure must not abort a whole pipeline run, so
# we back off and retry at this one choke point (every phase/batch/reviewer call
# funnels through chat_raw). Non-transient errors (auth, bad request) propagate.
_RETRY_EXCEPTIONS = (RateLimitError, APITimeoutError, APIConnectionError, InternalServerError)
_MAX_RETRY_ATTEMPTS = 5      # total attempts, including the first
_RETRY_BASE_DELAY = 2.0      # seconds; exponential base
_RETRY_MAX_DELAY = 30.0      # cap per-attempt wait


def _retry_after_seconds(err: Exception) -> Optional[float]:
    """Return the server-provided Retry-After delay (seconds) if present."""
    response = getattr(err, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    raw = headers.get("retry-after") or headers.get("Retry-After")
    if not raw:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None

# ---------------------------------------------------------------------------
# Provider configuration — maps provider names to base URLs, default models,
# and API key environment variables.  All providers use OpenAI-compatible
# endpoints so a single OpenAI client works for all of them.
# ---------------------------------------------------------------------------

@dataclass
class LLMConfig:
    """Configuration for the LLM provider."""
    provider: str  # openai, gemini, groq
    base_url: str
    api_key: str
    model_cheap: str
    model_review: str


PROVIDER_DEFAULTS = {
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "model_cheap": "gpt-3.5-turbo",
        "model_review": "gpt-3.5-turbo",
        "api_key_env": "OPENAI_API_KEY",
        "api_key_required": True,
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
        "model_cheap": "gemini-2.5-flash",
        "model_review": "gemini-2.5-flash",
        "api_key_env": "GEMINI_API_KEY",
        "api_key_required": True,
    },
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model_cheap": "llama-3.1-8b-instant",
        "model_review": "llama-3.3-70b-versatile",
        "api_key_env": "GROQ_API_KEY",
        "api_key_required": True,
    },
}

# Global config instance (set by configure())
_config: Optional[LLMConfig] = None
_client: Optional[OpenAI] = None


def configure(
    provider: str = "openai",
    model: Optional[str] = None,
    model_review: Optional[str] = None,
) -> None:
    """
    Initialize LLM configuration. Must be called before any chat_* functions.

    Args:
        provider: LLM provider (openai, gemini, groq)
        model: Model for generation stages (uses provider default if not set)
        model_review: Model for review stages (defaults to model if not set)
    """
    global _config, _client

    provider = provider.lower()
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError(f"Unknown provider: {provider}. Choose from: {list(PROVIDER_DEFAULTS.keys())}")

    defaults = PROVIDER_DEFAULTS[provider]

    # Resolve API key
    api_key_env = defaults["api_key_env"]
    api_key = os.getenv(api_key_env, "") if api_key_env else "not-needed"

    if defaults["api_key_required"] and not api_key:
        raise RuntimeError(f"Set {api_key_env} env var for {provider} provider.")

    # For providers that don't require API key, use a placeholder
    if not api_key:
        api_key = "not-needed"

    # Resolve models
    model_cheap = model or os.getenv("MODEL_CHEAP") or defaults["model_cheap"]
    if not model_cheap:
        raise ValueError(f"Model must be specified for {provider} provider (no default available).")

    model_review_final = model_review or os.getenv("MODEL_REVIEW") or model_cheap

    _config = LLMConfig(
        provider=provider,
        base_url=defaults["base_url"],
        api_key=api_key,
        model_cheap=model_cheap,
        model_review=model_review_final,
    )

    # Create client
    _client = OpenAI(api_key=_config.api_key, base_url=_config.base_url)

    logger.info("Configured provider: %s", provider)
    logger.info("Base URL: %s", _config.base_url)
    logger.info("Model (cheap): %s", _config.model_cheap)
    logger.info("Model (review): %s", _config.model_review)


def _ensure_configured() -> None:
    """Ensure configuration is initialized, using OpenAI defaults if not."""
    global _config, _client
    if _config is None:
        # Backward compatibility: auto-configure with OpenAI defaults
        # Check for legacy environment variable
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY env var or call configure() first.")
        configure(provider="openai")


def get_client() -> OpenAI:
    """Get configured OpenAI client instance."""
    _ensure_configured()
    if _client is None:
        raise RuntimeError("LLM client not initialized. Call configure() first.")
    return _client


# --- Module-level model properties (backward compatible) ---

def _get_model_cheap() -> str:
    _ensure_configured()
    if _config is None:
        raise RuntimeError("LLM config not initialized. Call configure() first.")
    return _config.model_cheap


def _get_model_review() -> str:
    _ensure_configured()
    if _config is None:
        raise RuntimeError("LLM config not initialized. Call configure() first.")
    return _config.model_review


# For backward compatibility with chains.py imports
class _ModelProperty:
    """Lazy string that reads from config when converted to str."""
    def __init__(self, getter):
        self._getter = getter

    def __str__(self):
        return self._getter()

    def __repr__(self):
        return repr(self._getter())

    def __eq__(self, other):
        if isinstance(other, _ModelProperty):
            return self._getter() == other._getter()
        return self._getter() == other

    def __hash__(self):
        return hash(self._getter())

    def __bool__(self):
        return True  # Always truthy so `if model:` checks work


# These act as lazy properties that read from _config when accessed
MODEL_CHEAP = _ModelProperty(_get_model_cheap)
MODEL_REVIEW = _ModelProperty(_get_model_review)


def _resolve_model(model) -> str:
    """Convert model parameter to string, handling _ModelProperty instances."""
    if model is None:
        return _config.model_cheap if _config else ""
    if isinstance(model, _ModelProperty):
        return str(model)
    return str(model)

# --- Core call helpers ---

def _fix_json_quotes(text: str) -> str:
    """Fix quote issues in LLM JSON output.

    1. Replace Unicode curly/smart quotes with ASCII single quotes.
    2. Escape unescaped double quotes inside JSON string values.
    """
    # Normalize Unicode quotes to single quotes
    for ch in '\u201c\u201d\u201e\u201f':  # " " „ ‟
        text = text.replace(ch, "'")
    for ch in '\u2018\u2019\u201a\u201b':  # ' ' ‚ ‛
        text = text.replace(ch, "'")

    # If it already parses, return as-is
    try:
        json.loads(text)
        return text
    except json.JSONDecodeError:
        pass

    # Escape inner double quotes inside string values
    result = []
    i = 0
    in_string = False

    while i < len(text):
        ch = text[i]

        if not in_string:
            result.append(ch)
            if ch == '"':
                in_string = True
        else:
            if ch == '\\':
                result.append(ch)
                i += 1
                if i < len(text):
                    result.append(text[i])
            elif ch == '"':
                # Look ahead past whitespace for structural JSON chars
                j = i + 1
                while j < len(text) and text[j] in ' \t\n\r':
                    j += 1
                if j >= len(text) or text[j] in ':,}]':
                    result.append(ch)
                    in_string = False
                else:
                    result.append('\\"')
            else:
                result.append(ch)
        i += 1

    return ''.join(result)


def _salvage_json(text: str) -> str:
    """
    Best-effort: extract the largest well-formed JSON object/array prefix.
    - Finds first '{' or '[' and then balances braces/brackets while tracking quotes/escapes.
    - Returns the balanced prefix; raises if none can be found.
    """
    s = text
    start = None
    for i, ch in enumerate(s):
        if ch in "{[":
            start = i
            break
    if start is None:
        raise ValueError("No JSON start '{' or '[' found.")

    stack = []
    in_str = False
    esc = False
    end = None
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if not stack:
                    raise ValueError("Unbalanced JSON: extra closing bracket.")
                opener = stack.pop()
                if (opener, ch) not in {("{","}"), ("[","]")}:
                    raise ValueError("Mismatched JSON brackets.")
                if not stack:
                    end = i + 1  # include this closing char
                    break
    if end is None:
        # Not fully balanced; try to return until the last complete pair we saw
        if stack:
            # Attempt a softer trim: find last comma before failure and close array/object
            prefix = s[start:len(s)]
            # crude but practical: cut at last comma and try to close
            last_comma = prefix.rfind(",")
            if last_comma > 0:
                prefix = prefix[:last_comma]
            # Try to close with the matching closers
            closers = "".join("}" if c=="{" else "]" for c in reversed(stack))
            return prefix + closers
        raise ValueError("Could not balance JSON.")
    return s[start:end]


def _create_with_retry(client: OpenAI, request_kwargs: dict) -> Any:
    """Call chat.completions.create, retrying transient errors with backoff.

    Rate limits (429), timeouts, connection resets and provider 5xx are retried
    with exponential backoff + jitter, honouring a server ``Retry-After`` header
    when present. After ``_MAX_RETRY_ATTEMPTS`` the last error is re-raised.
    """
    last_err: Optional[Exception] = None
    for attempt in range(1, _MAX_RETRY_ATTEMPTS + 1):
        try:
            return client.chat.completions.create(**request_kwargs, timeout=300)
        except _RETRY_EXCEPTIONS as err:
            last_err = err
            if attempt >= _MAX_RETRY_ATTEMPTS:
                break
            delay = _retry_after_seconds(err)
            if delay is None:
                delay = min(_RETRY_BASE_DELAY * (2 ** (attempt - 1)), _RETRY_MAX_DELAY)
                delay += random.uniform(0, delay * 0.25)  # jitter to avoid thundering herd
            logger.warning(
                "LLM call failed (%s), retrying in %.1fs (attempt %d/%d)",
                type(err).__name__, delay, attempt, _MAX_RETRY_ATTEMPTS,
            )
            time.sleep(delay)
    assert last_err is not None
    raise last_err


def chat_raw(prompt_text: str, model: Optional[str] = None, json_mode: bool = False) -> str:
    """
    Low-level wrapper around Chat Completions.
    You pass a single combined string (instructions + payload).
    Returns assistant text (string).

    Args:
        prompt_text: The prompt to send
        model: Model to use (defaults to configured model_cheap)
        json_mode: If True, request JSON response format (supported by OpenAI, Groq, Gemini)
    """
    _ensure_configured()
    chosen = _resolve_model(model)

    logger.debug("LLM CALL - MODEL: %s", chosen)
    logger.debug("LLM CALL - PROMPT length: %d", len(prompt_text))
    logger.debug("LLM CALL - PROMPT preview: %s", prompt_text[:800] + ("..." if len(prompt_text) > 800 else ""))

    client = get_client()

    # Build request kwargs - use stronger JSON instructions for Groq/Llama
    system_msg = (
        "You are a helpful assistant that outputs valid JSON only. "
        "CRITICAL: Use proper JSON syntax - use double quotes for strings, "
        "use square brackets [] for arrays (NOT parentheses), "
        "use null (NOT None), use true/false (NOT True/False). "
        "Do not write code. Output the JSON data directly."
    )
    request_kwargs = {
        "model": chosen,
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt_text},
        ],
        "temperature": 0,
    }

    # Add JSON mode for providers that support it (OpenAI, Groq, Gemini via OpenAI compat)
    if json_mode:
        request_kwargs["response_format"] = {"type": "json_object"}

    resp = _create_with_retry(client, request_kwargs)
    text = resp.choices[0].message.content or ""
    logger.debug("LLM CALL - RAW RESPONSE (first 800 chars): %s", text[:800] + ("..." if len(text) > 800 else ""))
    return text

# ---------------------------------------------------------------------------
# Robust JSON parsing pipeline.  LLMs often return JSON wrapped in markdown
# code fences, with Unicode curly quotes, or double-encoded as a string.
# The fallback chain: strip fences → fix quotes → direct parse → extract
# JSON block → salvage balanced prefix → unwrap double-encoding → unwrap
# common dict wrappers.
# ---------------------------------------------------------------------------

_JSON_BLOCK = re.compile(r"\{.*\}|\[.*\]", re.S)
_CODE_FENCE = re.compile(r"```(\w*)\s*([\s\S]*?)```", re.I)

def _strip_code_fences(text: str) -> str:
    """
    Remove markdown code fences (```json ... ``` or ``` ... ```) from text.
    Returns the content inside the fence, or original text if no fence found.
    Raises ValueError if a non-JSON code block (e.g., python) is detected.
    """
    match = _CODE_FENCE.search(text)
    if match:
        lang = match.group(1).lower()
        content = match.group(2).strip()
        # Reject non-JSON code blocks - model misunderstood the task
        if lang and lang not in ("json", ""):
            raise ValueError(
                f"Model returned {lang} code instead of JSON. "
                "This usually means the model misunderstood the task. "
                "Try a different model (e.g., gpt-4o or gemini-2.0-flash)."
            )
        return content
    return text

def _extract_json_block(text: str) -> Optional[str]:
    """
    Grab the first JSON-looking block ({...} or [...]) from a string.
    First strips any markdown code fences.
    """
    # First try to extract from code fences
    text = _strip_code_fences(text)
    m = _JSON_BLOCK.search(text.strip())
    return m.group(0) if m else None

def _looks_like_json(s: str) -> bool:
    s = s.strip()
    return (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]"))

def chat_json(prompt_text: str, model: Optional[str] = None) -> Any:
    _ensure_configured()
    chosen = _resolve_model(model)

    logger.debug("chat_json - MODEL: %s", chosen)
    logger.debug("chat_json - PROMPT len: %d", len(prompt_text))
    # Don't use json_mode - it causes issues with Groq (outputs Python tuples instead of arrays)
    # OpenAI/Gemini work fine without it
    text = chat_raw(prompt_text, model=chosen, json_mode=False)
    logger.debug("chat_json - RAW TEXT LENGTH: %d", len(text))

    # 0) Strip markdown code fences if present (common with Groq/Llama models)
    text_clean = _strip_code_fences(text)

    # 0.5) Fix common quote issues (Unicode curly quotes, unescaped inner quotes)
    text_clean = _fix_json_quotes(text_clean)

    # 1) Try straight parse
    try:
        return json.loads(text_clean)
    except Exception:
        pass

    logger.debug("chat_json - Direct JSON parse FAILED")

    # 2) Try extracting a JSON-looking block
    block = _extract_json_block(text_clean) or text_clean
    logger.debug("chat_json - Extracted block length: %d", len(block))
    try:
        logger.debug("chat_json - Trying block JSON parse...")
        obj = json.loads(block)
    except Exception:
        logger.debug("chat_json - Block JSON parse FAILED, trying salvage...")
        # 3) Salvage: recover the largest balanced prefix
        try:
            logger.debug("chat_json - Salvaging JSON...")
            repaired = _salvage_json(text_clean)
            obj = json.loads(repaired)
        except Exception as e:
            raise ValueError(f"Model returned non-JSON or unparsable content:\n{e}\n---\n{text}")

    # 4) Unwrap double-encoded JSON up to 3 times
    unwrap_guard = 0
    while isinstance(obj, str) and _looks_like_json(obj) and unwrap_guard < 3:
        obj = json.loads(obj)
        unwrap_guard += 1

    # 5) Unwrap common object wrappers (json_mode sometimes forces object output)
    # If we got {"rows": [...]} or {"data": [...]} etc., extract the array
    if isinstance(obj, dict) and len(obj) == 1:
        key = next(iter(obj))
        if key.lower() in WRAPPER_KEYS:
            inner = obj[key]
            if isinstance(inner, list):
                logger.debug("chat_json - Unwrapped array from key '%s'", key)
                obj = inner

    logger.debug("chat_json - FINAL TYPE: %s", type(obj).__name__)
    if isinstance(obj, list):
        logger.debug("chat_json - FINAL LIST length: %d", len(obj))
        if obj:
            try:
                logger.debug("chat_json - SAMPLE ROW: %s", str(obj[0])[:200])
            except Exception:
                pass
    if isinstance(obj, dict):
        logger.debug("chat_json - FINAL DICT keys: %s", list(obj.keys()))
    return obj