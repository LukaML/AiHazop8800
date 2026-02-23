# tests/test_prompt_loader.py
"""Tests for prompt_loader.py — YAML loading and placeholder rendering."""
import pytest

from src.prompt_loader import load_prompt, render_payload


class TestLoadPrompt:
    """Tests for the load_prompt function."""

    def test_load_existing_prompt(self):
        result = load_prompt("l1_init")
        assert isinstance(result, dict)
        assert "title" in result

    def test_load_nonexistent_prompt_raises(self):
        with pytest.raises(FileNotFoundError, match="Prompt file not found"):
            load_prompt("nonexistent_prompt_xyz")

    def test_loaded_prompt_has_expected_keys(self):
        result = load_prompt("l1_init")
        assert "instructions" in result
        assert "payload_template" in result

    def test_all_stage_prompts_loadable(self):
        """All main stage prompts should load without error."""
        for name in ("l1_init", "l2_init", "l3_init"):
            p = load_prompt(name)
            assert isinstance(p, dict)
            assert "instructions" in p


class TestRenderPayload:
    """Tests for the render_payload function."""

    def test_render_replaces_placeholder(self):
        template = "Hello {{name}}, welcome!"
        result = render_payload(template, name="World")
        assert result == "Hello World, welcome!"

    def test_render_multiple_placeholders(self):
        template = "{{a}} and {{b}} and {{c}}"
        result = render_payload(template, a="X", b="Y", c="Z")
        assert result == "X and Y and Z"

    def test_render_no_match_unchanged(self):
        template = "Nothing to {{replace}} here"
        result = render_payload(template, other="value")
        assert "{{replace}}" in result

    def test_render_empty_template(self):
        result = render_payload("", key="val")
        assert result == ""

    def test_render_json_value(self):
        import json
        template = "Rows: {{rows_json}}"
        rows = [{"id": 1}]
        result = render_payload(template, rows_json=json.dumps(rows))
        assert '"id": 1' in result
