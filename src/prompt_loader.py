"""Load YAML prompt templates from src/prompts/ and render {{placeholder}} values.

Each YAML file defines instructions, constraints, few-shot examples, and a
payload template for a specific LLM call (e.g. l1_init, reviewer_patch_l2).
"""
import pathlib
import yaml
from typing import Any, Dict

PROMPTS_DIR = pathlib.Path(__file__).parent / "prompts"

def load_prompt(name: str) -> Dict[str, Any]:
    """
    Load a YAML prompt from prompts/{name}.yaml.
    Returns a dict with keys: title, inputs, instructions, output_schema,
    constraints, payload_template.
    """
    path = PROMPTS_DIR / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"Prompt file not found: {path}. "
            f"Ensure all prompt YAML files are in {PROMPTS_DIR}/"
        )
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in prompt file {path}: {e}")

def render_payload(template: str, **kwargs) -> str:
    """
    Replace {{placeholders}} in the YAML template with provided stringified values.
    All JSON-like data passed here should be json.dumps(...) from the orchestrator.
    """
    s = template
    for k, v in kwargs.items():
        s = s.replace("{{" + k + "}}", v)
    return s