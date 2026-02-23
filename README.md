# HazopLLM

An LLM-powered HAZOP (Hazard and Operability Study) analysis pipeline. Generates safety analysis tables for system functions using a three-stage LLM workflow with automated validation and review loops.

## Setup

### Requirements

- Python 3.10+
- An API key for at least one supported LLM provider

### Installation

```bash
pip install -r requirements.txt
```

### LLM Provider Configuration

Set the API key for your chosen provider as an environment variable:

```bash
export OPENAI_API_KEY="your-key"   # OpenAI (default provider)
export GEMINI_API_KEY="your-key"   # Google Gemini
export GROQ_API_KEY="your-key"     # Groq
```

| Provider | Default Model | API Key Env Var |
|----------|---------------|-----------------|
| openai   | gpt-3.5-turbo | `OPENAI_API_KEY` |
| gemini   | gemini-2.5-flash | `GEMINI_API_KEY` |
| groq     | llama-3.1-8b-instant | `GROQ_API_KEY` |

## Usage

### CLI Pipeline

```bash
# Basic usage (OpenAI, default)
python -m src.run_pipeline src/functions.txt --notes "System description" --outdir out

# Specify provider and model
python -m src.run_pipeline src/functions.txt --provider gemini --outdir out

# With RAG (Retrieval-Augmented Generation) from reference documents
python -m src.run_pipeline src/functions.txt --use-rag --rag-files "docs/,specs.pdf" --rag-embedder local

# Limit deviations per guideword
python -m src.run_pipeline src/functions.txt --max_devs_per_gw 2 --outdir out
```

**Input:** A functions file (plain text with one function per line, YAML with `functions` key, or JSON array).

**Output:** HTML report (`out/hazop.html`) and CSV export (`out/hazop.csv`).

### Web GUI

```bash
python -m src.gui.app
```

Opens a browser-based interface for interactive analysis with human-in-the-loop editing, row rating, regeneration, and export.

## Architecture

### Three-Stage Pipeline (L1 -> L2 -> L3)

The pipeline generates HAZOP rows through three successive enrichment stages:

1. **L1 (Deviations):** Generates deviation descriptions for each function x guideword combination using the 11 standard HAZOP guidewords (no, more, less, as well as, part of, reverse, other than, early, late, before, after).

2. **L2 (Causes):** Adds root cause analysis to each L1 row.

3. **L3 (Effects):** Adds system-level effects and a safety triage flag (`potentially_dangerous`).

Each stage follows an init-validate-review-repair cycle:

```
INIT -> VALIDATE --[pass]--> next stage
           |
        [issues]
           |
        REVIEW -> REGEN -> VALIDATE -> ...
```

After L3, a holistic cross-stage review checks consistency across all rows and can cascade fixes back through earlier stages.

### Key Modules

| Module | Purpose |
|--------|---------|
| `src/graph_full.py` | LangGraph state machine defining the full pipeline |
| `src/run_pipeline.py` | CLI entry point |
| `src/chains.py` | LLM chain functions for generation, review, and patching |
| `src/validators.py` | Pydantic-based validation with coverage checks |
| `src/models.py` | Data models (HazopL1Row, HazopL2Row, HazopL3Row) |
| `src/llm_client.py` | Multi-provider LLM client with JSON parsing |
| `src/rag_utils.py` | Document loading and retrieval (TF-IDF or OpenAI embeddings) |
| `src/prompts/` | YAML prompt templates for each LLM call |

### RAG Support

The pipeline optionally augments LLM context with retrieved content from reference documents (TXT, CSV, XLSX, PDF). Two embedding backends are supported:
- `local`: TF-IDF vectorization (no API calls needed)
- `openai`: OpenAI text embeddings

## Testing

```bash
python -m pytest tests/ -v
```
