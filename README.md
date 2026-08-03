# HazopLLM

HazopLLM implements the AI-HAZOP-8800 methodology as an LLM-powered, eight-phase
hazard-analysis pipeline. It accepts one or more structured AI-component contexts,
runs generation, validation, review, and repair loops for phases L1 through L8, and
exports an HTML/CSV safety worksheet. A FastAPI web GUI is included for interactive
analysis, editing, regeneration, and export.

## Requirements

- Python 3.10 or newer (64-bit recommended)
- An API key for OpenAI, Google Gemini, or Groq to run an analysis
- A terminal opened in the repository root

## Windows setup (PowerShell)

The commands below use Python 3.11 as an example. `py -0p` lists the Python versions
installed through the Windows Python launcher; replace `3.11` with any installed
version that is 3.10 or newer. If the `py` launcher is unavailable but
`python --version` reports 3.10 or newer, create the environment with
`python -m venv .venv` instead.

```powershell
py -0p
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks `Activate.ps1`, allow locally created scripts for the current
PowerShell process only, then activate the environment again:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1
```

`-Scope Process` does not require an administrator and expires when that PowerShell
window closes. You can also avoid activation and any execution-policy change by
calling the virtual environment's interpreter directly:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.gui.app
```

Smoke-check a fresh installation before configuring a provider (these commands work
whether or not the environment is activated):

```powershell
.\.venv\Scripts\python.exe -c "import src.gui.app; print('GUI import OK')"
.\.venv\Scripts\python.exe -m src.run_pipeline --help
```

The GUI import check also verifies that the multipart upload dependency is present.

## LLM provider configuration

The CLI reads the selected provider's API key from the environment. In PowerShell,
set one of these process-scoped variables in the same window that will run the CLI:

```powershell
$env:OPENAI_API_KEY = "your-key"
$env:GEMINI_API_KEY = "your-key"
$env:GROQ_API_KEY = "your-key"
```

Only the key for the provider you use is required. These values disappear when the
PowerShell window closes. In legacy Command Prompt (`cmd.exe`), the equivalent syntax
is:

```bat
set OPENAI_API_KEY=your-key
set GEMINI_API_KEY=your-key
set GROQ_API_KEY=your-key
```

The web GUI has its own **Set Key** field. GUI keys are kept in memory only and are
not read from the shell environment, persisted to disk, logged, or returned to the
browser.

| Provider | Default generation model | CLI environment variable |
|----------|--------------------------|--------------------------|
| `openai` | `gpt-3.5-turbo` | `OPENAI_API_KEY` |
| `gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY` |
| `groq` | `llama-3.1-8b-instant` | `GROQ_API_KEY` |

The review model defaults to the selected generation model. Models can be overridden
with `--model` and `--model-review`.

## Usage

### Web GUI

With the virtual environment activated:

```powershell
python -m src.gui.app
```

Open <http://127.0.0.1:8000>, select a provider, enter its key in **Set Key**, and
load or enter an analysis context. The API health check is available at
<http://127.0.0.1:8000/health> and the interactive API documentation at
<http://127.0.0.1:8000/docs>.

### CLI pipeline

The CLI accepts a YAML or JSON context file. A single context must define
`component`, `component_class`, `aspect`, `odd`, and `scenario`; a multi-component
file wraps those objects in a `components` list. See `src/examples/`.

```powershell
# Single-component example
python -m src.run_pipeline src/examples/cyclist.yaml --provider openai --outdir out

# Multi-component example with Gemini
python -m src.run_pipeline src/examples/shuttle_fleet.yaml --provider gemini --outdir out

# Restrict the catalogue guidewords used by the run
python -m src.run_pipeline src/examples/cyclist.yaml --guidewords no,less,more --outdir out
```

The default outputs are `out/hazop.html` and `out/hazop.csv`.

#### Guideword catalogue

By default, a run uses the eight guidewords marked **Yes** below to keep runtime and
LLM cost bounded. Pass `--guidewords all` to use the complete catalogue, or pass a
comma-separated list of IDs to select a custom subset.

| ID | Name | Short description | Default |
|----|------|-------------------|---------|
| `no` | No / Missing | Expected signal or output is absent. | **Yes** |
| `more` | More | Quantity, magnitude, confidence, or effect is too high. | **Yes** |
| `less` | Less | Quantity, magnitude, confidence, or effect is too low. | **Yes** |
| `wrong` | Wrong | A semantic, numeric, spatial, temporal, or logical value is incorrect. | **Yes** |
| `late` | Late | Output arrives after a safety-relevant deadline. | **Yes** |
| `early` | Early | Output or action occurs prematurely. | No |
| `intermittent` | Intermittent | Output alternates, flickers, disappears, and returns. | No |
| `frozen` | Frozen / Stale | An old output is reused as though it were current. | **Yes** |
| `corrupted` | Corrupted | Data is invalid, damaged, or cannot be interpreted safely. | No |
| `inverted` | Inverted | Meaning, sign, side, direction, or mapping is reversed. | No |
| `misordered` | Misordered | Sequence, lifecycle, or time alignment is incorrect. | No |
| `inconsistent` | Inconsistent | Output contradicts other data or physical plausibility. | **Yes** |
| `overgeneralized` | Overgeneralized | A learned pattern is applied too broadly. | No |
| `overfitted` | Overfitted | Benchmark behavior is good, but deployment generalization is poor. | No |
| `biased` | Biased | Performance differs across ODD slices, object types, locations, or groups. | No |
| `uncertain_but_confident` | Uncertain but confident | Confidence is high despite ambiguity or unknown input. | **Yes** |
| `certain_but_unstable` | Certain but unstable | Output changes even though reported confidence remains high. | No |
| `resource_limited` | Resource-limited | Runtime resource pressure degrades AI quality or timing. | No |
| `unmonitored` | Unmonitored | A failure is not detected by a safety mechanism. | No |
| `unsafe_fallback` | Unsafe fallback | The degradation response creates another hazard. | No |
| `valid_but_unsafe` | Valid-but-unsafe | A value passes interface checks but is semantically unsafe. | No |

### macOS/Linux setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
export OPENAI_API_KEY="your-key"
python -m src.run_pipeline src/examples/cyclist.yaml --provider openai --outdir out
```

## Architecture

The pipeline enriches each applicable worksheet row through eight phases:

1. **L1 - Failure modes:** one failure mode per configured guideword.
2. **L2 - Hazards and triage:** hazardous behavior, potential harm, and safety relevance.
3. **L3 - Initial risk:** ordinal risk factors and code-computed initial risk.
4. **L4 - Acceptance:** `ACCEPT`, `IMPROVE`, `RESTRICT`, or `INVESTIGATE` decision.
5. **L5 - Safety goals:** goals for rows that require further action.
6. **L6 - Measures:** respecifications, safety functions, and operational measures.
7. **L7 - Residual risk:** code-computed risk after the proposed measures.
8. **L8 - Evidence:** required evidence and open assumptions.

```text
[YAML/JSON context(s)] -> [ L1 -> L2 -> ... -> L8 ] -> [Holistic review] -> [HTML / CSV / GUI]
                                  ^                         |
                                  +---- repair/cascade -----+
                                     earliest faulty phase
```

Every L1-L8 phase uses the same validation loop:

```text
[Generate] -> [Validate] ---- pass ----> [Next phase]
                  ^  |
                  |  +---- issues ----> [Review] -> [Repair]
                  |                                  |
                  +----------------------------------+
```

Each phase follows a generate, validate, review, and repair cycle. A final holistic
review checks cross-phase consistency and can cascade a repair through downstream
phases.

| Module | Purpose |
|--------|---------|
| `src/graph_full.py` | L1-L8 LangGraph state machine and repair cascade |
| `src/run_pipeline.py` | CLI entry point and HTML/CSV worksheet writers |
| `src/ai_chains.py`, `src/chains.py` | Generation, review, and patch chains |
| `src/validators.py`, `src/models.py` | Phase validation and data models |
| `src/catalogue_loader.py`, `src/risk_model.py` | Methodology catalogues and risk calculation |
| `src/llm_client.py` | OpenAI-compatible provider client |
| `src/gui/` | FastAPI backend and browser interface |
