# src/run_pipeline.py
"""CLI entry point — loads input functions, runs the LangGraph pipeline, and
writes HTML/CSV output files.

Usage:
  python -m src.run_pipeline src/functions.txt --provider openai --outdir out
"""
from __future__ import annotations
import argparse
import logging
import os
import json
from typing import List, Any, Optional

from .llm_client import configure as configure_llm, PROVIDER_DEFAULTS
from .graph_full import build_full_graph, HazopGraphState
from .models import Guideword
from .row_utils import _suffix

logger = logging.getLogger(__name__)

try:
    import yaml
except Exception:
    yaml = None

# =========================
#        I/O helpers
# =========================

def load_functions(path: str) -> List[str]:
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    if ext in (".txt", ".list"):
        funcs = [ln.strip() for ln in data.splitlines() if ln.strip()]
        if not funcs:
            raise ValueError("No functions found")
        return funcs

    if ext in (".yaml", ".yml"):
        if not yaml:
            raise RuntimeError("pyyaml not installed")
        obj = yaml.safe_load(data)
        if isinstance(obj, dict) and isinstance(obj.get("functions"), list):
            return [str(x) for x in obj["functions"]]
        raise ValueError("YAML must contain key 'functions'")

    if ext == ".json":
        obj = json.loads(data)
        if isinstance(obj, dict) and isinstance(obj.get("functions"), list):
            return [str(x) for x in obj["functions"]]
        if isinstance(obj, list):
            return [str(x) for x in obj]
        raise ValueError("Bad JSON format")

    raise ValueError(f"Unsupported functions file: {ext}")

# ---------------------------------------------------------------------------
# Merge L1+L2+L3 rows into a single flat list for output.  Rows are joined
# by their numeric suffix (the part after the dash in "L1-5") because each
# stage uses its own prefix.
# ---------------------------------------------------------------------------

def merge_l1_l2_l3(rows_l1, rows_l2, rows_l3):
    by2 = {_suffix(r.get("row_id")): r for r in rows_l2 if isinstance(r, dict)}
    by3 = {_suffix(r.get("row_id")): r for r in rows_l3 if isinstance(r, dict)}

    def _prettify(text: Any, *, kind: str) -> str:
        """Make strings more readable and avoid colon-style labels like 'X: Y'.

        This is a display-only cleanup so your tables don't look like
        "label: value" fragments.
        """
        s = "" if text is None else str(text).strip()
        if not s:
            return ""

        # Turn "X: Y" into a single natural sentence (avoid ':').
        if ":" in s:
            left, right = [p.strip() for p in s.split(":", 1)]
            if left and right:
                if kind == "cause":
                    s = f"{left} that causes {right}"
                elif kind == "effect":
                    s = f"{left} which results in {right}"
                else:
                    s = f"{left} - {right}"

        # Safety net: remove any remaining colons.
        s = s.replace(":", " - ")

        # Ensure it ends like a sentence.
        if s and s[-1] not in ".!?":
            s = s + "."

        # Collapse whitespace.
        s = " ".join(s.split())
        return s

    out = []
    for r1 in rows_l1:
        sfx = _suffix(r1.get("row_id"))
        r2 = by2.get(sfx, {})
        r3 = by3.get(sfx, {})
        out.append({
            "row": sfx,
            "function": r1.get("function"),
            "guideword": r1.get("guideword"),
            "deviation": _prettify(r1.get("deviation"), kind="deviation"),
            "cause": _prettify(r2.get("cause") or r2.get("causes"), kind="cause"),
            "effect": _prettify(r3.get("effect") or r3.get("effects"), kind="effect"),
            # Phase 1: include the boolean triage indicator from L3 (default False)
            "potentially_dangerous": bool(r3.get("potentially_dangerous", False)),
        })
    return out

# =========================
# Output writers — generate HTML table and CSV file from merged rows.
# =========================

def write_html(rows, path):
    import html as _html
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def esc(x): return _html.escape("" if x is None else str(x))

    head = """<!doctype html><html><head><meta charset="utf-8">
<title>HAZOP Results</title>
<style>
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:20px;}
table{border-collapse:collapse;width:100%;}
th,td{border:1px solid #ddd;padding:8px;vertical-align:top;}
th{background:#f5f5f5;text-align:left;}
small{color:#666;}
code{background:#f0f0f0;padding:2px 4px;border-radius:4px;}
</style></head><body>
<h1>HAZOP Results</h1>
<p><small>Generated by L1→L2→L3 pipeline (full guideword coverage).</small></p>
<table>
<thead><tr>
<th>#</th>
<th>Function</th>
<th>Guideword</th>
<th>Deviation</th>
<th>Cause</th>
<th>Effect</th>
<th>Potentially dangerous</th>
</tr></thead><tbody>
"""
    body_lines = []
    prev_fn = None
    for i, r in enumerate(rows, 1):
        fn = r.get('function')
        # Insert a group separator when the function changes (except for the first row)
        if i > 1 and fn != prev_fn:
            body_lines.append(
                "<tr><td colspan='7' style='background:#ddd;height:4px;'></td></tr>"
            )
        prev_fn = fn
        dangerous = bool(r.get('potentially_dangerous'))
        status_text = "Dangerous" if dangerous else "Not dangerous"
        status_bg = "#fdd" if dangerous else "#dfd"
        body_lines.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td>{esc(fn)}</td>"
            f"<td><code>{esc(r.get('guideword'))}</code></td>"
            f"<td>{esc(r.get('deviation'))}</td>"
            f"<td>{esc(r.get('cause'))}</td>"
            f"<td>{esc(r.get('effect'))}</td>"
            f"<td style='background:{status_bg};text-align:center;'>{status_text}</td>"
            "</tr>"
        )
    tail = "</tbody></table></body></html>"
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(body_lines) + tail)

def write_csv(rows, path):
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Column names for CSV: last column reflects danger status as text instead of boolean
    cols = ["function", "guideword", "deviation", "cause", "effect", "Potentially dangerous"]

    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            dangerous = bool(r.get("potentially_dangerous", False))
            status_text = "Dangerous" if dangerous else "Not dangerous"
            w.writerow({
                "function": r.get("function", ""),
                "guideword": r.get("guideword", ""),
                "deviation": r.get("deviation", ""),
                "cause": r.get("cause", ""),
                "effect": r.get("effect", ""),
                "Potentially dangerous": status_text,
            })

# =========================
#    PIPELINE RUNNER
# =========================

def run(
    functions,
    notes,
    max_devs,
    html_out,
    csv_out,
    *,
    rag_enabled: bool = False,
    rag_paths: Optional[List[str]] = None,
    rag_embedder: str = "local",
    rag_min_sim: Optional[float] = None,
):
    guideword_map = {g.value: g.value for g in Guideword}

    state: HazopGraphState = {
        "functions": functions,
        "guideword_map": guideword_map,
        "max_devs_per_gw": max_devs,
        "notes": notes,
        # RAG is now handled inside LangGraph
        "rag_enabled": bool(rag_enabled),
        "rag_paths": list(rag_paths or []),
        "rag_embedder": (rag_embedder or "local").strip().lower(),
        "rag_min_sim": rag_min_sim if rag_min_sim is not None else 0.0,
    }

    logger.info("=== Starting HAZOP pipeline for %d functions ===", len(functions))
    graph = build_full_graph().compile()  # no CompiledGraph type needed
    out = graph.invoke(state)

    logger.debug("FINAL: L3 rows=%d", len(out.get('rows_l3') or []))
    logger.debug("FINAL: L3 missing_effect=%d", sum(1 for r in (out.get('rows_l3') or []) if not str(r.get('effect','')).strip()))

    final = merge_l1_l2_l3(out.get("rows_l1", []), out.get("rows_l2", []), out.get("rows_l3", []))
    logger.info("=== Pipeline complete: %d final rows ===", len(final))

    if html_out:
        write_html(final, html_out)
        logger.info("Wrote HTML output: %s", html_out)
    if csv_out:
        write_csv(final, csv_out)
        logger.info("Wrote CSV output: %s", csv_out)

    return {
        "l1_ok": out.get("l1_ok"),
        "l2_ok": out.get("l2_ok"),
        "l3_ok": out.get("l3_ok"),
        "holistic_ok": out.get("holistic_ok"),
        "rows": len(final),
        "out_html": html_out,
        "out_csv": csv_out,
    }

# =========================
# CLI argument parser and entry point.
# =========================

def main():
    ap = argparse.ArgumentParser(description="Run HAZOP LangGraph pipeline")

    ap.add_argument("functions", help="Path to functions file (txt/yaml/json)")
    ap.add_argument("--notes", default="", help="Notes passed to L1/L2/L3")
    ap.add_argument("--max_devs_per_gw", type=int, default=2, help="Max deviations per guideword (default: 2)")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--html", default="hazop.html", help="HTML filename ('' to disable)")
    ap.add_argument("--csv",  default="hazop.csv",  help="CSV filename ('' to disable)")

    # RAG arguments
    ap.add_argument("--use-rag", action="store_true", default=False,
                    help="Enable Retrieval-Augmented Generation (RAG) using provided files")
    ap.add_argument("--rag-files", default="", metavar="PATHS",
                    help=(
                        "Comma-separated list of file or directory paths used as the RAG knowledge base. "
                        "Supported: TXT/MD, CSV, XLSX/XLS, PDF. Directories are scanned recursively."
                    ))

    ap.add_argument("--rag-embedder", default="local", choices=["local", "openai", "gemini"],
                    help="Retrieval backend: 'local' (TF-IDF, offline), 'openai', or 'gemini' (semantic embeddings). Default: local")
    ap.add_argument("--rag-min-sim", type=float, default=None, metavar="FLOAT",
                    help="Minimum cosine similarity for RAG retrieval (default: 0.1 local, 0.25 openai/gemini)")

    # LLM provider arguments
    ap.add_argument("--provider", default="openai",
                    choices=list(PROVIDER_DEFAULTS.keys()),
                    help="LLM provider (default: openai)")
    ap.add_argument("--model", default=None,
                    help="Model for generation stages (provider-specific default if not set)")
    ap.add_argument("--model-review", default=None,
                    help="Model for holistic review (defaults to --model)")

    # Logging arguments
    ap.add_argument("-v", "--verbose", action="store_true",
                    help="Enable verbose (DEBUG) logging output")

    args = ap.parse_args()

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(name)s - %(levelname)s - %(message)s'
    )

    # Configure LLM provider before running pipeline
    configure_llm(
        provider=args.provider,
        model=args.model,
        model_review=args.model_review,
    )

    funcs = load_functions(args.functions)
    os.makedirs(args.outdir, exist_ok=True)
    html_path = os.path.join(args.outdir, args.html) if args.html else None
    csv_path  = os.path.join(args.outdir, args.csv)  if args.csv  else None

    final_notes = args.notes or ""
    rag_paths = [p.strip() for p in (args.rag_files or "").split(",") if p.strip()]

    summary = run(
        functions=funcs,
        notes=final_notes,
        max_devs=args.max_devs_per_gw,
        html_out=html_path,
        csv_out=csv_path,
        rag_enabled=bool(args.use_rag),
        rag_paths=rag_paths,
        rag_embedder=args.rag_embedder,
        rag_min_sim=args.rag_min_sim,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
