# src/run_pipeline.py
"""CLI entry point for the AI-HAZOP-8800 pipeline.

Loads an AI component analysis context (component, class, aspect, ODD, scenario),
runs the LangGraph phase engine (L1 Failure Mode → L2 Hazard → L3 Initial Risk →
L4 Acceptance → L5 Safety Goals → L6 Measures → L7 Residual Risk → L8 Evidence,
followed by a holistic cross-phase review), and writes the worksheet as HTML and CSV.

Usage:
  python -m src.run_pipeline src/examples/cyclist.yaml --provider gemini --outdir out
  python -m src.run_pipeline ctx.yaml --guidewords no,less,wrong --provider groq
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional

from .llm_client import configure as configure_llm, PROVIDER_DEFAULTS
from .graph_full import build_full_graph
from .catalogue_loader import (
    get_guidewords,
    get_class_questions,
    get_aspect_hint,
    load_catalogue,
)
from .row_utils import _suffix, _is_meta_text, _scrub_meta
from .risk_model import acceptance_criterion_text
from .validators import is_exportable_row, is_safety_relevant, incomplete_reason

logger = logging.getLogger(__name__)

try:
    import yaml
except Exception:
    yaml = None

_CTX_FIELDS = ("component", "component_class", "aspect", "odd", "scenario")


# =========================
#        Input loading
# =========================

def load_contexts(path: str) -> List[Dict[str, str]]:
    """Load one or more analysis contexts from a YAML/JSON file.

    Accepts either a single context object with the five context fields, or an
    object with a ``components`` list of such objects.
    """
    ext = os.path.splitext(path)[1].lower()
    with open(path, "r", encoding="utf-8") as f:
        data = f.read()

    if ext in (".yaml", ".yml"):
        if not yaml:
            raise RuntimeError("pyyaml not installed")
        obj = yaml.safe_load(data)
    elif ext == ".json":
        obj = json.loads(data)
    else:
        raise ValueError(f"Unsupported context file: {ext} (use .yaml or .json)")

    if isinstance(obj, dict) and isinstance(obj.get("components"), list):
        items = obj["components"]
    elif isinstance(obj, list):
        items = obj
    elif isinstance(obj, dict):
        items = [obj]
    else:
        raise ValueError("Context file must be an object or a list of objects")

    contexts: List[Dict[str, str]] = []
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            raise ValueError(f"Context #{i} is not an object")
        missing = [f for f in _CTX_FIELDS if not str(it.get(f, "")).strip()]
        if missing:
            raise ValueError(f"Context #{i} missing required fields: {missing}")
        contexts.append({f: str(it[f]).strip() for f in _CTX_FIELDS})
    return contexts


# =========================
#     Worksheet assembly
# =========================

def assemble_worksheet(state_out: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge L1–L8 rows (by row_id suffix) into worksheet rows.

    Every L1 row becomes a worksheet row. Rows that were filtered out as
    not-dangerous after L2 simply have empty L3–L8 fields.
    """
    def _idx(key):
        return {_suffix(r.get("row_id")): r for r in (state_out.get(key) or []) if isinstance(r, dict)}

    rows_l1 = state_out.get("rows_l1") or []
    by2, by3, by4 = _idx("rows_l2"), _idx("rows_l3"), _idx("rows_l4")
    by5, by6, by7, by8 = _idx("rows_l5"), _idx("rows_l6"), _idx("rows_l7"), _idx("rows_l8")

    def _join(v):
        if isinstance(v, list):
            return "; ".join(s for s in (str(x).strip() for x in v) if s and not _is_meta_text(s))
        s = str(v or "")
        return "" if _is_meta_text(s) else s

    def _measures(r6):
        parts = []
        for f in ("respecifications", "safety_functions", "passive_operational_measures"):
            parts.extend(str(x).strip() for x in (r6.get(f) or []) if str(x).strip())
        return "; ".join(parts)

    out: List[Dict[str, Any]] = []
    for r1 in rows_l1:
        if not isinstance(r1, dict):
            continue
        sfx = _suffix(r1.get("row_id"))
        r2, r3, r4 = by2.get(sfx, {}), by3.get(sfx, {}), by4.get(sfx, {})
        r5, r6, r7, r8 = by5.get(sfx, {}), by6.get(sfx, {}), by7.get(sfx, {}), by8.get(sfx, {})
        # Coverage: every configured guideword (every L1 row) is kept so the worksheet
        # has exactly one row per guideword. Invalid L1 rows are still blocked from
        # downstream phases (see graph_full._gen_l2); meta text is scrubbed below.
        ir = r3.get("initial_risk")
        rr = r7.get("residual_risk")
        merged_raw: Dict[str, Any] = {}
        for src in (r1, r2, r3, r4, r5, r6, r7, r8):
            if isinstance(src, dict):
                merged_raw.update(src)
        # Safety relevance drives which downstream columns are populated. is_safety_relevant honours
        # the L2 triage flag and also catches VRU/collision rows (architecture §6).
        dangerous = is_safety_relevant(merged_raw)
        out.append({
            "hazard_id": r1.get("row_id", ""),
            "component": r1.get("component", ""),
            "component_class": r1.get("component_class", ""),
            "aspect": r1.get("aspect", ""),
            "odd": r1.get("odd", ""),
            "scenario": r1.get("scenario", ""),
            "guideword": r1.get("guideword", ""),
            "failure_mode": _scrub_meta(r1.get("failure_mode", "")),
            "hazardous_behavior": _scrub_meta(r2.get("hazardous_behavior", "")),
            "potential_harm": _scrub_meta(r2.get("potential_harm", "")),
            "potentially_dangerous": dangerous,
            "initial_risk": ("%.2e" % ir) if isinstance(ir, (int, float)) else "",
            "risk_status": r3.get("risk_status", "") if dangerous else "",
            "acceptance_criterion": acceptance_criterion_text() if dangerous else "",
            "safety_decision": r4.get("safety_decision", "") if dangerous else "",
            "ai_safety_goals": _join(r5.get("ai_safety_goals")),
            "measures": _measures(r6),
            "residual_risk": ("%.2e" % rr) if isinstance(rr, (int, float)) else "",
            "residual_status": r7.get("residual_status", ""),
            "evidence": _join(r8.get("evidence")),
            "open_assumptions": _join(r8.get("open_assumptions")),
            # Internal status (PART 9) — outside the paper columns; marks but never omits.
            "_complete": is_exportable_row(merged_raw),
            "_incomplete_reason": incomplete_reason(merged_raw),
        })
    if logger.isEnabledFor(logging.INFO):
        for r in out:
            if not r.get("_complete", True):
                logger.info("INCOMPLETE row %s (guideword=%s): %s",
                            r.get("hazard_id", "?"), r.get("guideword", "?"),
                            r.get("_incomplete_reason") or "unknown")
    return out


# =========================
#        Output writers
# =========================

# Columns mirror the AI-HAZOP-8800 paper §12 template (plus the useful Risk/Residual
# Status columns). No "Dangerous" column — danger is an internal flag only.
_WORKSHEET_COLUMNS = [
    ("hazard_id", "Hazard ID"),
    ("component", "Component"),
    ("component_class", "Class"),
    ("aspect", "Aspect"),
    ("odd", "ODD"),
    ("scenario", "Scenario"),
    ("guideword", "Guideword/Question"),
    ("failure_mode", "Failure mode"),
    ("hazardous_behavior", "Hazardous behavior"),
    ("potential_harm", "Potential harm"),
    ("initial_risk", "Initial risk"),
    ("risk_status", "Risk Status"),
    ("acceptance_criterion", "Acceptance criterion"),
    ("safety_decision", "Safety decision"),
    ("ai_safety_goals", "AI Safety Goal"),
    ("measures", "Measures"),
    ("residual_risk", "Residual risk"),
    ("residual_status", "Residual Status"),
    ("evidence", "Evidence"),
    ("open_assumptions", "Open assumptions"),
]


def write_html(rows: List[Dict[str, Any]], path: str, acceptance: str = "") -> None:
    import html as _html
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def esc(x):
        return _html.escape("" if x is None else str(x))

    def _cls(key, val):
        v = str(val or "").strip().upper()
        if key in ("initial_risk", "residual_risk"):
            return "num"
        if key in ("risk_status", "residual_status"):
            return "status-ok" if v == "ACCEPTABLE" else ("status-above" if v else "")
        if key == "safety_decision":
            return "dec-accept" if v == "ACCEPT" else ("dec-act" if v else "")
        return ""

    ths = "".join(f"<th>{esc(label)}</th>" for _, label in _WORKSHEET_COLUMNS)
    head = f"""<!doctype html><html><head><meta charset="utf-8">
<title>AI-HAZOP-8800 Worksheet</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:16px;color:#1f2933;}}
.wrap{{overflow-x:auto;border:1px solid #e2e8f0;border-radius:8px;margin-top:10px;}}
table{{border-collapse:collapse;width:100%;font-size:12px;}}
th,td{{border:1px solid #e2e8f0;padding:6px 8px;vertical-align:top;text-align:left;max-width:300px;overflow-wrap:anywhere;}}
thead th{{position:sticky;top:0;background:#1e3a8a;color:#fff;font-weight:600;}}
tbody tr:nth-child(even){{background:#f8fafc;}}
tr.dangerous{{background:#fff5f5;}}
tr.incomplete{{outline:2px solid #f59e0b;}}
td.num{{font-family:ui-monospace,Menlo,Consolas,monospace;white-space:nowrap;text-align:right;}}
.status-ok{{color:#067647;font-weight:600;}}.status-above{{color:#b42318;font-weight:600;}}
.dec-accept{{color:#067647;font-weight:600;}}.dec-act{{color:#b54708;font-weight:600;}}
small{{color:#64748b;}} code{{background:#eef2ff;padding:2px 6px;border-radius:6px;font-weight:600;}}
</style></head><body>
<h1>AI-HAZOP-8800 Worksheet</h1>
<p><small>Generated by the L1→L8 AI-HAZOP-8800 pipeline. Acceptance criterion: {esc(acceptance)}. Rows outlined amber are incomplete (failed final validation).</small></p>
<div class="wrap"><table>
<thead><tr>{ths}</tr></thead><tbody>
"""
    body = []
    for r in rows:
        classes = []
        if r.get("potentially_dangerous"):
            classes.append("dangerous")
        reason = ""
        if not r.get("_complete", True):
            classes.append("incomplete")
            reason = r.get("_incomplete_reason") or "incomplete (failed final validation)"
        row_cls = f" class='{' '.join(classes)}'" if classes else ""
        title_attr = f" title='{esc(reason)}'" if reason else ""
        tds = []
        for key, _ in _WORKSHEET_COLUMNS:
            val = r.get(key, "")
            cls = _cls(key, val)
            attr = f" class='{cls}'" if cls else ""
            if key == "guideword":
                tds.append(f"<td{attr}><code>{esc(val)}</code></td>")
            else:
                tds.append(f"<td{attr}>{esc(val)}</td>")
        body.append(f"<tr{row_cls}{title_attr}>" + "".join(tds) + "</tr>")
    tail = "</tbody></table></div></body></html>"
    with open(path, "w", encoding="utf-8") as f:
        f.write(head + "\n".join(body) + tail)


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    import csv
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    # Paper columns + trailing internal "Complete"/"Reason" status (outside the paper worksheet).
    cols = [label for _, label in _WORKSHEET_COLUMNS] + ["Complete", "Reason"]
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {label: r.get(key, "") for key, label in _WORKSHEET_COLUMNS}
            complete = r.get("_complete", True)
            out["Complete"] = "Yes" if complete else "No"
            out["Reason"] = "" if complete else (r.get("_incomplete_reason") or "incomplete")
            w.writerow(out)


# =========================
#        Pipeline runner
# =========================

def _build_state(ctx: Dict[str, str], notes: str, guideword_override: Optional[List[str]]) -> Dict[str, Any]:
    guidewords = get_guidewords(override=guideword_override)
    return {
        "ctx": ctx,
        "guidewords": guidewords,
        "class_questions": get_class_questions(ctx["component_class"]),
        "aspect_hint": get_aspect_hint(ctx["aspect"]),
        "notes": notes or "",
    }


def run(
    contexts: List[Dict[str, str]],
    notes: str,
    html_out: Optional[str],
    csv_out: Optional[str],
    guideword_override: Optional[List[str]] = None,
) -> Dict[str, Any]:
    graph = build_full_graph().compile()
    acc = load_catalogue("acceptance_criterion")
    acceptance_label = f"{acc.get('mode', 'MEM')} target R ≤ {acc.get('mem_target')}"

    all_rows: List[Dict[str, Any]] = []
    ok_keys = [f"l{i}_ok" for i in range(1, 9)] + ["holistic_ok"]
    phase_ok = {kk: True for kk in ok_keys}

    for ci, ctx in enumerate(contexts):
        logger.info("=== Component %d/%d: %s (%s) ===", ci + 1, len(contexts), ctx["component"], ctx["component_class"])
        state = _build_state(ctx, notes, guideword_override)
        out = graph.invoke(state)
        for kk in phase_ok:
            phase_ok[kk] = phase_ok[kk] and bool(out.get(kk, True))
        all_rows.extend(assemble_worksheet(out))

    logger.info("=== Pipeline complete: %d worksheet rows ===", len(all_rows))

    if html_out:
        write_html(all_rows, html_out, acceptance=acceptance_label)
        logger.info("Wrote HTML output: %s", html_out)
    if csv_out:
        write_csv(all_rows, csv_out)
        logger.info("Wrote CSV output: %s", csv_out)

    dangerous = sum(1 for r in all_rows if r.get("potentially_dangerous"))
    return {
        **phase_ok,
        "components": len(contexts),
        "rows": len(all_rows),
        "dangerous_rows": dangerous,
        "out_html": html_out,
        "out_csv": csv_out,
    }


# =========================
#        CLI
# =========================

def main():
    ap = argparse.ArgumentParser(description="Run the AI-HAZOP-8800 LangGraph pipeline")
    ap.add_argument("context", help="Path to analysis context file (.yaml/.json) with component/class/aspect/odd/scenario")
    ap.add_argument("--notes", default="", help="Extra context passed to all phases")
    ap.add_argument("--guidewords", default="", help="Comma-separated guideword ids to use, or 'all' (default: catalogue's enabled set)")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument("--html", default="hazop.html", help="HTML filename ('' to disable)")
    ap.add_argument("--csv", default="hazop.csv", help="CSV filename ('' to disable)")

    ap.add_argument("--provider", default="openai", choices=list(PROVIDER_DEFAULTS.keys()),
                    help="LLM provider (default: openai)")
    ap.add_argument("--model", default=None, help="Model for generation stages (provider default if unset)")
    ap.add_argument("--model-review", default=None, help="Model for review (defaults to --model)")

    ap.add_argument("-v", "--verbose", action="store_true", help="Enable DEBUG logging")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(name)s - %(levelname)s - %(message)s",
    )

    configure_llm(provider=args.provider, model=args.model, model_review=args.model_review)

    contexts = load_contexts(args.context)
    os.makedirs(args.outdir, exist_ok=True)
    html_path = os.path.join(args.outdir, args.html) if args.html else None
    csv_path = os.path.join(args.outdir, args.csv) if args.csv else None

    guideword_override = [s.strip() for s in args.guidewords.split(",") if s.strip()] or None

    summary = run(
        contexts=contexts,
        notes=args.notes or "",
        html_out=html_path,
        csv_out=csv_path,
        guideword_override=guideword_override,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
