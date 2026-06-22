# src/gui/services/html_importer.py
"""HTML import — parse a previously exported AI-HAZOP-8800 worksheet back into rows.

Parsing is header-driven (maps <th> labels to field keys), so it tolerates the wide
L1–L8 worksheet from either the CLI or the GUI. ``build_states_from_rows`` then
synthesises per-component LangGraph-style states (one row per phase) so imported
rows can be displayed, edited, and regenerated like a normal run.
"""
import re
import html as _html
from typing import Any, Dict, List, Tuple

from src.run_pipeline import _WORKSHEET_COLUMNS
from src.catalogue_loader import get_guidewords

# Label -> field key (built from the CLI worksheet column definitions).
_LABEL_TO_KEY = {label.strip().lower(): key for key, label in _WORKSHEET_COLUMNS}

_LIST_FIELDS = (
    "ai_safety_goals", "measures", "evidence", "open_assumptions",
    "respecifications", "safety_functions", "passive_operational_measures",
)
_CTX_FIELDS = ("component", "component_class", "aspect", "odd", "scenario")

# Fields grouped by phase for state synthesis.
_PHASE_FIELDS = {
    "l1": ("guideword", "failure_mode"),
    "l2": ("hazardous_behavior", "potential_harm", "potentially_dangerous"),
    "l3": ("E", "PF", "PND", "PNM", "S", "initial_risk", "risk_status", "risk_rationale"),
    "l4": ("safety_decision", "acceptance_rationale"),
    "l5": ("ai_safety_goals",),
    "l6": ("respecifications", "safety_functions", "passive_operational_measures"),
    "l7": ("residual_E", "residual_PF", "residual_PND", "residual_PNM", "residual_S",
           "residual_risk", "residual_status", "residual_rationale"),
    "l8": ("evidence", "open_assumptions"),
}


def _split_list(text: str) -> List[str]:
    return [p.strip() for p in str(text or "").split(";") if p.strip()]


def parse_hazop_html(html_content: str) -> List[Dict[str, Any]]:
    """Parse a HAZOP worksheet table into a list of field dicts (header-driven)."""
    thead_m = re.search(r"<thead>(.*?)</thead>", html_content, re.DOTALL)
    tbody_m = re.search(r"<tbody>(.*?)</tbody>", html_content, re.DOTALL)
    if not tbody_m:
        raise ValueError("No <tbody> found in HTML")

    # Build the column index -> field-key map from the header labels.
    labels: List[str] = []
    if thead_m:
        header_cells = re.findall(r"<th[^>]*>(.*?)</th>", thead_m.group(1), re.DOTALL)
        labels = [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in header_cells]

    col_keys: List[str] = []
    for label in labels:
        low = label.lower()
        if low in ("#", ""):
            col_keys.append("_index")
        elif low == "dangerous":
            col_keys.append("potentially_dangerous")
        elif low == "rating":
            col_keys.append("rating")
        else:
            col_keys.append(_LABEL_TO_KEY.get(low, "_skip"))

    rows_html = re.findall(r"<tr>(.*?)</tr>", tbody_m.group(1), re.DOTALL)
    parsed: List[Dict[str, Any]] = []
    for row_html in rows_html:
        if "colspan" in row_html:
            continue
        cells = re.findall(r"<td[^>]*>(.*?)</td>", row_html, re.DOTALL)
        texts = [_html.unescape(re.sub(r"<[^>]+>", "", c)).strip() for c in cells]
        if not texts:
            continue

        row: Dict[str, Any] = {}
        for i, text in enumerate(texts):
            key = col_keys[i] if i < len(col_keys) else "_skip"
            if key in ("_skip", "_index"):
                continue
            if key == "potentially_dangerous":
                low = text.lower()
                row[key] = ("yes" in low or "dangerous" in low) and "not" not in low
            elif key == "rating":
                row["rating"] = {"correct": "correct", "partial": "partially_correct",
                                 "incorrect": "incorrect"}.get(text.lower(), "unrated")
            elif key == "measures":
                # Combined measures column -> recover individual items as respecifications.
                row["respecifications"] = _split_list(text)
            elif key in _LIST_FIELDS:
                row[key] = _split_list(text)
            else:
                row[key] = text
        if any(row.get(k) for k in ("failure_mode", "hazardous_behavior")) or row:
            parsed.append(row)

    if not parsed:
        raise ValueError("No data rows found in HTML table")
    return parsed


def build_states_from_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Group parsed rows by component and synthesise per-component pipeline states."""
    groups: Dict[str, List[Dict[str, Any]]] = {}
    order: List[str] = []
    for r in rows:
        comp = str(r.get("component", "")).strip()
        if comp not in groups:
            groups[comp] = []
            order.append(comp)
        groups[comp].append(r)

    guidewords = get_guidewords(enabled_only=False)
    contexts: List[Dict[str, str]] = []
    states: List[Dict[str, Any]] = []

    for comp in order:
        grp = groups[comp]
        first = grp[0]
        ctx = {f: str(first.get(f, "")).strip() for f in _CTX_FIELDS}
        contexts.append(ctx)

        state: Dict[str, Any] = {
            "ctx": ctx,
            "guidewords": guidewords,
            "class_questions": [],
            "aspect_hint": "",
            "notes": "",
        }
        for ph, fields in _PHASE_FIELDS.items():
            phase_rows = []
            for i, r in enumerate(grp):
                pr: Dict[str, Any] = {"row_id": f"{ph.upper()}-{i}"}
                for cf in _CTX_FIELDS:
                    pr[cf] = ctx[cf]
                for fkey in fields:
                    if fkey in r:
                        pr[fkey] = r[fkey]
                phase_rows.append(pr)
            state[f"rows_{ph}"] = phase_rows
        states.append(state)

    return contexts, states
