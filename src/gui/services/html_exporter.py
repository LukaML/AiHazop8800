# src/gui/services/html_exporter.py
"""HTML worksheet export (final values) with danger highlighting + rating column.

Columns mirror the CLI worksheet (run_pipeline._WORKSHEET_COLUMNS) so that GUI and
CLI HTML can be imported interchangeably; a Rating column is appended.
"""
from typing import Any, Dict, List
import html
import re

from .state_manager import state_manager, RowState, _stringify
from src.run_pipeline import _WORKSHEET_COLUMNS

_GOAL_TAG = re.compile(r'^\[?\s*SG\s*(\d+)\s*\]?\s*[-.:]?\s*', re.I)
_MEASURE_ID_RE = re.compile(r'^\((SF|R|P)\s*\d+\)\s*', re.I)


def export_to_html(run_id: str) -> str:
    run = state_manager.get_run(run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")
    return _build_html(state_manager.get_all_rows(run_id), run_id)


def _escape(text: Any) -> str:
    return html.escape("" if text is None else str(text))


# List-type worksheet columns rendered as labeled mini-lists (one item per line).
_LIST_PREFIX = {"ai_safety_goals": "SG", "evidence": "EV", "open_assumptions": "A"}
_LIST_KEYS = ("ai_safety_goals", "evidence", "open_assumptions", "measures")


def _list_items(final: Dict[str, Any], key: str) -> List[tuple]:
    items: List[tuple] = []

    def add(arr: Any, prefix: str) -> None:
        n = 0
        for x in (arr or []) if isinstance(arr, list) else ([arr] if arr else []):
            if isinstance(x, dict):
                t = " — ".join(str(v).strip() for v in x.values() if str(v).strip())
            else:
                t = str(x).strip()
            if t:
                n += 1
                items.append((f"{prefix}{n}", t))

    if key == "measures":
        add(final.get("respecifications"), "R")
        add(final.get("safety_functions"), "SF")
        add(final.get("passive_operational_measures"), "P")
    else:
        add(final.get(key), _LIST_PREFIX.get(key, "#"))
    return items


def _list_cell_html(final: Dict[str, Any], key: str) -> str:
    items = _list_items(final, key)
    if not items:
        return "<td style='color:#bbb;'>—</td>"
    inner = "".join(
        f"<div style='padding:3px 0;{'border-top:1px solid #eee;' if i else ''}'>"
        f"<b style='color:#555;margin-right:4px;'>{_escape(lab)}</b>{_escape(txt)}</div>"
        for i, (lab, txt) in enumerate(items)
    )
    return f"<td>{inner}</td>"


def _parse_goal_tag(x: Any) -> tuple:
    if isinstance(x, dict):
        s = " — ".join(str(v).strip() for v in x.values() if str(v).strip())
    else:
        s = str(x).strip()
    m = _GOAL_TAG.match(s)
    if m:
        return int(m.group(1) or 1), s[m.end():].strip()
    return 1, s


def _measure_groups(final: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = final.get("ai_safety_goals")
    goals = [g for g in raw if str(g).strip()] if isinstance(raw, list) else []
    ng = max(len(goals), 1)
    groups = [{"goal": f"SG{k}", "items": []} for k in range(1, ng + 1)]

    def add(arr: Any, cls: str) -> None:
        counter: Dict[int, int] = {}
        for x in (arr or []) if isinstance(arr, list) else []:
            g, text = _parse_goal_tag(x)
            if not text:
                continue
            g = min(max(g, 1), ng)
            m = _MEASURE_ID_RE.match(text)
            if m:
                label = re.sub(r"[()\s]", "", m.group(0)).upper()   # explicit id e.g. SF1
                text = text[m.end():].strip()
            else:
                counter[g] = counter.get(g, 0) + 1
                label = f"{cls}{g}.{counter[g]}"                     # derived fallback
            if not text:
                continue
            groups[g - 1]["items"].append((label, text))

    add(final.get("respecifications"), "R")
    add(final.get("safety_functions"), "SF")
    add(final.get("passive_operational_measures"), "P")
    return groups


def _measures_cell_html(final: Dict[str, Any]) -> str:
    # Only render goals that actually have measures (every goal is covered by the
    # L6 validator, so this never silently hides a gap in valid output and never
    # emits a 'no measure for this goal' placeholder).
    groups = [g for g in _measure_groups(final) if g["items"]]
    if not groups:
        return "<td style='color:#bbb;'>—</td>"
    blocks = []
    for i, g in enumerate(groups):
        head = f"<div style='font-weight:600;color:#555;'>{_escape(g['goal'])}</div>"
        body = "".join(
            f"<div style='padding-left:8px;'><b style='color:#777;margin-right:4px;'>{_escape(lab)}</b>{_escape(txt)}</div>"
            for lab, txt in g["items"]
        )
        sep = "border-top:1px solid #eee;" if i else ""
        blocks.append(f"<div style='padding:3px 0;{sep}'>{head}{body}</div>")
    return f"<td>{''.join(blocks)}</td>"


def _display(final: Dict[str, Any], key: str) -> str:
    if key == "guideword":
        return _stringify(final.get(key)).replace("_", " ")
    return _stringify(final.get(key))


def _rating_display(rating) -> tuple:
    value = rating.value if hasattr(rating, "value") else str(rating)
    return {
        "correct": ("Correct", "background:#dfd;text-align:center;"),
        "partially_correct": ("Partial", "background:#ffd;text-align:center;"),
        "incorrect": ("Incorrect", "background:#fdd;text-align:center;"),
    }.get(value, ("-", "color:#999;text-align:center;"))


def _cell_class(key: str, value: Any) -> str:
    v = str(value or "").strip().upper()
    if key in ("initial_risk", "residual_risk"):
        return "num"
    if key in ("risk_status", "residual_status"):
        return "status-ok" if v == "ACCEPTABLE" else ("status-above" if v else "")
    if key == "safety_decision":
        return "dec-accept" if v == "ACCEPT" else ("dec-act" if v else "")
    return ""


def _build_html(rows: List[RowState], run_id: str) -> str:
    ths = "".join(f"<th>{_escape(label)}</th>" for _, label in _WORKSHEET_COLUMNS)

    body = []
    for row in rows:
        final = row.final or {}
        classes = []
        if final.get("potentially_dangerous"):
            classes.append("dangerous")
        if getattr(row, "complete", True) is False:
            classes.append("incomplete")
        row_cls = f" class='{' '.join(classes)}'" if classes else ""

        tds = []
        for key, _ in _WORKSHEET_COLUMNS:
            if key == "measures":
                tds.append(_measures_cell_html(final))
            elif key in _LIST_KEYS:
                tds.append(_list_cell_html(final, key))
            elif key == "guideword":
                tds.append(f"<td><code>{_escape(_display(final, key))}</code></td>")
            else:
                cls = _cell_class(key, final.get(key))
                attr = f" class='{cls}'" if cls else ""
                tds.append(f"<td{attr}>{_escape(_display(final, key))}</td>")

        rating_text, rating_style = _rating_display(row.rating)
        body.append(
            f"<tr{row_cls}>"
            + "".join(tds)
            + f"<td style='{rating_style}'>{rating_text}</td>"
            "</tr>"
        )

    tbody = "\n".join(body)
    return f"""<!doctype html><html><head><meta charset="utf-8">
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
<p><small>Run ID: {_escape(run_id)} | Exported from GUI. Rows outlined amber are incomplete (failed final validation).</small></p>
<div class="wrap"><table>
<thead><tr>{ths}<th>Rating</th></tr></thead><tbody>
{tbody}
</tbody></table></div></body></html>"""
