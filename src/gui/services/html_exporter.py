# src/gui/services/html_exporter.py
"""HTML report generation with danger highlighting and rating column.

Shows only final (post-edit/post-regeneration) values.  Dangerous rows are
highlighted in red; rating cells use green/yellow/red colour coding.
"""
from typing import List
import html

from .state_manager import state_manager, RowState


def export_to_html(run_id: str) -> str:
    """Generate HTML report with final data only + rating column."""
    run = state_manager.get_run(run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")

    rows = state_manager.get_all_rows(run_id)
    return _build_html(rows, run_id)


def _escape(text: str) -> str:
    """HTML-escape text."""
    return html.escape(text or "")


def _get_rating_display(rating) -> tuple:
    """Return (text, css_class) for rating display."""
    if rating is None:
        return "-", "rating-unrated"

    rating_value = rating.value if hasattr(rating, 'value') else str(rating)

    mapping = {
        "correct": ("Correct", "rating-correct"),
        "partially_correct": ("Partial", "rating-partial"),
        "incorrect": ("Incorrect", "rating-incorrect"),
        "unrated": ("-", "rating-unrated"),
    }
    return mapping.get(rating_value, ("-", "rating-unrated"))


def _build_html(rows: List[RowState], run_id: str) -> str:
    """Build styled HTML table."""

    # Build table rows
    table_rows = []
    last_function = None
    row_num = 0

    for row in rows:
        row_num += 1

        # Add separator row when function changes (except for first row)
        if last_function is not None and row.function != last_function:
            table_rows.append(
                "<tr><td colspan='8' style='background:#ddd;height:4px;'></td></tr>"
            )
        last_function = row.function

        # Dangerous cell styling
        if row.potentially_dangerous_final:
            danger_style = "background:#fdd;text-align:center;"
            danger_text = "Dangerous"
        else:
            danger_style = "background:#dfd;text-align:center;"
            danger_text = "Not dangerous"

        # Rating display
        rating_text, rating_class = _get_rating_display(row.rating)
        rating_style = {
            "rating-correct": "background:#dfd;text-align:center;",
            "rating-partial": "background:#ffd;text-align:center;",
            "rating-incorrect": "background:#fdd;text-align:center;",
            "rating-unrated": "color:#999;text-align:center;",
        }.get(rating_class, "color:#999;text-align:center;")

        table_rows.append(
            f"<tr>"
            f"<td>{row_num}</td>"
            f"<td>{_escape(row.function)}</td>"
            f"<td><code>{_escape(row.guideword)}</code></td>"
            f"<td>{_escape(row.deviation_final)}</td>"
            f"<td>{_escape(row.cause_final)}</td>"
            f"<td>{_escape(row.effect_final)}</td>"
            f"<td style='{danger_style}'>{danger_text}</td>"
            f"<td style='{rating_style}'>{rating_text}</td>"
            f"</tr>"
        )

    tbody = "\n".join(table_rows)

    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>HAZOP Results</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:20px;}}
table{{border-collapse:collapse;width:100%;}}
th,td{{border:1px solid #ddd;padding:8px;vertical-align:top;}}
th{{background:#f5f5f5;text-align:left;}}
small{{color:#666;}}
code{{background:#f0f0f0;padding:2px 4px;border-radius:4px;}}
</style></head><body>
<h1>HAZOP Results</h1>
<p><small>Run ID: {_escape(run_id)} | Exported from GUI</small></p>
<table>
<thead><tr>
<th>#</th>
<th>Function</th>
<th>Guideword</th>
<th>Deviation</th>
<th>Cause</th>
<th>Effect</th>
<th>Potentially dangerous</th>
<th>Rating</th>
</tr></thead><tbody>
{tbody}
</tbody></table></body></html>"""
