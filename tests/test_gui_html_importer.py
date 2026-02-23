# tests/test_gui_html_importer.py
"""Tests for HTML import/parsing — supports both pipeline and GUI HTML formats."""
import pytest

from src.gui.services.html_importer import parse_hazop_html


# ---------------------------------------------------------------------------
# Sample HTML fragments
# ---------------------------------------------------------------------------

PIPELINE_HTML_7_COL = """<!doctype html><html><head></head><body>
<table>
<thead><tr><th>#</th><th>Function</th><th>Guideword</th><th>Deviation</th>
<th>Cause</th><th>Effect</th><th>Potentially dangerous</th></tr></thead>
<tbody>
<tr><td>1</td><td>Braking</td><td><code>no</code></td>
<td>No braking action.</td><td>Hydraulic leak.</td>
<td>Vehicle cannot stop.</td><td style='background:#fdd;text-align:center;'>Dangerous</td></tr>
<tr><td>2</td><td>Braking</td><td><code>more</code></td>
<td>Excessive braking.</td><td>Software fault.</td>
<td>Wheels lock up.</td><td style='background:#dfd;text-align:center;'>Not dangerous</td></tr>
</tbody></table></body></html>"""

GUI_HTML_8_COL = """<!doctype html><html><head></head><body>
<table>
<thead><tr><th>#</th><th>Function</th><th>Guideword</th><th>Deviation</th>
<th>Cause</th><th>Effect</th><th>Potentially dangerous</th><th>Rating</th></tr></thead>
<tbody>
<tr><td>1</td><td>Steering</td><td><code>reverse</code></td>
<td>Reverse steering.</td><td>Actuator fault.</td>
<td>Loss of control.</td><td style='background:#fdd;'>Dangerous</td>
<td style='background:#dfd;'>Correct</td></tr>
<tr><td>2</td><td>Steering</td><td><code>late</code></td>
<td>Late response.</td><td>Sensor lag.</td>
<td>Delayed action.</td><td style='background:#dfd;'>Not dangerous</td>
<td style='background:#ffd;'>Partial</td></tr>
</tbody></table></body></html>"""

HTML_WITH_SEPARATOR = """<html><body><table><thead><tr>
<th>#</th><th>Function</th><th>Guideword</th><th>Deviation</th>
<th>Cause</th><th>Effect</th><th>Dangerous</th></tr></thead>
<tbody>
<tr><td>1</td><td>F1</td><td>no</td><td>D1</td><td>C1</td><td>E1</td><td>Not dangerous</td></tr>
<tr><td colspan='7' style='background:#ddd;height:4px;'></td></tr>
<tr><td>2</td><td>F2</td><td>more</td><td>D2</td><td>C2</td><td>E2</td><td>Dangerous</td></tr>
</tbody></table></body></html>"""

HTML_WITH_ENTITIES = """<html><body><table><thead><tr>
<th>#</th><th>Function</th><th>Guideword</th><th>Deviation</th>
<th>Cause</th><th>Effect</th><th>Dangerous</th></tr></thead>
<tbody>
<tr><td>1</td><td>Heat &amp; Cool</td><td>no</td><td>No &amp; none</td>
<td>Cause &lt;1&gt;</td><td>Effect &quot;bad&quot;</td><td>Not dangerous</td></tr>
</tbody></table></body></html>"""


class TestParsePipelineHtml:
    def test_parses_7_column_format(self):
        rows = parse_hazop_html(PIPELINE_HTML_7_COL)
        assert len(rows) == 2
        assert rows[0]["function"] == "Braking"
        assert rows[0]["guideword"] == "no"
        assert rows[0]["deviation"] == "No braking action."
        assert rows[0]["cause"] == "Hydraulic leak."
        assert rows[0]["effect"] == "Vehicle cannot stop."

    def test_dangerous_column_parsing(self):
        rows = parse_hazop_html(PIPELINE_HTML_7_COL)
        assert rows[0]["potentially_dangerous"] is True
        assert rows[1]["potentially_dangerous"] is False


class TestParseGuiHtml:
    def test_parses_8_column_format(self):
        rows = parse_hazop_html(GUI_HTML_8_COL)
        assert len(rows) == 2
        assert rows[0]["function"] == "Steering"

    def test_rating_column_mapping(self):
        rows = parse_hazop_html(GUI_HTML_8_COL)
        assert rows[0]["rating"] == "correct"
        assert rows[1]["rating"] == "partially_correct"


class TestSeparatorRows:
    def test_separator_rows_skipped(self):
        rows = parse_hazop_html(HTML_WITH_SEPARATOR)
        assert len(rows) == 2
        assert rows[0]["function"] == "F1"
        assert rows[1]["function"] == "F2"


class TestHtmlEntities:
    def test_entities_unescaped(self):
        rows = parse_hazop_html(HTML_WITH_ENTITIES)
        assert rows[0]["function"] == "Heat & Cool"
        assert rows[0]["deviation"] == "No & none"
        assert rows[0]["cause"] == "Cause <1>"
        assert rows[0]["effect"] == 'Effect "bad"'


class TestInvalidHtml:
    def test_no_tbody_raises(self):
        with pytest.raises(ValueError, match="No <tbody>"):
            parse_hazop_html("<html><body><p>No table here</p></body></html>")

    def test_empty_table_raises(self):
        html = "<html><body><table><tbody></tbody></table></body></html>"
        with pytest.raises(ValueError, match="No data rows"):
            parse_hazop_html(html)
