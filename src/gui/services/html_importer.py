"""HTML import — parses previously generated HAZOP reports back into row data.

Supports both pipeline HTML (7 columns) and GUI HTML (8 columns with rating).
"""
import re
import html as _html
from typing import Dict, List


def parse_hazop_html(html_content: str) -> List[Dict]:
    """Parse HAZOP HTML table and return list of row dicts.

    Supports two formats:
    - Pipeline HTML (7 columns): #, Function, Guideword, Deviation, Cause, Effect, Potentially dangerous
    - GUI HTML (8 columns): adds Rating column
    """
    tbody_m = re.search(r'<tbody>(.*?)</tbody>', html_content, re.DOTALL)
    if not tbody_m:
        raise ValueError("No <tbody> found in HTML")

    rows_html = re.findall(r'<tr>(.*?)</tr>', tbody_m.group(1), re.DOTALL)

    parsed = []
    for row_html in rows_html:
        # Skip separator rows
        if 'colspan' in row_html:
            continue

        cells = re.findall(r'<td[^>]*>(.*?)</td>', row_html, re.DOTALL)
        # Strip HTML tags, unescape entities
        texts = [_html.unescape(re.sub(r'<[^>]+>', '', c)).strip() for c in cells]

        if len(texts) < 7:
            continue  # skip malformed rows

        # Map columns: [#, function, guideword, deviation, cause, effect, dangerous, ?rating]
        dangerous_text = texts[6].lower()
        dangerous = 'dangerous' in dangerous_text and 'not' not in dangerous_text

        row = {
            "function": texts[1],
            "guideword": texts[2],
            "deviation": texts[3],
            "cause": texts[4],
            "effect": texts[5],
            "potentially_dangerous": dangerous,
        }

        # Optional rating column (GUI export has 8 columns)
        if len(texts) >= 8:
            rating_text = texts[7].strip().lower()
            rating_map = {"correct": "correct", "partial": "partially_correct",
                          "incorrect": "incorrect"}
            row["rating"] = rating_map.get(rating_text, "unrated")

        parsed.append(row)

    if not parsed:
        raise ValueError("No data rows found in HTML table")

    return parsed
