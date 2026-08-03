# src/gui/services/html_importer.py
"""Import AI-HAZOP-8800 worksheets exported by the CLI or web GUI.

New GUI exports contain a versioned JSON payload with the complete row state.  The
visible table remains useful to people, while the payload makes export/import a
lossless round trip.  Older exports (including files created before the payload was
added) are parsed structurally with :class:`html.parser.HTMLParser`.
"""
from __future__ import annotations

import json
import math
import re
from html.parser import HTMLParser
from typing import Any, Dict, List, Optional, Tuple

from src.run_pipeline import _WORKSHEET_COLUMNS
from src.catalogue_loader import get_guidewords

# Label -> field key (built from the CLI worksheet column definitions).
_LABEL_TO_KEY = {label.strip().lower(): key for key, label in _WORKSHEET_COLUMNS}
_LABEL_TO_KEY.update({
    # Compatibility with the original L1-L3 HTML format.
    "function": "component",
    "guideword": "guideword",
    "deviation": "failure_mode",
    "cause": "hazardous_behavior",
    "effect": "potential_harm",
    "potentially dangerous": "potentially_dangerous",
})

_CTX_FIELDS = ("component", "component_class", "aspect", "odd", "scenario")
_NUMERIC_FIELDS = ("initial_risk", "residual_risk")
_EMPTY_MARKERS = {"", "-", "—", "–"}

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

_BLOCK_TAGS = {"div", "p", "li", "br"}
_MEASURE_RE = re.compile(
    r"^(?:\[\s*SG\s*(\d+)\s*\]\s*)?"
    r"\(?\s*(SF|R|P)\s*(\d+(?:\.\d+)?)\s*\)?\s*[-.:]?\s*(.*)$",
    re.I,
)
_GOAL_HEADING_RE = re.compile(r"^SG\s*(\d+)$", re.I)
_DISPLAY_LABEL_RE = {
    "ai_safety_goals": re.compile(r"^SG\s*\d+(?:\.\d+)?\s*", re.I),
    "evidence": re.compile(r"^EV\s*\d+(?:\.\d+)?\s*", re.I),
    "open_assumptions": re.compile(r"^A\s*\d+(?:\.\d+)?\s*", re.I),
}


def _attrs_dict(attrs: List[Tuple[str, Optional[str]]]) -> Dict[str, str]:
    return {str(k).lower(): "" if v is None else str(v) for k, v in attrs}


def _clean_lines(parts: List[str]) -> Tuple[str, List[str]]:
    raw = "".join(parts).replace("\r\n", "\n").replace("\r", "\n").replace("\xa0", " ")
    lines = []
    for part in raw.split("\n"):
        clean = re.sub(r"[\t\f\v ]+", " ", part).strip()
        if clean:
            lines.append(clean)
    return "\n".join(lines), lines


class _WorksheetHTMLParser(HTMLParser):
    """Collect table headers/rows and the optional lossless JSON payload."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.seen_tbody = False
        self.section: Optional[str] = None
        self.headers: List[Dict[str, Any]] = []
        self.rows: List[Dict[str, Any]] = []
        self._row: Optional[Dict[str, Any]] = None
        self._cell: Optional[Dict[str, Any]] = None
        self._cell_tag: Optional[str] = None
        self._in_payload = False
        self._payload_parts: List[str] = []
        self.payload_seen = False

    @property
    def payload_text(self) -> str:
        return "".join(self._payload_parts).strip()

    def _boundary(self) -> None:
        if not self._cell:
            return
        parts = self._cell["parts"]
        if parts and not str(parts[-1]).endswith("\n"):
            parts.append("\n")

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        attr_map = _attrs_dict(attrs)
        if tag == "script" and attr_map.get("id") == "ai-hazop-data":
            self.payload_seen = True
            self._in_payload = True
            return
        if tag == "thead":
            self.section = "head"
            return
        if tag == "tbody":
            self.seen_tbody = True
            self.section = "body"
            return
        if tag == "tr" and self.section in ("head", "body"):
            self._row = {"attrs": attr_map, "cells": []}
            return
        if tag in ("th", "td") and self._row is not None:
            self._cell_tag = tag
            self._cell = {"attrs": attr_map, "parts": []}
            return
        if self._cell is not None and tag in _BLOCK_TAGS:
            self._boundary()

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag == "script" and self._in_payload:
            self._in_payload = False
            return
        if tag in ("th", "td") and self._cell is not None and tag == self._cell_tag:
            text, lines = _clean_lines(self._cell["parts"])
            self._cell.update({"text": text, "lines": lines})
            if self._row is not None:
                self._row["cells"].append(self._cell)
            self._cell = None
            self._cell_tag = None
            return
        if self._cell is not None and tag in _BLOCK_TAGS:
            self._boundary()
            return
        if tag == "tr" and self._row is not None:
            if self.section == "head" and self._row["cells"] and not self.headers:
                self.headers = self._row["cells"]
            elif self.section == "body" and self._row["cells"]:
                self.rows.append(self._row)
            self._row = None
            return
        if tag in ("thead", "tbody"):
            self.section = None

    def handle_data(self, data: str) -> None:
        if self._in_payload:
            self._payload_parts.append(data)
        elif self._cell is not None:
            self._cell["parts"].append(data)


def _is_empty(text: Any) -> bool:
    return str(text or "").strip() in _EMPTY_MARKERS


def _split_list(text: str) -> List[str]:
    return [p.strip() for p in str(text or "").split(";") if not _is_empty(p)]


def _cell_items(cell: Dict[str, Any]) -> List[str]:
    lines = list(cell.get("lines") or [])
    if len(lines) <= 1:
        return _split_list(cell.get("text", ""))
    # Block markup already gives us exact item boundaries.  Do not split those
    # items again: natural-language goals/measures may legitimately contain ';'.
    return [line.strip() for line in lines if not _is_empty(line)]


def _parse_labeled_list(cell: Dict[str, Any], key: str) -> List[str]:
    pattern = _DISPLAY_LABEL_RE[key]
    out: List[str] = []
    for item in _cell_items(cell):
        text = pattern.sub("", item, count=1).strip()
        if not _is_empty(text):
            out.append(text)
    return out


def _parse_measures(cell: Dict[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {
        "respecifications": [],
        "safety_functions": [],
        "passive_operational_measures": [],
    }
    key_by_prefix = {
        "R": "respecifications",
        "SF": "safety_functions",
        "P": "passive_operational_measures",
    }
    current_goal = 1
    for item in _cell_items(cell):
        goal_m = _GOAL_HEADING_RE.fullmatch(item.strip())
        if goal_m:
            current_goal = int(goal_m.group(1))
            continue
        measure_m = _MEASURE_RE.match(item.strip())
        if measure_m:
            if measure_m.group(1):
                current_goal = int(measure_m.group(1))
            prefix = measure_m.group(2).upper()
            number = measure_m.group(3)
            text = measure_m.group(4).strip()
            if text:
                out[key_by_prefix[prefix]].append(
                    f"[SG{current_goal}] ({prefix}{number}) {text}"
                )
        elif not _is_empty(item):
            # CLI HTML historically flattened all measure types into one cell.
            out["respecifications"].append(item.strip())
    return out


def _guideword_aliases() -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for item in get_guidewords(enabled_only=False):
        gid = str(item.get("id", "")).strip()
        if not gid:
            continue
        for value in (gid, gid.replace("_", " "), item.get("name", "")):
            key = re.sub(r"[\s_-]+", " ", str(value or "").strip().lower())
            if key:
                aliases[key] = gid
    return aliases


def _normalise_guideword(text: str, aliases: Dict[str, str]) -> str:
    key = re.sub(r"[\s_-]+", " ", str(text or "").strip().lower())
    return aliases.get(key, str(text or "").strip())


def _parse_number(text: str) -> Any:
    if _is_empty(text):
        return ""
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return text
    if not math.isfinite(value):
        raise ValueError("Non-finite numeric value in HTML worksheet")
    return value


def _parse_rating(text: str) -> str:
    key = re.sub(r"[\s_-]+", " ", str(text or "").strip().lower())
    return {
        "correct": "correct",
        "partial": "partially_correct",
        "partially correct": "partially_correct",
        "incorrect": "incorrect",
    }.get(key, "unrated")


def _bool_attr(value: Optional[str]) -> Optional[bool]:
    if value is None:
        return None
    low = str(value).strip().lower()
    if low in ("true", "1", "yes"):
        return True
    if low in ("false", "0", "no"):
        return False
    return None


def _require_finite_numbers(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Embedded AI-HAZOP data contains a non-finite number")
    if isinstance(value, dict):
        for item in value.values():
            _require_finite_numbers(item)
    elif isinstance(value, list):
        for item in value:
            _require_finite_numbers(item)


def _rows_from_payload(payload: Any) -> List[Dict[str, Any]]:
    if not isinstance(payload, dict) or payload.get("schema") != "ai-hazop-8800":
        raise ValueError("Invalid embedded AI-HAZOP data payload")
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported embedded AI-HAZOP data version: {payload.get('version')}")
    _require_finite_numbers(payload)
    contexts = payload.get("contexts") or []
    if not isinstance(contexts, list):
        raise ValueError("Embedded AI-HAZOP contexts are invalid")
    items = payload.get("rows")
    if not isinstance(items, list):
        raise ValueError("Embedded AI-HAZOP data has no rows")

    rows: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("final"), dict):
            raise ValueError("Embedded AI-HAZOP row is invalid")
        row = dict(item["final"])
        component_index = item.get("component_index")
        if isinstance(component_index, int) and component_index >= 0:
            row["_component_index"] = component_index
            if component_index < len(contexts):
                ctx = contexts[component_index]
                if isinstance(ctx, dict):
                    row.update({f: str(ctx.get(f, row.get(f, ""))) for f in _CTX_FIELDS})
                    row["_import_context"] = {f: str(ctx.get(f, "")) for f in _CTX_FIELDS}
        row["_rating"] = _parse_rating(str(item.get("rating", "unrated")))
        complete = item.get("complete", True)
        if not isinstance(complete, bool):
            complete = _bool_attr(str(complete))
            if complete is None:
                raise ValueError("Embedded AI-HAZOP row completeness is invalid")
        row["_complete"] = complete
        if item.get("display_id") and not row.get("hazard_id"):
            row["hazard_id"] = str(item["display_id"])
        rows.append(row)
    if not rows:
        raise ValueError("No data rows found in embedded AI-HAZOP payload")
    return rows


def parse_hazop_html(html_content: str) -> List[Dict[str, Any]]:
    """Parse an exported HAZOP worksheet into field dictionaries."""
    parser = _WorksheetHTMLParser()
    parser.feed(html_content)
    parser.close()

    if parser.payload_seen:
        try:
            payload = json.loads(parser.payload_text)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid embedded AI-HAZOP data: {exc}") from exc
        return _rows_from_payload(payload)

    if not parser.seen_tbody:
        raise ValueError("No <tbody> found in HTML")

    col_keys: List[str] = []
    for cell in parser.headers:
        explicit = str(cell.get("attrs", {}).get("data-field", "")).strip()
        label = str(cell.get("text", "")).strip()
        low = label.lower()
        if explicit:
            col_keys.append(explicit)
        elif low in ("#", ""):
            col_keys.append("_index")
        elif low == "dangerous":
            col_keys.append("potentially_dangerous")
        elif low == "rating":
            col_keys.append("rating")
        else:
            col_keys.append(_LABEL_TO_KEY.get(low, "_skip"))
    full_worksheet = {
        "safety_decision", "residual_risk", "evidence", "open_assumptions",
    }.issubset(col_keys)

    aliases = _guideword_aliases()
    parsed: List[Dict[str, Any]] = []
    for dom_row in parser.rows:
        cells = dom_row.get("cells") or []
        if any("colspan" in (cell.get("attrs") or {}) for cell in cells):
            continue

        row_attrs = dom_row.get("attrs") or {}
        classes = set(str(row_attrs.get("class", "")).lower().split())
        explicit_danger = _bool_attr(row_attrs.get("data-dangerous"))
        explicit_complete = _bool_attr(row_attrs.get("data-complete"))
        if explicit_complete is not None:
            imported_complete: Optional[bool] = explicit_complete
        elif "incomplete" in classes:
            imported_complete = False
        elif full_worksheet:
            imported_complete = True
        else:
            # Old L1-L3 tables did not carry completeness metadata.  Let the
            # StateManager derive validity instead of claiming they are complete.
            imported_complete = None
        row: Dict[str, Any] = {
            "potentially_dangerous": explicit_danger if explicit_danger is not None else "dangerous" in classes,
            "_complete": imported_complete,
            "_rating": "unrated",
        }
        explicit_component_index = row_attrs.get("data-component-index")
        if explicit_component_index is not None:
            try:
                component_index = int(explicit_component_index)
            except (TypeError, ValueError):
                component_index = -1
            if component_index >= 0:
                row["_component_index"] = component_index

        for i, cell in enumerate(cells):
            key = col_keys[i] if i < len(col_keys) else "_skip"
            if key in ("_skip", "_index"):
                continue
            text = str((cell.get("attrs") or {}).get("data-value", cell.get("text", ""))).strip()
            if key == "potentially_dangerous":
                low = text.lower()
                row[key] = ("yes" in low or "dangerous" in low) and "not" not in low
            elif key == "rating":
                row["_rating"] = _parse_rating(text)
            elif key == "measures":
                row.update(_parse_measures(cell))
            elif key in ("ai_safety_goals", "evidence", "open_assumptions"):
                row[key] = _parse_labeled_list(cell, key)
            elif key in ("respecifications", "safety_functions", "passive_operational_measures"):
                row[key] = _cell_items(cell)
            elif key in _NUMERIC_FIELDS:
                row[key] = _parse_number(text)
            elif key == "guideword":
                row[key] = _normalise_guideword(text, aliases)
            else:
                row[key] = "" if _is_empty(text) else text

        if any(row.get(k) for k in ("component", "guideword", "failure_mode", "hazardous_behavior")):
            parsed.append(row)

    if not parsed:
        raise ValueError("No data rows found in HTML table")
    return parsed


def build_states_from_rows(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Group imported rows and synthesise per-component LangGraph-style states."""
    groups: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    contexts_by_key: Dict[Tuple[Any, ...], Dict[str, str]] = {}
    order: List[Tuple[Any, ...]] = []

    for row in rows:
        imported_ctx = row.get("_import_context")
        if isinstance(imported_ctx, dict):
            ctx = {f: str(imported_ctx.get(f, row.get(f, ""))).strip() for f in _CTX_FIELDS}
        else:
            ctx = {f: str(row.get(f, "")).strip() for f in _CTX_FIELDS}
        component_index = row.get("_component_index")
        if isinstance(component_index, int):
            key: Tuple[Any, ...] = ("index", component_index)
        else:
            key = ("context",) + tuple(ctx[f] for f in _CTX_FIELDS)
        if key not in groups:
            groups[key] = []
            contexts_by_key[key] = ctx
            order.append(key)
        groups[key].append(row)

    guidewords = get_guidewords(enabled_only=False)
    contexts: List[Dict[str, str]] = []
    states: List[Dict[str, Any]] = []

    for key in order:
        group = groups[key]
        ctx = contexts_by_key[key]
        contexts.append(ctx)
        state: Dict[str, Any] = {
            "ctx": ctx,
            "guidewords": guidewords,
            "class_questions": [],
            "aspect_hint": "",
            "notes": "",
            "_import_meta": {},
        }
        for phase, fields in _PHASE_FIELDS.items():
            phase_rows = []
            for i, row in enumerate(group):
                phase_row: Dict[str, Any] = {"row_id": f"{phase.upper()}-{i}"}
                for context_field in _CTX_FIELDS:
                    phase_row[context_field] = ctx[context_field]
                for field_key in fields:
                    if field_key in row:
                        phase_row[field_key] = row[field_key]
                phase_rows.append(phase_row)
                if phase == "l1":
                    state["_import_meta"][str(i)] = {
                        "rating": row.get("_rating", "unrated"),
                        "complete": row.get("_complete"),
                    }
            state[f"rows_{phase}"] = phase_rows
        states.append(state)

    return contexts, states
