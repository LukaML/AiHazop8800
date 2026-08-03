# src/gui/services/state_manager.py
"""In-memory state tracker for AI-HAZOP-8800 analysis runs.

Each run holds the full per-component LangGraph output states (so rows can be
regenerated with cascade) plus one ``RowState`` per worksheet row. A RowState
tracks the row's ``original`` (LLM) and ``final`` (post edit/regeneration) field
maps — generic dicts spanning all L1–L8 fields, so the same structure works for
every phase. Human edits are written back into the authoritative component state
so downstream regeneration sees them.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import uuid
import logging
from copy import deepcopy

from src.row_utils import _suffix, _scrub_meta
from src.risk_model import compute_risk, FACTORS, acceptance_criterion_text
from src.validators import is_exportable_row

from ..models import RowData, Rating, AnalysisStatus

logger = logging.getLogger(__name__)

# Phases in order; the per-component state stores rows under rows_l1 .. rows_l8.
_PHASES = ("l1", "l2", "l3", "l4", "l5", "l6", "l7", "l8")

# Which phase each editable field belongs to (writes go to that phase's rows).
FIELD_PHASE: Dict[str, str] = {
    "failure_mode": "l1",
    "hazardous_behavior": "l2",
    "potential_harm": "l2",
    "potentially_dangerous": "l2",
    "E": "l3", "PF": "l3", "PND": "l3", "PNM": "l3", "S": "l3",
    "risk_rationale": "l3",
    "safety_decision": "l4",
    "acceptance_rationale": "l4",
    "ai_safety_goals": "l5",
    "respecifications": "l6",
    "safety_functions": "l6",
    "passive_operational_measures": "l6",
    "residual_E": "l7", "residual_PF": "l7", "residual_PND": "l7",
    "residual_PNM": "l7", "residual_S": "l7", "residual_rationale": "l7",
    "evidence": "l8",
    "open_assumptions": "l8",
}

EDITABLE_FIELDS = tuple(FIELD_PHASE.keys())

# Fields whose values are lists (coerced from text on the way in).
LIST_FIELDS = (
    "ai_safety_goals", "respecifications", "safety_functions",
    "passive_operational_measures", "evidence", "open_assumptions",
)

_RESIDUAL_FACTORS = tuple(f"residual_{f}" for f in FACTORS)

# Read-only export column order (flat, research-grade).
_EXPORT_FIELDS = (
    "component", "component_class", "aspect", "odd", "scenario", "guideword",
    "failure_mode", "hazardous_behavior", "potential_harm", "potentially_dangerous",
    "E", "PF", "PND", "PNM", "S", "initial_risk", "risk_status", "risk_rationale",
    "safety_decision", "acceptance_rationale", "ai_safety_goals",
    "respecifications", "safety_functions", "passive_operational_measures",
    "residual_E", "residual_PF", "residual_PND", "residual_PNM", "residual_S",
    "residual_risk", "residual_status", "residual_rationale",
    "evidence", "open_assumptions",
)


def _item_to_str(x: Any) -> str:
    """Flatten a list item (str or dict) to readable text."""
    if isinstance(x, dict):
        return " — ".join(str(v).strip() for v in x.values() if str(v).strip())
    if isinstance(x, list):
        return "; ".join(str(i).strip() for i in x if str(i).strip())
    return str(x).strip()


def _stringify(v: Any) -> str:
    if isinstance(v, list):
        return "; ".join(s for s in (_item_to_str(x) for x in v) if s)
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if isinstance(v, float):
        return f"{v:.2e}"
    return "" if v is None else str(v)


def merge_state_rows(state: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Merge a component's L1–L8 rows (by row_id suffix) into one dict per L1 row.

    Keeps lists as lists and code-computed risk numeric. Non-dangerous / skipped
    rows simply lack the downstream fields.
    """
    l1 = state.get("rows_l1") or []
    idx = {
        ph: {_suffix(str(r.get("row_id"))): r for r in (state.get(f"rows_{ph}") or []) if isinstance(r, dict)}
        for ph in _PHASES[1:]
    }
    out: List[Dict[str, Any]] = []
    for r1 in l1:
        if not isinstance(r1, dict):
            continue
        sfx = _suffix(str(r1.get("row_id")))
        merged = dict(r1)
        for ph in _PHASES[1:]:
            src = idx[ph].get(sfx)
            if src:
                for k, v in src.items():
                    if k != "row_id":
                        merged[k] = v
        # Final safety net: never surface reviewer/validator meta text in the GUI.
        merged = {k: _scrub_meta(v) for k, v in merged.items()}
        out.append(merged)
    return out


@dataclass
class RowState:
    """State for one worksheet row: original vs final field maps + tracking."""
    row_id: str                 # canonical, GUI-facing (e.g. "c0__L1-3")
    display_id: str             # clean, human-facing (e.g. "L1-3")
    component_index: int
    component: str
    guideword: str
    original: Dict[str, Any] = field(default_factory=dict)
    final: Dict[str, Any] = field(default_factory=dict)
    rating: Rating = Rating.UNRATED
    edited_flag: bool = False
    regenerated_flag: bool = False
    complete: bool = True

    @property
    def dangerous_final(self) -> bool:
        return bool(self.final.get("potentially_dangerous", False))

    def to_row_data(self) -> RowData:
        return RowData(
            row_id=self.row_id,
            display_id=self.display_id,
            component_index=self.component_index,
            component=self.component,
            guideword=self.guideword,
            original=self.original,
            final=self.final,
            dangerous_final=self.dangerous_final,
            complete=self.complete,
            rating=self.rating,
            edited_flag=self.edited_flag,
            regenerated_flag=self.regenerated_flag,
        )

    def to_dict_for_export(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {"row_id": self.row_id, "component_index": self.component_index}
        for f in _EXPORT_FIELDS:
            out[f"{f}_original"] = _stringify(self.original.get(f))
            out[f"{f}_final"] = _stringify(self.final.get(f))
        out["rating"] = self.rating.value if self.rating else "unrated"
        out["edited_flag"] = self.edited_flag
        out["regenerated_flag"] = self.regenerated_flag
        return out


@dataclass
class AnalysisRun:
    """State for a single analysis run."""
    run_id: str
    provider: str
    model: str
    notes: str
    contexts: List[Dict[str, str]] = field(default_factory=list)

    status: AnalysisStatus = AnalysisStatus.PENDING
    progress: Optional[str] = None
    error: Optional[str] = None

    rows: Dict[str, RowState] = field(default_factory=dict)
    # Full LangGraph output state per component (index aligns with `contexts`).
    states: List[Dict[str, Any]] = field(default_factory=list)

    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None


def _canonical_id(component_index: int, row_id: str) -> str:
    return f"c{component_index}__{row_id}"


def _imported_rating(meta: Dict[str, Any]) -> Rating:
    """Coerce optional HTML-import metadata without breaking normal runs."""
    value = meta.get("rating", Rating.UNRATED)
    if isinstance(value, Rating):
        return value
    try:
        return Rating(str(value))
    except (TypeError, ValueError):
        return Rating.UNRATED


def _imported_complete(meta: Dict[str, Any], final: Dict[str, Any]) -> bool:
    """Prefer an imported completeness flag; otherwise derive it as before."""
    if "complete" in meta:
        value = meta.get("complete")
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalised = value.strip().lower()
            if normalised in ("true", "1", "yes"):
                return True
            if normalised in ("false", "0", "no"):
                return False
    return is_exportable_row(final)


def _apply_derived(final: Dict[str, Any], display_id: str) -> None:
    """Stamp paper display-only fields (Hazard ID, Acceptance criterion) onto a row's map."""
    final["hazard_id"] = display_id
    final["acceptance_criterion"] = (
        acceptance_criterion_text() if bool(final.get("potentially_dangerous")) else ""
    )


class StateManager:
    """Manages analysis runs and row states in memory."""

    def __init__(self):
        self._runs: Dict[str, AnalysisRun] = {}

    def create_run(self, provider: str, model: str, contexts: List[Dict[str, str]], notes: str) -> str:
        run_id = str(uuid.uuid4())[:8]
        self._runs[run_id] = AnalysisRun(
            run_id=run_id, provider=provider, model=model, notes=notes,
            contexts=deepcopy(contexts),
        )
        logger.info("Created analysis run: %s", run_id)
        return run_id

    def get_run(self, run_id: str) -> Optional[AnalysisRun]:
        return self._runs.get(run_id)

    def update_status(self, run_id: str, status: AnalysisStatus,
                      progress: Optional[str] = None, error: Optional[str] = None) -> None:
        run = self._runs.get(run_id)
        if run:
            run.status = status
            run.progress = progress
            run.error = error
            if status == AnalysisStatus.COMPLETED:
                run.completed_at = datetime.utcnow()
            logger.info("Run %s status: %s", run_id, status)

    # --- row initialisation -------------------------------------------------

    def set_states(self, run_id: str, states: List[Dict[str, Any]]) -> None:
        """Store the per-component LangGraph output states and (re)build rows."""
        run = self._runs.get(run_id)
        if not run:
            logger.error("Run not found: %s", run_id)
            return
        run.states = [deepcopy(s) for s in states]
        run.rows.clear()
        for ci, state in enumerate(run.states):
            # Coverage: keep one row per configured guideword (do not omit). Invalid L1
            # rows are still blocked from downstream phases in graph_full._gen_l2.
            for ordinal, merged in enumerate(merge_state_rows(state), start=1):
                import_meta = state.get("_import_meta") or {}
                row_meta = import_meta.get(_suffix(str(merged.get("row_id"))), {})
                if not isinstance(row_meta, dict):
                    row_meta = {}
                cid = _canonical_id(ci, str(merged.get("row_id")))
                # Display id: leading number = component (1-based), trailing = row in it.
                # Component 1 -> L1-1..L1-n, component 2 -> L2-1..L2-n, etc.
                display_id = f"L{ci + 1}-{ordinal}"
                original = deepcopy(merged)
                final = deepcopy(merged)
                _apply_derived(original, display_id)
                _apply_derived(final, display_id)
                run.rows[cid] = RowState(
                    row_id=cid,
                    display_id=display_id,
                    component_index=ci,
                    component=str(merged.get("component", "")),
                    guideword=str(merged.get("guideword", "")),
                    original=original,
                    final=final,
                    rating=_imported_rating(row_meta),
                    complete=_imported_complete(row_meta, final),
                )
        logger.info("Initialized %d rows for run %s", len(run.rows), run_id)

    # --- editing ------------------------------------------------------------

    def get_row(self, run_id: str, row_id: str) -> Optional[RowState]:
        run = self._runs.get(run_id)
        return run.rows.get(row_id) if run else None

    def _state_row(self, run: AnalysisRun, ci: int, phase: str, sfx: str) -> Optional[Dict[str, Any]]:
        if ci >= len(run.states):
            return None
        rows = run.states[ci].get(f"rows_{phase}") or []
        for r in rows:
            if isinstance(r, dict) and _suffix(str(r.get("row_id"))) == sfx:
                return r
        return None

    def update_row_fields(self, run_id: str, row_id: str, fields: Dict[str, Any]) -> Optional[RowState]:
        """Write human edits into the authoritative component state and refresh the row."""
        run = self._runs.get(run_id)
        if not run:
            return None
        row = run.rows.get(row_id)
        if not row:
            return None

        ci = row.component_index
        sfx = _suffix(row_id)
        changed = False
        touched_phases = set()

        for key, value in (fields or {}).items():
            if key not in FIELD_PHASE:
                continue
            phase = FIELD_PHASE[key]
            state_row = self._state_row(run, ci, phase, sfx)
            if state_row is None:
                continue
            if key in LIST_FIELDS:
                value = _coerce_list(value)
            elif key == "potentially_dangerous":
                value = bool(value)
            state_row[key] = value
            changed = True
            touched_phases.add(phase)

        if changed:
            self._recompute_risk(run, ci, sfx, touched_phases)
            row.final = _rebuild_final(run.states[ci], sfx)
            _apply_derived(row.final, row.display_id)
            row.complete = is_exportable_row(row.final)
            row.edited_flag = True
            logger.info("Row %s edited in run %s", row_id, run_id)
        return row

    def _recompute_risk(self, run: AnalysisRun, ci: int, sfx: str, phases: set) -> None:
        if "l3" in phases:
            r3 = self._state_row(run, ci, "l3", sfx)
            if r3 is not None:
                risk, status, _ = compute_risk(r3)
                if risk is not None:
                    r3["initial_risk"] = risk
                r3["risk_status"] = status
        if "l7" in phases:
            r7 = self._state_row(run, ci, "l7", sfx)
            if r7 is not None:
                risk, status, _ = compute_risk({f: r7.get(f"residual_{f}") for f in FACTORS})
                if risk is not None:
                    r7["residual_risk"] = risk
                r7["residual_status"] = status

    def update_row_rating(self, run_id: str, row_id: str, rating: Rating) -> Optional[RowState]:
        row = self.get_row(run_id, row_id)
        if not row:
            return None
        row.rating = rating
        logger.info("Row %s rated as %s in run %s", row_id, rating, run_id)
        return row

    # --- regeneration -------------------------------------------------------

    def refresh_rows_from_state(self, run_id: str, row_ids: List[str]) -> List[RowState]:
        """Rebuild ``final`` for the given rows from the (post-cascade) component state."""
        run = self._runs.get(run_id)
        if not run:
            return []
        updated: List[RowState] = []
        for rid in row_ids:
            row = run.rows.get(rid)
            if not row:
                continue
            row.final = _rebuild_final(run.states[row.component_index], _suffix(rid))
            _apply_derived(row.final, row.display_id)
            row.complete = is_exportable_row(row.final)
            row.regenerated_flag = True
            updated.append(row)
        return updated

    def get_all_rows(self, run_id: str) -> List[RowState]:
        run = self._runs.get(run_id)
        if not run:
            return []

        def sort_key(r: RowState) -> Tuple[int, int]:
            try:
                return (r.component_index, int(_suffix(r.row_id)))
            except (ValueError, AttributeError):
                return (r.component_index, 0)

        return sorted(run.rows.values(), key=sort_key)


def _coerce_list(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    s = str(value or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.split(";")]
    return [p for p in parts if p]


def _rebuild_final(state: Dict[str, Any], sfx: str) -> Dict[str, Any]:
    for merged in merge_state_rows(state):
        if _suffix(str(merged.get("row_id"))) == sfx:
            return deepcopy(merged)
    return {}


# Global state manager instance
state_manager = StateManager()
