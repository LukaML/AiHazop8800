# src/gui/services/state_manager.py
"""In-memory state tracker for analysis runs — stores rows, assigns canonical IDs,
and tracks original vs final (edited/regenerated) values for research-grade export.

Each row has an original value (from the LLM) and a final value (after human
edits or LLM regeneration).  The human-override pattern applies to all editable
fields: human edits take precedence over AI output.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid
import logging
from copy import deepcopy

from ..models import RowData, Rating, AnalysisStatus

logger = logging.getLogger(__name__)


def _prettify(text: Any, *, kind: str) -> str:
    """
    Make strings more readable and avoid colon-style labels like 'X: Y'.
    Converts "Label: description" into natural sentences.
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


@dataclass
class RowState:
    """Internal state for a single HAZOP row, tracking original vs final values."""
    row_id: str
    function: str
    guideword: str

    deviation_original: str
    deviation_final: str

    cause_original: str
    cause_final: str

    effect_original: str
    effect_final: str

    potentially_dangerous_ai: bool
    potentially_dangerous_human: Optional[bool] = None

    rating: Rating = Rating.UNRATED
    edited_flag: bool = False
    regenerated_flag: bool = False

    @property
    def potentially_dangerous_final(self) -> bool:
        """Human override takes precedence over AI."""
        if self.potentially_dangerous_human is not None:
            return self.potentially_dangerous_human
        return self.potentially_dangerous_ai

    def to_row_data(self) -> RowData:
        """Convert to API response model."""
        return RowData(
            row_id=self.row_id,
            function=self.function,
            guideword=self.guideword,
            deviation_original=self.deviation_original,
            deviation_final=self.deviation_final,
            cause_original=self.cause_original,
            cause_final=self.cause_final,
            effect_original=self.effect_original,
            effect_final=self.effect_final,
            potentially_dangerous_ai=self.potentially_dangerous_ai,
            potentially_dangerous_human=self.potentially_dangerous_human,
            potentially_dangerous_final=self.potentially_dangerous_final,
            rating=self.rating,
            edited_flag=self.edited_flag,
            regenerated_flag=self.regenerated_flag,
        )

    def to_dict_for_export(self) -> Dict[str, Any]:
        """Convert to dict for CSV export."""
        return {
            "row_id": self.row_id,
            "function": self.function,
            "guideword": self.guideword,
            "deviation_original": self.deviation_original,
            "deviation_final": self.deviation_final,
            "cause_original": self.cause_original,
            "cause_final": self.cause_final,
            "effect_original": self.effect_original,
            "effect_final": self.effect_final,
            "potentially_dangerous_ai": self.potentially_dangerous_ai,
            "potentially_dangerous_human": self.potentially_dangerous_human,
            "potentially_dangerous_final": self.potentially_dangerous_final,
            "rating": self.rating.value if self.rating else "unrated",
            "edited_flag": self.edited_flag,
            "regenerated_flag": self.regenerated_flag,
        }


@dataclass
class AnalysisRun:
    """State for a single analysis run."""
    run_id: str
    provider: str
    model: str
    rag_enabled: bool
    rag_embedder: Optional[str]
    functions: List[str]
    notes: str
    max_devs_per_gw: int

    status: AnalysisStatus = AnalysisStatus.PENDING
    progress: Optional[str] = None
    error: Optional[str] = None

    rows: Dict[str, RowState] = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # RAG persistence
    rag_paths: List[str] = field(default_factory=list)
    _rag_notes_l1: str = ""
    _rag_notes_l2: str = ""
    _rag_notes_l3: str = ""

    # Raw pipeline output for regeneration
    _raw_l1: List[Dict[str, Any]] = field(default_factory=list)
    _raw_l2: List[Dict[str, Any]] = field(default_factory=list)
    _raw_l3: List[Dict[str, Any]] = field(default_factory=list)


class StateManager:
    """
    Manages analysis runs and row states in memory.
    Thread-safe for concurrent access.
    """

    def __init__(self):
        self._runs: Dict[str, AnalysisRun] = {}

    def create_run(
        self,
        provider: str,
        model: str,
        functions: List[str],
        notes: str,
        max_devs_per_gw: int,
        rag_enabled: bool = False,
        rag_embedder: Optional[str] = None,
        rag_paths: Optional[List[str]] = None,
    ) -> str:
        """Create a new analysis run and return its ID."""
        run_id = str(uuid.uuid4())[:8]
        run = AnalysisRun(
            run_id=run_id,
            provider=provider,
            model=model,
            rag_enabled=rag_enabled,
            rag_embedder=rag_embedder,
            functions=functions,
            notes=notes,
            max_devs_per_gw=max_devs_per_gw,
            rag_paths=rag_paths or [],
        )
        self._runs[run_id] = run
        logger.info("Created analysis run: %s", run_id)
        return run_id

    def get_run(self, run_id: str) -> Optional[AnalysisRun]:
        """Get an analysis run by ID."""
        return self._runs.get(run_id)

    def update_status(
        self,
        run_id: str,
        status: AnalysisStatus,
        progress: Optional[str] = None,
        error: Optional[str] = None,
    ) -> None:
        """Update run status."""
        run = self._runs.get(run_id)
        if run:
            run.status = status
            run.progress = progress
            run.error = error
            if status == AnalysisStatus.COMPLETED:
                run.completed_at = datetime.utcnow()
            logger.info("Run %s status: %s", run_id, status)

    def initialize_rows_from_pipeline(
        self,
        run_id: str,
        rows_l1: List[Dict[str, Any]],
        rows_l2: List[Dict[str, Any]],
        rows_l3: List[Dict[str, Any]],
    ) -> None:
        """
        Initialize row states from pipeline output.
        Maps L1/L2/L3 rows by suffix to create unified RowState objects.
        """
        run = self._runs.get(run_id)
        if not run:
            logger.error("Run not found: %s", run_id)
            return

        # Store raw rows for regeneration
        run._raw_l1 = deepcopy(rows_l1)
        run._raw_l2 = deepcopy(rows_l2)
        run._raw_l3 = deepcopy(rows_l3)

        # Build lookup by suffix
        def suffix(row_id: str) -> str:
            if "-" in str(row_id):
                return str(row_id).split("-", 1)[1]
            return str(row_id)

        l2_by_suffix = {suffix(r.get("row_id", "")): r for r in rows_l2}
        l3_by_suffix = {suffix(r.get("row_id", "")): r for r in rows_l3}

        run.rows.clear()

        # Canonical ID assignment — use the L3 row_id (or L1 if L3 is missing)
        # as the GUI-facing row identifier.
        for r1 in rows_l1:
            sfx = suffix(r1.get("row_id", ""))
            r2 = l2_by_suffix.get(sfx, {})
            r3 = l3_by_suffix.get(sfx, {})

            # Use the L3 row_id as canonical (or L1 if L3 missing).
            # Explicit None checks avoid falsy-int pitfall (row_id=0 is falsy).
            raw_id = r3.get("row_id")
            if raw_id is None or raw_id == "":
                raw_id = r1.get("row_id")
            if raw_id is None or raw_id == "":
                raw_id = f"L1-{sfx}"
            row_id = str(raw_id)

            deviation = _prettify(r1.get("deviation", ""), kind="deviation")
            cause = _prettify(r2.get("cause") or r2.get("causes", ""), kind="cause")
            effect = _prettify(r3.get("effect") or r3.get("effects", ""), kind="effect")
            dangerous_ai = bool(r3.get("potentially_dangerous", False))

            row_state = RowState(
                row_id=row_id,
                function=str(r1.get("function", "")).strip(),
                guideword=str(r1.get("guideword", "")).strip(),
                deviation_original=deviation,
                deviation_final=deviation,
                cause_original=cause,
                cause_final=cause,
                effect_original=effect,
                effect_final=effect,
                potentially_dangerous_ai=dangerous_ai,
            )
            run.rows[row_id] = row_state

        logger.info("Initialized %d rows for run %s", len(run.rows), run_id)

    def initialize_rows_from_import(self, run_id: str, rows: List[Dict]) -> None:
        """Initialize row states from imported HTML data."""
        run = self._runs.get(run_id)
        if not run:
            return

        run.rows.clear()
        raw_l1, raw_l2, raw_l3 = [], [], []

        for i, r in enumerate(rows):
            row_id = f"L3-{i}"
            deviation = r.get("deviation", "")
            cause = r.get("cause", "")
            effect = r.get("effect", "")
            dangerous = bool(r.get("potentially_dangerous", False))
            rating_str = r.get("rating", "unrated")
            try:
                rating = Rating(rating_str)
            except ValueError:
                rating = Rating.UNRATED

            row_state = RowState(
                row_id=row_id,
                function=r.get("function", "").strip(),
                guideword=r.get("guideword", "").strip(),
                deviation_original=deviation,
                deviation_final=deviation,
                cause_original=cause,
                cause_final=cause,
                effect_original=effect,
                effect_final=effect,
                potentially_dangerous_ai=dangerous,
                rating=rating,
            )
            run.rows[row_id] = row_state

            # Synthesize raw data for regeneration support
            base = {"row_id": f"L1-{i}", "function": r.get("function", ""),
                    "guideword": r.get("guideword", ""), "deviation": deviation}
            raw_l1.append(dict(base))
            raw_l2.append({**base, "row_id": f"L2-{i}", "cause": cause})
            raw_l3.append({**base, "row_id": f"L3-{i}", "cause": cause,
                           "effect": effect, "potentially_dangerous": dangerous})

        run._raw_l1 = raw_l1
        run._raw_l2 = raw_l2
        run._raw_l3 = raw_l3
        run.status = AnalysisStatus.COMPLETED
        run.completed_at = datetime.utcnow()
        logger.info("Imported %d rows for run %s", len(run.rows), run_id)

    def get_row(self, run_id: str, row_id: str) -> Optional[RowState]:
        """Get a specific row from a run."""
        run = self._runs.get(run_id)
        if run:
            return run.rows.get(row_id)
        return None

    def update_row_fields(
        self,
        run_id: str,
        row_id: str,
        deviation: Optional[str] = None,
        cause: Optional[str] = None,
        effect: Optional[str] = None,
        potentially_dangerous: Optional[bool] = None,
    ) -> Optional[RowState]:
        """Update editable fields on a row."""
        row = self.get_row(run_id, row_id)
        if not row:
            return None

        changed = False

        if deviation is not None and deviation != row.deviation_final:
            row.deviation_final = deviation
            changed = True

        if cause is not None and cause != row.cause_final:
            row.cause_final = cause
            changed = True

        if effect is not None and effect != row.effect_final:
            row.effect_final = effect
            changed = True

        if potentially_dangerous is not None:
            row.potentially_dangerous_human = potentially_dangerous
            changed = True

        if changed:
            row.edited_flag = True
            logger.info("Row %s edited in run %s", row_id, run_id)

        return row

    def update_row_rating(
        self,
        run_id: str,
        row_id: str,
        rating: Rating,
    ) -> Optional[RowState]:
        """Update rating on a row."""
        row = self.get_row(run_id, row_id)
        if not row:
            return None

        row.rating = rating
        logger.info("Row %s rated as %s in run %s", row_id, rating, run_id)
        return row

    # --- Row update/merge after regeneration ---

    def update_rows_after_regeneration(
        self,
        run_id: str,
        regenerated_rows: List[Dict[str, Any]],
        row_ids: List[str],
    ) -> List[RowState]:
        """
        Update rows after regeneration.
        Only updates the specified row_ids, marks them as regenerated.
        """
        run = self._runs.get(run_id)
        if not run:
            return []

        updated = []
        regen_by_id = {r.get("row_id"): r for r in regenerated_rows}

        for row_id in row_ids:
            existing = run.rows.get(row_id)
            if not existing:
                continue

            regen = regen_by_id.get(row_id, {})
            if regen:
                # Update final values from regeneration (apply prettify to remove colons)
                if "deviation" in regen:
                    existing.deviation_final = _prettify(regen["deviation"], kind="deviation")
                if "cause" in regen or "causes" in regen:
                    existing.cause_final = _prettify(regen.get("cause") or regen.get("causes", ""), kind="cause")
                if "effect" in regen or "effects" in regen:
                    existing.effect_final = _prettify(regen.get("effect") or regen.get("effects", ""), kind="effect")
                if "potentially_dangerous" in regen:
                    existing.potentially_dangerous_ai = bool(regen["potentially_dangerous"])

                existing.regenerated_flag = True
                updated.append(existing)
                logger.info("Row %s regenerated in run %s", row_id, run_id)

        return updated

    def get_all_rows(self, run_id: str) -> List[RowState]:
        """Get all rows for a run in order."""
        run = self._runs.get(run_id)
        if not run:
            return []

        # Sort by row_id numeric suffix
        def sort_key(row: RowState) -> int:
            try:
                return int(str(row.row_id).split("-")[1])
            except (IndexError, ValueError, AttributeError):
                return 0

        return sorted(run.rows.values(), key=sort_key)

    def get_raw_rows(self, run_id: str) -> tuple:
        """Get raw L1/L2/L3 rows for regeneration."""
        run = self._runs.get(run_id)
        if not run:
            return [], [], []
        return run._raw_l1, run._raw_l2, run._raw_l3


# Global state manager instance
state_manager = StateManager()
