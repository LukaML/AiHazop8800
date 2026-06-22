# src/gui/services/csv_exporter.py
"""Research-grade CSV export — original and final values for every L1–L8 field,
plus human ratings, edit/regeneration flags, and run metadata."""
import csv
import io
from datetime import datetime

from .state_manager import state_manager, _EXPORT_FIELDS

_META_COLUMNS = ["run_id", "provider", "model", "row_id", "component_index"]
_TAIL_COLUMNS = ["rating", "edited_flag", "regenerated_flag", "timestamp"]


def _columns() -> list:
    cols = list(_META_COLUMNS)
    for f in _EXPORT_FIELDS:
        cols.append(f"{f}_original")
        cols.append(f"{f}_final")
    cols.extend(_TAIL_COLUMNS)
    return cols


def export_to_csv(run_id: str) -> str:
    """Export analysis results to CSV. Returns CSV content as a string."""
    run = state_manager.get_run(run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")

    rows = state_manager.get_all_rows(run_id)
    timestamp = datetime.utcnow().isoformat() + "Z"

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=_columns())
    writer.writeheader()

    for row in rows:
        record = row.to_dict_for_export()
        record["run_id"] = run_id
        record["provider"] = run.provider
        record["model"] = run.model
        record["timestamp"] = timestamp
        writer.writerow(record)

    return output.getvalue()
