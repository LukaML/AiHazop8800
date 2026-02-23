# src/gui/services/csv_exporter.py
"""CSV export from analysis state — includes original and final values,
human ratings, edit/regeneration flags, and run metadata for research use."""
import csv
import io
from datetime import datetime
from .state_manager import state_manager

# Research-grade CSV columns
CSV_COLUMNS = [
    "run_id",
    "provider",
    "model",
    "rag_enabled",
    "embeddings_backend",
    "row_id",
    "function",
    "guideword",
    "deviation_original",
    "cause_original",
    "effect_original",
    "deviation_final",
    "cause_final",
    "effect_final",
    "potentially_dangerous_ai",
    "potentially_dangerous_human",
    "potentially_dangerous_final",
    "rating",
    "edited_flag",
    "regenerated_flag",
    "timestamp",
]


def export_to_csv(run_id: str) -> str:
    """
    Export analysis results to CSV format.
    Returns CSV content as a string.
    """
    run = state_manager.get_run(run_id)
    if not run:
        raise ValueError(f"Run not found: {run_id}")

    rows = state_manager.get_all_rows(run_id)
    timestamp = datetime.utcnow().isoformat() + "Z"

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    for row in rows:
        writer.writerow({
            "run_id": run_id,
            "provider": run.provider,
            "model": run.model,
            "rag_enabled": run.rag_enabled,
            "embeddings_backend": run.rag_embedder or "",
            "row_id": row.row_id,
            "function": row.function,
            "guideword": row.guideword,
            "deviation_original": row.deviation_original,
            "cause_original": row.cause_original,
            "effect_original": row.effect_original,
            "deviation_final": row.deviation_final,
            "cause_final": row.cause_final,
            "effect_final": row.effect_final,
            "potentially_dangerous_ai": row.potentially_dangerous_ai,
            "potentially_dangerous_human": row.potentially_dangerous_human if row.potentially_dangerous_human is not None else "",
            "potentially_dangerous_final": row.potentially_dangerous_final,
            "rating": row.rating.value if row.rating else "unrated",
            "edited_flag": row.edited_flag,
            "regenerated_flag": row.regenerated_flag,
            "timestamp": timestamp,
        })

    return output.getvalue()
