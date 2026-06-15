"""DuckDB persistence for daily fraud-control run reports."""
from __future__ import annotations

import json
from pathlib import Path

import duckdb

_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_reports (
    refresh_key VARCHAR,
    data_version VARCHAR,
    report_json VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
)
"""


class ReportStore:
    """Append-only JSON report store for run-level monitoring history."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with duckdb.connect(str(self.path)) as con:
            con.execute(_SCHEMA)

    def write_report(self, refresh_key: str, data_version: str, report: dict) -> None:
        payload = json.dumps(report, sort_keys=True, default=str)
        with duckdb.connect(str(self.path)) as con:
            con.execute(
                """
                INSERT INTO run_reports (refresh_key, data_version, report_json)
                VALUES (?, ?, ?)
                """,
                [refresh_key, data_version, payload],
            )

    def read_latest(self) -> dict | None:
        with duckdb.connect(str(self.path), read_only=True) as con:
            row = con.execute(
                """
                SELECT refresh_key, data_version, report_json
                FROM run_reports
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """
            ).fetchone()

        if row is None:
            return None
        return {
            "refresh_key": row[0],
            "data_version": row[1],
            "report": json.loads(row[2]),
        }
