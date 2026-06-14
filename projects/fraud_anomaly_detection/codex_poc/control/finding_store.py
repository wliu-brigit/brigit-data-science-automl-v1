"""DuckDB-backed finding snapshots for the fraud-control loop."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.codex_poc.control.contract import FindingSet

_SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    refresh_key VARCHAR,
    data_version VARCHAR,
    content_hash VARCHAR,
    method VARCHAR,
    method_version VARCHAR,
    user_id VARCHAR,
    score DOUBLE,
    evidence VARCHAR
)
"""

_SNAPSHOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS finding_snapshots (
    refresh_key VARCHAR,
    data_version VARCHAR,
    content_hash VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
)
"""


class FindingStore:
    """Append-only finding snapshots with trim-when-unchanged semantics."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with duckdb.connect(str(self.path)) as con:
            con.execute(_SCHEMA)
            con.execute(_SNAPSHOT_SCHEMA)

    def refresh_keys(self) -> list[str]:
        with duckdb.connect(str(self.path), read_only=True) as con:
            rows = con.execute(
                "SELECT refresh_key FROM finding_snapshots ORDER BY created_at, rowid"
            ).fetchall()
        return [row[0] for row in rows]

    def write_snapshot(
        self, refresh_key: str, data_version: str, finding_sets: list[FindingSet]
    ) -> bool:
        """Write a snapshot unless its material content matches the latest one."""
        frame = self._frame(finding_sets)
        content_hash = self._hash(finding_sets)
        if content_hash == self._latest_hash():
            return False

        frame = frame.assign(
            refresh_key=refresh_key,
            data_version=data_version,
            content_hash=content_hash,
        )
        with duckdb.connect(str(self.path)) as con:
            con.execute(
                """
                INSERT INTO finding_snapshots (refresh_key, data_version, content_hash)
                VALUES (?, ?, ?)
                """,
                [refresh_key, data_version, content_hash],
            )
            if not frame.empty:
                con.execute(
                    """
                    INSERT INTO findings
                    SELECT refresh_key, data_version, content_hash, method, method_version,
                           user_id, score, evidence
                    FROM frame
                    """
                )
        return True

    def read_latest(self) -> pd.DataFrame:
        keys = self.refresh_keys()
        if not keys:
            return pd.DataFrame(
                columns=[
                    "refresh_key",
                    "data_version",
                    "method",
                    "method_version",
                    "user_id",
                    "score",
                    "evidence",
                ]
            )

        with duckdb.connect(str(self.path), read_only=True) as con:
            return con.execute(
                """
                SELECT refresh_key, data_version, method, method_version, user_id, score, evidence
                FROM findings
                WHERE refresh_key = ?
                """,
                [keys[-1]],
            ).df()

    def _latest_hash(self) -> str | None:
        keys = self.refresh_keys()
        if not keys:
            return None
        with duckdb.connect(str(self.path), read_only=True) as con:
            row = con.execute(
                "SELECT content_hash FROM finding_snapshots WHERE refresh_key = ? LIMIT 1",
                [keys[-1]],
            ).fetchone()
        return None if row is None else row[0]

    @staticmethod
    def _frame(finding_sets: list[FindingSet]) -> pd.DataFrame:
        if not finding_sets:
            return pd.DataFrame(
                columns=["user_id", "method", "method_version", "score", "evidence"]
            )
        frame = pd.concat([finding_set.to_frame() for finding_set in finding_sets], ignore_index=True)
        frame["evidence"] = frame["evidence"].map(
            lambda evidence: json.dumps(evidence, sort_keys=True, default=str)
        )
        return frame

    @staticmethod
    def _hash(finding_sets: list[FindingSet]) -> str:
        records = []
        for finding_set in finding_sets:
            frame = finding_set.to_frame()
            if frame.empty:
                records.append(
                    {
                        "method": finding_set.method,
                        "method_version": finding_set.method_version,
                        "user_id": None,
                        "score": None,
                        "evidence": None,
                    }
                )
                continue
            for row in frame.itertuples(index=False):
                records.append(
                    {
                        "method": row.method,
                        "method_version": row.method_version,
                        "user_id": row.user_id,
                        "score": float(row.score),
                        "evidence": json.dumps(row.evidence, sort_keys=True, default=str),
                    }
                )
        records = sorted(
            records,
            key=lambda record: (
                record["method"],
                record["method_version"],
                "" if record["user_id"] is None else record["user_id"],
                -1.0 if record["score"] is None else record["score"],
                "" if record["evidence"] is None else record["evidence"],
            ),
        )
        canonical = json.dumps(records, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
