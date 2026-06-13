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


class FindingStore:
    """Append-only finding snapshots with trim-when-unchanged semantics."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with duckdb.connect(str(self.path)) as con:
            con.execute(_SCHEMA)

    def refresh_keys(self) -> list[str]:
        with duckdb.connect(str(self.path), read_only=True) as con:
            rows = con.execute(
                "SELECT DISTINCT refresh_key FROM findings ORDER BY refresh_key"
            ).fetchall()
        return [row[0] for row in rows]

    def write_snapshot(
        self, refresh_key: str, data_version: str, finding_sets: list[FindingSet]
    ) -> bool:
        """Write a snapshot unless its material content matches the latest one."""
        frame = self._frame(finding_sets)
        content_hash = self._hash(frame)
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
                "SELECT content_hash FROM findings WHERE refresh_key = ? LIMIT 1",
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
    def _hash(frame: pd.DataFrame) -> str:
        canonical = frame[
            ["method", "method_version", "user_id", "score", "evidence"]
        ].sort_values(["method", "method_version", "user_id"]).to_csv(index=False)
        return hashlib.sha256(canonical.encode()).hexdigest()
