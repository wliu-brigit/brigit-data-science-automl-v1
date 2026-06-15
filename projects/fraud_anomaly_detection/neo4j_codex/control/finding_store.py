"""DuckDB-backed finding snapshots for the fraud-control loop."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.neo4j_codex.control.contract import FindingSet
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.metadata import MethodMetadata

_SCHEMA = """
CREATE TABLE IF NOT EXISTS findings (
    snapshot_id VARCHAR,
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
    snapshot_id VARCHAR,
    refresh_key VARCHAR,
    data_version VARCHAR,
    content_hash VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
)
"""

_METHOD_SNAPSHOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS method_snapshots (
    snapshot_id VARCHAR,
    refresh_key VARCHAR,
    data_version VARCHAR,
    method VARCHAR,
    method_version VARCHAR,
    method_type VARCHAR,
    time_semantics VARCHAR,
    promotion_tier VARCHAR,
    enforcement_projection VARCHAR,
    enabled BOOLEAN,
    params_json VARCHAR
)
"""


class FindingStore:
    """Append-only finding snapshots with trim-when-unchanged semantics."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with duckdb.connect(str(self.path)) as con:
            con.execute(_SCHEMA)
            con.execute(_SNAPSHOT_SCHEMA)
            con.execute(_METHOD_SNAPSHOT_SCHEMA)

    def refresh_keys(self) -> list[str]:
        with duckdb.connect(str(self.path), read_only=True) as con:
            rows = con.execute(
                "SELECT refresh_key FROM finding_snapshots ORDER BY created_at, rowid"
            ).fetchall()
        return [row[0] for row in rows]

    def write_snapshot(
        self,
        refresh_key: str,
        data_version: str,
        finding_sets: list[FindingSet],
        method_metadata: list[MethodMetadata],
    ) -> bool:
        """Write a snapshot unless its material content matches the latest one."""
        _validate_metadata_snapshot(finding_sets, method_metadata)
        frame = self._frame(finding_sets)
        metadata_frame = self._metadata_frame(method_metadata)
        content_hash = self._hash(finding_sets, method_metadata)
        if content_hash == self._latest_hash():
            return False

        snapshot_id = str(uuid4())
        frame = frame.assign(
            snapshot_id=snapshot_id,
            refresh_key=refresh_key,
            data_version=data_version,
            content_hash=content_hash,
        )
        metadata_frame = metadata_frame.assign(
            snapshot_id=snapshot_id,
            refresh_key=refresh_key,
            data_version=data_version,
        )
        with duckdb.connect(str(self.path)) as con:
            con.execute(
                """
                INSERT INTO finding_snapshots (snapshot_id, refresh_key, data_version, content_hash)
                VALUES (?, ?, ?, ?)
                """,
                [snapshot_id, refresh_key, data_version, content_hash],
            )
            con.execute(
                """
                INSERT INTO method_snapshots
                SELECT snapshot_id, refresh_key, data_version, method, method_version,
                       method_type, time_semantics, promotion_tier,
                       enforcement_projection, enabled, params_json
                FROM metadata_frame
                """
            )
            if not frame.empty:
                con.execute(
                    """
                    INSERT INTO findings
                    SELECT snapshot_id, refresh_key, data_version, content_hash, method,
                           method_version, user_id, score, evidence
                    FROM frame
                    """
                )
        return True

    def read_latest(self) -> pd.DataFrame:
        latest = self.latest_snapshot()
        if latest is None:
            return pd.DataFrame(
                columns=[
                    "snapshot_id",
                    "refresh_key",
                    "data_version",
                    "content_hash",
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
                SELECT snapshot_id, refresh_key, data_version, content_hash, method,
                       method_version, user_id, score, evidence
                FROM findings
                WHERE snapshot_id = ?
                ORDER BY method, method_version, user_id, score, evidence
                """,
                [latest["snapshot_id"]],
            ).df()

    def latest_snapshot(self) -> dict | None:
        with duckdb.connect(str(self.path), read_only=True) as con:
            row = con.execute(
                """
                SELECT snapshot_id, refresh_key, data_version, content_hash
                FROM finding_snapshots
                ORDER BY created_at DESC, rowid DESC
                LIMIT 1
                """
            ).fetchone()
        if row is None:
            return None
        return {
            "snapshot_id": row[0],
            "refresh_key": row[1],
            "data_version": row[2],
            "content_hash": row[3],
        }

    def read_latest_method_metadata(self) -> pd.DataFrame:
        latest = self.latest_snapshot()
        if latest is None:
            return pd.DataFrame(
                columns=[
                    "snapshot_id",
                    "refresh_key",
                    "data_version",
                    "method",
                    "method_version",
                    "method_type",
                    "time_semantics",
                    "promotion_tier",
                    "enforcement_projection",
                    "enabled",
                    "params_json",
                ]
            )
        with duckdb.connect(str(self.path), read_only=True) as con:
            return con.execute(
                """
                SELECT snapshot_id, refresh_key, data_version, method, method_version,
                       method_type, time_semantics, promotion_tier,
                       enforcement_projection, enabled, params_json
                FROM method_snapshots
                WHERE snapshot_id = ?
                ORDER BY method
                """,
                [latest["snapshot_id"]],
            ).df()

    def _latest_hash(self) -> str | None:
        latest = self.latest_snapshot()
        return None if latest is None else str(latest["content_hash"])

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
    def _metadata_frame(method_metadata: list[MethodMetadata]) -> pd.DataFrame:
        rows = [
            {
                "method": metadata.name,
                "method_version": metadata.version,
                "method_type": metadata.method_type,
                "time_semantics": metadata.time_semantics,
                "promotion_tier": metadata.promotion_tier,
                "enforcement_projection": metadata.enforcement_projection,
                "enabled": metadata.enabled,
                "params_json": json.dumps(_plain_json(metadata.params), sort_keys=True),
            }
            for metadata in method_metadata
        ]
        return pd.DataFrame(
            rows,
            columns=[
                "method",
                "method_version",
                "method_type",
                "time_semantics",
                "promotion_tier",
                "enforcement_projection",
                "enabled",
                "params_json",
            ],
        )

    @staticmethod
    def _hash(finding_sets: list[FindingSet], method_metadata: list[MethodMetadata]) -> str:
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
        metadata_records = [
            {
                "method": metadata.name,
                "method_version": metadata.version,
                "method_type": metadata.method_type,
                "time_semantics": metadata.time_semantics,
                "promotion_tier": metadata.promotion_tier,
                "enforcement_projection": metadata.enforcement_projection,
                "enabled": metadata.enabled,
                "params": json.dumps(_plain_json(metadata.params), sort_keys=True),
            }
            for metadata in method_metadata
        ]
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
        canonical = json.dumps(
            {
                "findings": records,
                "method_metadata": sorted(
                    metadata_records,
                    key=lambda item: item["method"],
                ),
            },
            sort_keys=True,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


def _validate_metadata_snapshot(
    finding_sets: list[FindingSet],
    method_metadata: list[MethodMetadata],
) -> None:
    if len(finding_sets) != len(method_metadata):
        raise ValueError(
            f"Expected one metadata record per FindingSet; got {len(method_metadata)} "
            f"metadata records for {len(finding_sets)} finding sets"
        )
    for finding_set, metadata in zip(finding_sets, method_metadata, strict=True):
        if finding_set.method != metadata.name:
            raise ValueError(
                f"FindingSet method {finding_set.method!r} does not match metadata "
                f"{metadata.name!r}"
            )
        if finding_set.method_version != metadata.version:
            raise ValueError(
                f"FindingSet version {finding_set.method_version!r} does not match "
                f"metadata version {metadata.version!r}"
            )


def _plain_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_plain_json(item) for item in value]
    if isinstance(value, set | frozenset):
        return sorted(_plain_json(item) for item in value)
    return value
