"""DuckDB-backed plug candidate fact snapshots."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import duckdb
import pandas as pd

_SNAPSHOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS plug_candidate_fact_snapshots (
    snapshot_id VARCHAR,
    refresh_key VARCHAR,
    data_version VARCHAR,
    selection_hash VARCHAR,
    fact_hash VARCHAR,
    created_at TIMESTAMP DEFAULT current_timestamp
)
"""

_FACT_SCHEMA = """
CREATE TABLE IF NOT EXISTS plug_candidate_facts (
    snapshot_id VARCHAR,
    refresh_key VARCHAR,
    data_version VARCHAR,
    selection_hash VARCHAR,
    entity_type VARCHAR,
    entity_value VARCHAR,
    support BIGINT,
    mature_users BIGINT,
    bad_users BIGINT,
    coverage BIGINT,
    dpd45_precision DOUBLE,
    innocents BIGINT
)
"""

FACT_COLUMNS = [
    "entity_type",
    "entity_value",
    "support",
    "mature_users",
    "bad_users",
    "coverage",
    "dpd45_precision",
    "innocents",
]


class PlugFactStore:
    """Append-only snapshots of extracted candidate plug facts."""

    def __init__(self, path: Path | str):
        self.path = Path(path)
        with duckdb.connect(str(self.path)) as con:
            con.execute(_SNAPSHOT_SCHEMA)
            con.execute(_FACT_SCHEMA)

    def write_snapshot(
        self,
        refresh_key: str,
        data_version: str,
        selection_users: list[str] | pd.Series,
        facts: pd.DataFrame,
    ) -> str:
        frame = _normalize_facts(facts)
        selection_hash = _selection_hash(selection_users)
        fact_hash = _fact_hash(frame)
        snapshot_id = str(uuid4())
        frame = frame.assign(
            snapshot_id=snapshot_id,
            refresh_key=refresh_key,
            data_version=data_version,
            selection_hash=selection_hash,
        )
        with duckdb.connect(str(self.path)) as con:
            con.execute(
                """
                INSERT INTO plug_candidate_fact_snapshots
                    (snapshot_id, refresh_key, data_version, selection_hash, fact_hash)
                VALUES (?, ?, ?, ?, ?)
                """,
                [snapshot_id, refresh_key, data_version, selection_hash, fact_hash],
            )
            if not frame.empty:
                con.execute(
                    """
                    INSERT INTO plug_candidate_facts
                    SELECT snapshot_id, refresh_key, data_version, selection_hash,
                           entity_type, entity_value, support, mature_users,
                           bad_users, coverage, dpd45_precision, innocents
                    FROM frame
                    """
                )
        return snapshot_id

    def read_latest(self) -> pd.DataFrame:
        latest = self.latest_snapshot()
        if latest is None:
            return pd.DataFrame(columns=["snapshot_id", *FACT_COLUMNS])
        with duckdb.connect(str(self.path), read_only=True) as con:
            return con.execute(
                """
                SELECT snapshot_id, entity_type, entity_value, support, mature_users,
                       bad_users, coverage, dpd45_precision, innocents
                FROM plug_candidate_facts
                WHERE snapshot_id = ?
                ORDER BY entity_type, entity_value
                """,
                [latest["snapshot_id"]],
            ).df()

    def latest_snapshot(self) -> dict | None:
        with duckdb.connect(str(self.path), read_only=True) as con:
            row = con.execute(
                """
                SELECT snapshot_id, refresh_key, data_version, selection_hash, fact_hash
                FROM plug_candidate_fact_snapshots
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
            "selection_hash": row[3],
            "fact_hash": row[4],
        }


def _normalize_facts(facts: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in FACT_COLUMNS if column not in facts.columns]
    if missing:
        raise KeyError(f"Candidate facts are missing columns: {', '.join(missing)}")
    return facts[FACT_COLUMNS].copy()


def _selection_hash(selection_users: list[str] | pd.Series) -> str:
    users = sorted({str(user_id) for user_id in pd.Series(selection_users)})
    return hashlib.sha256(json.dumps(users).encode()).hexdigest()


def _fact_hash(facts: pd.DataFrame) -> str:
    records = facts.sort_values(["entity_type", "entity_value"]).to_dict("records")
    canonical = json.dumps(records, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()
