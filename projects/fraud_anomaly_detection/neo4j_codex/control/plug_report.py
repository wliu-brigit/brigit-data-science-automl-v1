"""Plug validation reports split by discovery coverage and outside discovery."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.metrics import (
    load_advances,
    outcome,
    user_truth,
)


def summarize_plugs(
    store: Path | str,
    burned_keys: pd.DataFrame,
    discovery_users: Iterable[str],
    eligible_users: Iterable[str] | None = None,
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> dict:
    """Measure who deployable burned keys would touch and what outcomes they carry."""
    discovery = {str(user_id) for user_id in discovery_users}
    eligible = None if eligible_users is None else {str(user_id) for user_id in eligible_users}
    if eligible is not None:
        discovery &= eligible

    covered = _covered_users(store, burned_keys, eligible, start_ts, end_ts)
    covered_discovery = covered & discovery
    outside_discovery = covered - discovery
    uncovered_discovery = discovery - covered

    truth = user_truth(load_advances(store), start=start_ts, end=end_ts)
    return {
        "burned_key_count": int(len(burned_keys)),
        "covered_users": _bucket(truth, covered),
        "covered_discovery": _bucket(truth, covered_discovery),
        "outside_discovery": _bucket(truth, outside_discovery),
        "uncovered_discovery": _bucket(truth, uncovered_discovery),
    }


def _covered_users(
    store: Path | str,
    burned_keys: pd.DataFrame,
    eligible_users: set[str] | None,
    start_ts: pd.Timestamp | None,
    end_ts: pd.Timestamp | None,
) -> set[str]:
    if not {"entity_type", "entity_value"}.issubset(burned_keys.columns):
        return set()
    keyset = {
        (str(row.entity_type), str(row.entity_value))
        for row in burned_keys[["entity_type", "entity_value"]].itertuples(index=False)
    }
    if not keyset:
        return set()

    with duckdb.connect(str(store), read_only=True) as con:
        edges = con.execute(
            """
            SELECT
                CAST(user_id AS VARCHAR) AS user_id,
                CAST(entity_type AS VARCHAR) AS entity_type,
                CAST(entity_value AS VARCHAR) AS entity_value,
                ts
            FROM edges
            """
        ).df()

    if edges.empty:
        return set()

    key_frame = pd.DataFrame(list(keyset), columns=["entity_type", "entity_value"])
    frame = edges.merge(key_frame, on=["entity_type", "entity_value"], how="inner")
    frame["ts"] = pd.to_datetime(frame["ts"])
    if eligible_users is not None:
        frame = frame[frame["user_id"].isin(eligible_users)]
    if start_ts is not None:
        frame = frame[frame["ts"] > pd.Timestamp(start_ts)]
    if end_ts is not None:
        frame = frame[frame["ts"] <= pd.Timestamp(end_ts)]
    return set(frame["user_id"].astype(str))


def _bucket(truth: pd.DataFrame, users: set[str]) -> dict:
    return {
        "n_users": len(users),
        "outcomes": outcome(users, truth),
    }
