"""Advance-grain outcome rollups for discovery and plug validation."""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd


def summarize_users(
    store: Path | str,
    users: Iterable[str],
    start_ts: pd.Timestamp | None = None,
    end_ts: pd.Timestamp | None = None,
) -> dict:
    """Summarize DPD45 outcomes for a user set at advance grain."""
    user_ids = _normalize_users(users)
    empty = _empty_summary(len(user_ids))
    if not user_ids:
        return empty

    with duckdb.connect(str(store), read_only=True) as con:
        advances = con.execute(
            """
            SELECT
                CAST(user_id AS VARCHAR) AS user_id,
                label_mature_d45,
                label_gross_dpd45,
                feature_as_of_ts
            FROM advances
            """
        ).df()

    if advances.empty:
        return empty

    frame = advances[advances["user_id"].isin(user_ids)].copy()
    if frame.empty:
        return empty

    frame["feature_as_of_ts"] = pd.to_datetime(frame["feature_as_of_ts"])
    if start_ts is not None:
        frame = frame[frame["feature_as_of_ts"] > pd.Timestamp(start_ts)]
    if end_ts is not None:
        frame = frame[frame["feature_as_of_ts"] <= pd.Timestamp(end_ts)]
    if frame.empty:
        return empty

    mature = frame["label_mature_d45"].fillna(False).astype(bool)
    dpd45 = mature & frame["label_gross_dpd45"].fillna(False).astype(bool)
    n_mature = int(mature.sum())
    n_dpd45 = int(dpd45.sum())
    n_users_with_advances = int(frame["user_id"].nunique())
    n_dpd45_users = int(frame.loc[dpd45, "user_id"].nunique())

    return {
        "n_users": len(user_ids),
        "n_users_with_advances": n_users_with_advances,
        "n_advances": int(len(frame)),
        "n_mature_advances": n_mature,
        "n_dpd45_advances": n_dpd45,
        "dpd45_advance_rate": _rate(n_dpd45, n_mature),
        "n_dpd45_users": n_dpd45_users,
        "dpd45_user_rate": _rate(n_dpd45_users, n_users_with_advances),
    }


def _normalize_users(users: Iterable[str]) -> list[str]:
    return sorted({str(user_id) for user_id in users})


def _empty_summary(n_users: int) -> dict:
    return {
        "n_users": n_users,
        "n_users_with_advances": 0,
        "n_advances": 0,
        "n_mature_advances": 0,
        "n_dpd45_advances": 0,
        "dpd45_advance_rate": 0.0,
        "n_dpd45_users": 0,
        "dpd45_user_rate": 0.0,
    }


def _rate(numerator: int, denominator: int) -> float:
    return 0.0 if denominator == 0 else float(numerator / denominator)
