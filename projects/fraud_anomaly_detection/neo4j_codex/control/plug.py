"""Plug derivation: extract candidate keys, validate facts, then qualify."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def candidate_stats(
    store: Path | str,
    discovered_users: list[str] | pd.Series,
    eligible_users: list[str] | pd.Series | None = None,
    end_ts: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Compute per-entity plug facts for discovered users.

    This is the expensive extract+validate pass. Its output is factual: support,
    DPD45 precision, discovered-user coverage, and mature innocent count.
    """
    discovered = set(pd.Series(discovered_users).astype(str))
    eligible = None if eligible_users is None else set(pd.Series(eligible_users).astype(str))
    with duckdb.connect(str(store), read_only=True) as con:
        edge_users = con.execute(
            """
            SELECT DISTINCT entity_type, entity_value, CAST(user_id AS VARCHAR) AS user_id, ts
            FROM edges
            """
        ).df()
        outcomes = con.execute(
            """
            SELECT
                CAST(user_id AS VARCHAR) AS user_id,
                feature_as_of_ts,
                max(CASE WHEN label_mature_d45 THEN 1 ELSE 0 END) AS mature,
                max(CASE WHEN label_mature_d45 AND label_gross_dpd45 THEN 1 ELSE 0 END) AS bad
            FROM advances
            GROUP BY 1, 2
            """
        ).df()

    if eligible is not None:
        edge_users = edge_users[edge_users["user_id"].isin(eligible)]
        outcomes = outcomes[outcomes["user_id"].isin(eligible)]
    if end_ts is not None:
        edge_users = edge_users[pd.to_datetime(edge_users["ts"]) <= pd.Timestamp(end_ts)]
        outcomes = outcomes[
            pd.to_datetime(outcomes["feature_as_of_ts"]) <= pd.Timestamp(end_ts)
        ]
    edge_users = edge_users.drop_duplicates(["entity_type", "entity_value", "user_id"])
    outcomes = (
        outcomes.groupby("user_id", as_index=False)
        .agg(mature=("mature", "max"), bad=("bad", "max"))
        if not outcomes.empty
        else pd.DataFrame(columns=["user_id", "mature", "bad"])
    )

    candidate_keys = edge_users.loc[
        edge_users["user_id"].isin(discovered), ["entity_type", "entity_value"]
    ].drop_duplicates()
    if candidate_keys.empty:
        return _empty_stats()
    edge_users = edge_users.merge(candidate_keys, on=["entity_type", "entity_value"], how="inner")

    frame = edge_users.merge(outcomes, on="user_id", how="left").fillna(
        {"mature": 0, "bad": 0}
    )
    frame["discovered"] = frame["user_id"].isin(discovered)
    frame["discovered_user_id"] = frame["user_id"].where(frame["discovered"])

    stats = (
        frame.groupby(["entity_type", "entity_value"])
        .agg(
            support=("user_id", "nunique"),
            mature_users=("mature", "sum"),
            bad_users=("bad", "sum"),
            coverage=("discovered_user_id", "nunique"),
        )
        .reset_index()
    )
    denominator = stats["mature_users"].where(stats["mature_users"] > 0)
    stats["dpd45_precision"] = (stats["bad_users"] / denominator).fillna(0.0)
    stats["innocents"] = stats["mature_users"] - stats["bad_users"]
    return stats


def qualify(stats: pd.DataFrame, config) -> pd.DataFrame:
    """Cheap conjunctive threshold filter over candidate stats."""
    keep = (
        (stats["support"] >= config.min_support)
        & (stats["coverage"] >= config.min_coverage)
        & (stats["dpd45_precision"] >= config.block_tier_precision)
    )
    return (
        stats.loc[keep]
        .sort_values(
            ["dpd45_precision", "coverage", "support"], ascending=False, kind="stable"
        )
        .reset_index(drop=True)
    )


def _empty_stats() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "entity_type",
            "entity_value",
            "support",
            "mature_users",
            "bad_users",
            "coverage",
            "dpd45_precision",
            "innocents",
        ]
    )
