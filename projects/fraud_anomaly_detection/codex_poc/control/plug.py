"""Plug derivation: extract candidate keys, validate facts, then qualify."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def candidate_stats(
    store: Path | str,
    discovered_users: pd.Series,
    eligible_users: list[str] | pd.Series | None = None,
) -> pd.DataFrame:
    """Compute per-entity plug facts for discovered users.

    This is the expensive extract+validate pass. Its output is factual: support,
    DPD45 precision, discovered-user coverage, and mature innocent count.
    """
    discovered = set(discovered_users.astype(str))
    eligible = None if eligible_users is None else set(pd.Series(eligible_users).astype(str))
    with duckdb.connect(str(store), read_only=True) as con:
        edge_users = con.execute(
            """
            SELECT entity_type, entity_value, CAST(user_id AS VARCHAR) AS user_id
            FROM edges
            GROUP BY 1, 2, 3
            """
        ).df()
        outcomes = con.execute(
            """
            SELECT
                CAST(user_id AS VARCHAR) AS user_id,
                max(CASE WHEN label_mature_d45 THEN 1 ELSE 0 END) AS mature,
                max(CASE WHEN label_mature_d45 AND label_gross_dpd45 THEN 1 ELSE 0 END) AS bad
            FROM advances
            GROUP BY 1
            """
        ).df()

    if eligible is not None:
        edge_users = edge_users[edge_users["user_id"].isin(eligible)]

    frame = edge_users.merge(outcomes, on="user_id", how="left").fillna(
        {"mature": 0, "bad": 0}
    )
    frame["discovered"] = frame["user_id"].isin(discovered)

    stats = (
        frame.groupby(["entity_type", "entity_value"])
        .agg(
            support=("user_id", "nunique"),
            mature_users=("mature", "sum"),
            bad_users=("bad", "sum"),
            coverage=("discovered", "sum"),
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
