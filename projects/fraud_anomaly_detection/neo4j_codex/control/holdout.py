"""Two-state leak-free holdout split for control promotion."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


@dataclass(frozen=True)
class TwoStateSplit:
    cutoff: pd.Timestamp
    newest: pd.Timestamp
    state_a_users: list[str]
    holdout_users: list[str]


def two_state_split(store: Path | str, config) -> TwoStateSplit:
    with duckdb.connect(str(store), read_only=True) as con:
        advances = con.execute(
            "SELECT CAST(user_id AS VARCHAR) AS user_id, feature_as_of_ts FROM advances"
        ).df()

    advances["feature_as_of_ts"] = pd.to_datetime(advances["feature_as_of_ts"])
    newest = advances["feature_as_of_ts"].max()
    cutoff = newest - pd.Timedelta(days=config.holdout_days)
    state_a_users = sorted(advances.loc[advances["feature_as_of_ts"] <= cutoff, "user_id"].unique())
    holdout_users = sorted(advances.loc[advances["feature_as_of_ts"] > cutoff, "user_id"].unique())
    return TwoStateSplit(
        cutoff=cutoff,
        newest=newest,
        state_a_users=state_a_users,
        holdout_users=holdout_users,
    )
