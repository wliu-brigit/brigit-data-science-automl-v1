"""The one outcome metric for the control loop — discovery precision and plug coverage.

`user_truth` rolls advances up to a per-user truth frame, with an optional
`feature_as_of_ts` window; `outcome` scores a user set against that frame. Discovery
(no window) and plug/coverage validation (windowed before/after a cutoff) differ only
in the frame they build — the scoring is one definition, so every panel is comparable
by construction. `outcome` reports both DPD45 user-rate denominators: over the whole
set (discovery) and over users with an advance in the frame (coverage).
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.scenarios import SCENARIOS, assign


def load_advances(store: Path | str) -> pd.DataFrame:
    """Load the advance grain from a DuckDB graph store."""
    with duckdb.connect(str(store), read_only=True) as con:
        return con.execute("SELECT * FROM advances").df()


def asof_advances(advances: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Advances whose feature snapshot is at or before ``cutoff`` (as-of state)."""
    return advances[
        pd.to_datetime(advances["feature_as_of_ts"]) <= pd.Timestamp(cutoff)
    ].copy()


def window_advances(
    advances: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Advances whose ``feature_as_of_ts`` falls in ``(start, end]`` (either bound optional)."""
    if start is None and end is None:
        return advances
    ts = pd.to_datetime(advances["feature_as_of_ts"])
    mask = pd.Series(True, index=advances.index)
    if start is not None:
        mask &= ts > pd.Timestamp(start)
    if end is not None:
        mask &= ts <= pd.Timestamp(end)
    return advances[mask]


_TRUTH_COLUMNS = ("dpd45", "n_advances", "n_mature_advances", "n_dpd45_advances")


def user_truth(
    advances: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    end: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Roll advances up to a per-user truth frame (DPD45 + scenario flags).

    ``start``/``end`` window the advance grain on ``feature_as_of_ts`` before the
    rollup — the same filter the plug/coverage measurement needs (advances after a
    cutoff, or up to it). Without a window every user with an advance is included.
    """
    advances = window_advances(advances, start=start, end=end)
    if advances.empty:
        return pd.DataFrame(columns=_TRUTH_COLUMNS, index=pd.Index([], name="user_id"))
    flags = assign(advances)
    truth = pd.DataFrame(
        {
            "user_id": advances.user_id.astype(str),
            "mature_d45": advances.label_mature_d45.fillna(False).astype(bool),
            "dpd45": (
                advances.label_mature_d45.fillna(False).astype(bool)
                & advances.label_gross_dpd45.fillna(False).astype(bool)
            ),
            "is_fraud": advances.is_fraud.fillna(False).astype(bool),
        }
    )
    for scenario in SCENARIOS:
        truth[f"scenario_{scenario.name}"] = flags[f"scenario_{scenario.name}"].fillna(
            False
        ).astype(bool)
    truth["scenario_any"] = flags.scenario_any.fillna(False).astype(bool)

    rolled = truth.groupby("user_id").agg(
        mature_d45=("mature_d45", "max"),
        dpd45=("dpd45", "max"),
        is_fraud=("is_fraud", "max"),
        scenario_any=("scenario_any", "max"),
        n_advances=("user_id", "size"),
        n_mature_advances=("mature_d45", "sum"),
        n_dpd45_advances=("dpd45", "sum"),
    )
    for scenario in SCENARIOS:
        col = f"scenario_{scenario.name}"
        rolled[col] = truth.groupby("user_id")[col].max()
    return rolled


def empty_outcome() -> dict:
    """Zeroed outcome panel for an empty user set."""
    return {
        "users": 0,
        "users_with_advances": 0,
        "dpd45_users": 0,
        "dpd45_user_rate": 0.0,
        "dpd45_user_rate_with_advances": 0.0,
        "advances": 0,
        "mature_advances": 0,
        "dpd45_advances": 0,
        "dpd45_advance_rate": 0.0,
    }


def _rate(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def outcome(users: Iterable[str], truth: pd.DataFrame) -> dict:
    """Score a user set against a truth frame — the one discovery/coverage metric.

    Reports both DPD45 user-rate denominators: ``dpd45_user_rate`` over the whole set
    (discovery's convention) and ``dpd45_user_rate_with_advances`` over only the users
    that have an advance in the frame (the coverage/plug convention). Users absent from
    the frame have no scorable advance: they count toward ``users`` but not
    ``users_with_advances`` — so a windowed frame measures only who transacted in it.
    """
    user_ids = {str(user_id) for user_id in users}
    if not user_ids:
        return empty_outcome()
    sub_users = truth.loc[truth.index.isin(user_ids)]
    users_with_advances = int(len(sub_users))
    dpd45_users = int(sub_users.dpd45.sum()) if users_with_advances else 0
    advances = int(sub_users.n_advances.sum()) if users_with_advances else 0
    mature_advances = int(sub_users.n_mature_advances.sum()) if users_with_advances else 0
    dpd45_advances = int(sub_users.n_dpd45_advances.sum()) if users_with_advances else 0
    return {
        "users": len(user_ids),
        "users_with_advances": users_with_advances,
        "dpd45_users": dpd45_users,
        "dpd45_user_rate": _rate(dpd45_users, len(user_ids)),
        "dpd45_user_rate_with_advances": _rate(dpd45_users, users_with_advances),
        "advances": advances,
        "mature_advances": mature_advances,
        "dpd45_advances": dpd45_advances,
        "dpd45_advance_rate": _rate(dpd45_advances, mature_advances),
    }
