"""Monitoring stats for burned keys over a held-out user set."""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd


def holdout_effect(store: Path | str, burned_keys: pd.DataFrame, holdout_users: list[str]) -> dict:
    held = set(map(str, holdout_users))
    if {"entity_type", "entity_value"}.issubset(burned_keys.columns):
        keyset = set(zip(burned_keys["entity_type"], burned_keys["entity_value"]))
    else:
        keyset = set()

    with duckdb.connect(str(store), read_only=True) as con:
        edges = con.execute(
            "SELECT entity_type, entity_value, CAST(user_id AS VARCHAR) AS user_id FROM edges"
        ).df()
        outcomes = con.execute(
            """
            SELECT
                CAST(user_id AS VARCHAR) AS user_id,
                max(CASE WHEN label_mature_d45 AND label_gross_dpd45 THEN 1 ELSE 0 END) AS bad,
                max(CASE WHEN label_mature_d45 THEN 1 ELSE 0 END) AS mature
            FROM advances
            GROUP BY 1
            """
        ).df()

    caught = {
        row.user_id
        for row in edges.itertuples(index=False)
        if (row.entity_type, row.entity_value) in keyset
    }
    caught_held = caught & held
    outcome_by_user = outcomes.set_index("user_id")
    held_bad = {
        user_id
        for user_id in held
        if user_id in outcome_by_user.index and outcome_by_user.loc[user_id, "bad"] == 1
    }
    innocent_caught = {
        user_id
        for user_id in caught_held
        if user_id in outcome_by_user.index
        and outcome_by_user.loc[user_id, "mature"] == 1
        and outcome_by_user.loc[user_id, "bad"] == 0
    }
    return {
        "holdout_users": len(held),
        "prevented_bad": len(caught_held & held_bad),
        "innocents_blocked": len(innocent_caught),
        "leaked_bad": len(held_bad - caught_held),
    }
