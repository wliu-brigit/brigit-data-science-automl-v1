"""Tiny control-loop stores for tests."""
from __future__ import annotations

import duckdb
import pandas as pd
import pytest


@pytest.fixture
def tiny_store(tmp_path):
    path = tmp_path / "tiny.duckdb"
    con = duckdb.connect(str(path))
    advances = pd.DataFrame(
        {
            "advance_id": ["a1", "a2", "a3", "a4", "a5"],
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "is_fraud": [True, False, False, False, False],
            "label_gross_dpd45": [True, True, True, False, False],
            "label_mature_d45": [True, True, True, True, True],
            "feature_as_of_ts": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-15", "2026-02-20", "2026-01-03"]
            ),
            "identity_created_time": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-10", "2026-02-18", "2025-12-01"]
            ),
            "loan_amount": [150.0, 120.0, 90.0, 80.0, 50.0],
            "prior_advances_on_bank_account_7d": [1, 1, 0, 0, 0],
            "users_on_bank_account_72h": [2, 2, 0, 0, 0],
            "users_on_persistent_account_id_72h": [2, 2, 0, 0, 0],
            "is_joint": [0, 0, 0, 0, 0],
            "users_on_device_id_72h": [0, 0, 0, 0, 0],
        }
    )
    edges = pd.DataFrame(
        {
            "advance_id": ["a1", "a2", "a3", "a4", "a5"],
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "entity_type": ["bank", "bank", "bank", "device", "device"],
            "entity_value": ["acctA", "acctA", "acctA", "devX", "devX"],
            "ts": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-15", "2026-02-20", "2026-01-03"]
            ),
            "source": ["advance"] * 5,
        }
    )
    con.register("advances_df", advances)
    con.register("edges_df", edges)
    con.execute("CREATE TABLE advances AS SELECT * FROM advances_df")
    con.execute("CREATE TABLE edges AS SELECT * FROM edges_df")
    con.execute(
        """
        CREATE TABLE users AS
        SELECT DISTINCT CAST(user_id AS VARCHAR) AS user_id
        FROM advances_df
        """
    )
    con.close()
    return path
