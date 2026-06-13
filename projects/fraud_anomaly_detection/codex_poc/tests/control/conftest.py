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
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "is_fraud": [True, False, False, False, False],
            "label_gross_dpd45": [True, True, True, False, False],
            "label_mature_d45": [True, True, True, True, True],
            "feature_as_of_ts": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-15", "2026-02-20", "2026-01-03"]
            ),
        }
    )
    edges = pd.DataFrame(
        {
            "user_id": ["u1", "u2", "u3", "u4", "u5"],
            "entity_type": ["bank", "bank", "bank", "device", "device"],
            "entity_value": ["acctA", "acctA", "acctA", "devX", "devX"],
            "ts": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-02-15", "2026-02-20", "2026-01-03"]
            ),
            "source": ["advance"] * 5,
        }
    )
    con.execute("CREATE TABLE advances AS SELECT * FROM advances")
    con.execute("CREATE TABLE edges AS SELECT * FROM edges")
    con.close()
    return path
