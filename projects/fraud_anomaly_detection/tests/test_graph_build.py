"""Store builder: lossless edges, sentinel screening, self-contained snapshot."""

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")  # project deps: uv sync --group fraud

from projects.fraud_anomaly_detection.graph.build import build_store  # noqa: E402

pytestmark = pytest.mark.unit


def _q(path, sql):
    with duckdb.connect(str(path), read_only=True) as con:
        return con.execute(sql).fetchall()


def test_edges_lossless_per_type(toy_store):
    counts = dict(_q(toy_store, "SELECT entity_type, count(*) FROM edges GROUP BY 1"))
    assert counts == {"device": 9, "bank": 2}


def test_no_degree_cap_applied(toy_store):
    [(n_users,)] = _q(toy_store,
        "SELECT n_users FROM entities WHERE entity_value = 'dH'")
    assert n_users == 5  # hub kept in full — caps are analysis-time only


def test_sentinels_screened_and_counted(toy_store):
    assert _q(toy_store, "SELECT count(*) FROM edges WHERE entity_value = 'none'") == [(0,)]
    [(v,)] = _q(toy_store, "SELECT value FROM meta WHERE key = 'screened_device'")
    assert int(v) == 1


def test_duplicate_source_rows_dedup_to_one_edge(toy_store):
    assert _q(toy_store,
        "SELECT count(*) FROM edges WHERE advance_id = 'a02'") == [(1,)]


def test_advances_snapshot_is_complete(toy_store, toy_df):
    [(n,)] = _q(toy_store, "SELECT count(*) FROM advances")
    assert n == len(toy_df)  # full snapshot, duplicates and sentinels included
    cols = [r[0] for r in _q(toy_store, "DESCRIBE advances")]
    assert set(cols) == set(toy_df.columns)


def test_aggregates_consistent_with_edges(toy_store):
    mismatches = _q(toy_store, """
        SELECT e.entity_value FROM entities e JOIN (
            SELECT entity_type, entity_value, count(DISTINCT user_id) nu
            FROM edges GROUP BY 1, 2) c
        USING (entity_type, entity_value) WHERE e.n_users <> c.nu""")
    assert mismatches == []
    [(n_users,)] = _q(toy_store, "SELECT count(*) FROM users")
    assert n_users == 9


def test_rebuild_is_idempotent(toy_df, tmp_path):
    path = tmp_path / "g.duckdb"
    first = build_store(toy_df, path, source_label="toy")
    second = build_store(toy_df, path, source_label="toy")
    assert first == second
    assert _q(path, "SELECT count(*) FROM edges") == [(11,)]
