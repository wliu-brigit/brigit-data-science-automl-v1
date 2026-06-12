"""Store builder: lossless edges, sentinel screening, self-contained snapshot."""

import pandas as pd
import pytest

duckdb = pytest.importorskip("duckdb")  # project deps: uv sync --group fraud

from projects.fraud_anomaly_detection.graph.build import build_store  # noqa: E402

_TS = pd.Timestamp

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


def test_edge_timestamps_keep_time_of_day(toy_store):
    # The prior effort's worst bug was day-bucketing: rings burst intra-day.
    [(ts,)] = _q(toy_store,
        "SELECT ts FROM edges WHERE advance_id = 'a11'")
    assert (ts.hour, ts.minute) == (11, 0)


def test_parquet_source_builds_identically(toy_df, tmp_path):
    pq = tmp_path / "toy.parquet"
    toy_df.to_parquet(pq)
    from_df = build_store(toy_df, tmp_path / "a.duckdb", source_label="toy")
    from_pq = build_store(pq, tmp_path / "b.duckdb", source_label="toy")
    assert from_df == from_pq


# ── link-grain edges (the advance-grain blind-spot fix) ──────────────────────


def test_store_without_links_is_advance_source_only(toy_store):
    assert _q(toy_store, "SELECT DISTINCT source FROM edges") == [("advance",)]


def test_link_edges_tagged_screened_and_deduped(toy_store_with_links):
    counts = dict(_q(toy_store_with_links,
                     "SELECT source, count(*) FROM edges GROUP BY 1"))
    # 11 advance edges (unchanged); 5 link rows -> sentinel screened,
    # duplicate deduped -> 3 link edges
    assert counts == {"advance": 11, "link": 3}
    assert _q(toy_store_with_links,
              "SELECT count(*) FROM edges WHERE source = 'link'"
              " AND entity_value = 'none'") == [(0,)]


def test_link_only_user_in_users_with_zero_advances(toy_store_with_links):
    [(n_adv, ict)] = _q(toy_store_with_links,
        "SELECT n_advances, identity_created_time FROM users WHERE user_id = 'uL'")
    assert n_adv == 0
    assert ict == _TS("2026-01-04 08:00")


def test_borrower_identity_created_time_comes_from_advances(toy_store_with_links):
    [(ict,)] = _q(toy_store_with_links,
        "SELECT identity_created_time FROM users WHERE user_id = 'u2'")
    assert ict == _TS("2025-12-15 00:00")


def test_entities_count_link_users(toy_store_with_links):
    [(n_users,)] = _q(toy_store_with_links,
        "SELECT n_users FROM entities WHERE entity_value = 'dH'")
    assert n_users == 6  # 5 borrowers + uL via link edge


def test_unknown_link_entity_type_rejected(toy_df, toy_links, tmp_path):
    bad = toy_links.copy()
    bad.loc[0, "entity_type"] = "carrier_pigeon"
    with pytest.raises(ValueError, match="carrier_pigeon"):
        build_store(toy_df, tmp_path / "bad.duckdb", links=bad)
