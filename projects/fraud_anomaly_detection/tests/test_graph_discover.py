"""Discovery queues: residual ring members, bad neighbours, farms, pairs, fresh rings."""

import textwrap

import igraph as ig
import pandas as pd
import pytest

pytest.importorskip("duckdb")  # project deps: uv sync --group fraud
pytest.importorskip("igraph")

from projects.fraud_anomaly_detection.graph.discover import (  # noqa: E402
    bad_neighbours,
    emerging_farms,
    fresh_rings,
    multi_witness_pairs,
    residual_ring_members,
    suspicion_queue,
)
from projects.fraud_anomaly_detection.graph.load import load_graph  # noqa: E402

pytestmark = pytest.mark.unit

TOY_REGISTER = """\
version: "test.1"
scenarios:
  - name: big_loan
    tier: review
    status: draft
    entry_date: 2026-06-09
    theory: toy
    trigger:
      - column: loan_amount
        op: ">"
        value: 100
"""


@pytest.fixture()
def toy_graph(toy_store, tmp_path):
    register = tmp_path / "register.yaml"
    register.write_text(textwrap.dedent(TOY_REGISTER))
    return load_graph(toy_store, register_path=register, node_attrs=("is_fraud",))


def test_residual_ring_members_surface_unflagged_co_members(toy_graph):
    out = residual_ring_members(toy_graph, flag="scenario_big_loan",
                                min_users=3, min_types=2)
    # the u1-u2-u3 ring qualifies (3 users, device+bank); u2 is flagged ->
    # the queue holds the unflagged co-members, with the ring evidence.
    assert set(out["user_id"]) == {"u1", "u3"}
    assert (out["ring_flagged"] == 1).all()
    # the 5-user hub ring is single-type -> excluded by min_types.


def test_bad_neighbours_unions_flags_and_excludes_flagged(toy_graph):
    out = bad_neighbours(toy_graph, flags=("scenario_big_loan", "is_fraud"), max_hops=2)
    # u1 is fraud (seed), u2 is scenario-flagged (seed) -> only u3 qualifies:
    # 1 hop from u2 (via b1), 2 hops from u1 (b1-u2-d1).
    assert list(out["user_id"]) == ["u3"]
    row = out.iloc[0]
    assert (row["hops_to_scenario_big_loan"], row["hops_to_is_fraud"]) == (1, 2)


def test_emerging_farms_ranks_by_velocity_and_ignores_covered(toy_store):
    flags = pd.Series(False, index=[f"u{i}" for i in range(1, 10)])
    out = emerging_farms(toy_store, user_flags=flags, min_users=3)
    top = out.iloc[0]
    # dH: 5 users in 4 days = the fastest accumulation, nobody flagged yet.
    assert (top["entity_value"], top["flagged_coverage"]) == ("dH", 0.0)
    fully_flagged = flags.copy()
    fully_flagged[["u5", "u6", "u7", "u8", "u9"]] = True
    out2 = emerging_farms(toy_store, user_flags=fully_flagged, min_users=3)
    assert out2.iloc[0]["flagged_coverage"] == 1.0  # known farm, rank score 0


def test_multi_witness_pairs_requires_two_channels_and_no_flags(tmp_path):
    from projects.fraud_anomaly_detection.graph.build import build_store

    def row(adv, user):
        return {
            "advance_id": adv, "user_id": user,
            "feature_as_of_ts": pd.Timestamp("2026-01-01 10:00"),
            "device_id": "dX", "bank_account_key": None, "persistent_account_id": None,
            "phone_key": "pX", "address_key": None, "email_key": None, "ip_address": None,
            "loan_amount": 50.0, "is_fraud": 0, "label_gross_dpd45": 0,
            "label_mature_d45": 1, "is_neobank_high_risk_institution": 0,
            "expected_dpd45_date": pd.NaT,
        }

    store = tmp_path / "mini.duckdb"
    build_store(pd.DataFrame([row("x1", "ua"), row("x2", "ub")]), store, source_label="mini")
    none_flagged = pd.Series(False, index=["ua", "ub"])
    out = multi_witness_pairs(store, user_flags=none_flagged)
    assert len(out) == 1 and out.iloc[0]["n_types"] == 2  # device AND phone agree
    one_flagged = none_flagged.copy()
    one_flagged["ua"] = True
    assert len(multi_witness_pairs(store, user_flags=one_flagged)) == 0


def test_suspicion_queue_excludes_flagged_and_ranks_neighbours_first(toy_store):
    g = load_graph(toy_store, scenarios=False)
    q = suspicion_queue(g, seed_flag="is_fraud", exclude_flags=("is_fraud",))
    assert "u1" not in set(q.user_id)  # the seed itself is never queued
    assert q.iloc[0].user_id == "u2"  # closest unflagged neighbour tops the queue


def test_suspicion_queue_drops_disconnected_zero_score_users():
    g = ig.Graph()
    g.add_vertices(
        4,
        attributes={
            "name": ["user:u1", "device:d1", "user:u2", "device:d2"],
            "kind": ["user", "device", "user", "device"],
            "raw_id": ["u1", "d1", "u2", "d2"],
            "is_fraud": [True, False, False, False],
        },
    )
    g.add_edges([(0, 1), (2, 3)])

    q = suspicion_queue(g, seed_flag="is_fraud", exclude_flags=("is_fraud",))

    assert q.empty


def test_fresh_rings_window_slices_recent_formation(toy_store):
    # store's latest edge is a04 (Feb 1); 3 days back holds only that lone edge.
    assert len(fresh_rings(toy_store, days=3)) == 0
    out = fresh_rings(toy_store, days=60)  # window reaches back over January
    assert ((out["n_users"] == 3) & (out["n_types"] == 2)).any()  # the u1-u2-u3 ring
