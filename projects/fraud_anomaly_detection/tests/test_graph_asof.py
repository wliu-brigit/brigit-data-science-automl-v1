"""Leak-free event-ordered features: strictly-prior reads, seed maturity, self-exclusion."""

import pandas as pd
import pytest

pytest.importorskip("duckdb")  # project deps: uv sync --group fraud
pytest.importorskip("igraph")

from projects.fraud_anomaly_detection.graph.asof import leakfree_features  # noqa: E402

pytestmark = pytest.mark.unit


def _by_advance(out):
    return out.set_index("advance_id")


def test_reads_are_strictly_prior(toy_store):
    out = _by_advance(leakfree_features(toy_store, degree_cap=None))
    # a02 (u2 on d1, Jan 2): at that moment the graph holds only a01 (u1-d1).
    # Read happens BEFORE a02's own edge is added: component is {u1,d1} plus
    # the incoming u2 -> 2 users, 1 type; u1's seed has NOT matured yet (Jan 11).
    row = out.loc["a02"]
    assert (row.comp_users, row.comp_types, row.nb_comp, row.nb_d1) == (2, 1, 0, 0)


def test_seed_activates_at_maturity_not_at_event(toy_store):
    out = _by_advance(leakfree_features(toy_store, degree_cap=None))
    # a04 (u3 on b1, Feb 1): component is u1-d1-u2-b1 (2 users, device+bank),
    # u3 incoming -> 3 users / 2 types. u1's bad outcome matured Jan 11 < Feb 1,
    # so ONE other known-bad user sits in the ring (nb_comp). u1 never touched
    # b1 directly, so no distance-1 seed (nb_d1).
    row = out.loc["a04"]
    assert (row.comp_users, row.comp_types, row.nb_comp, row.nb_d1) == (3, 2, 1, 0)


def test_own_prior_default_is_excluded(tmp_path):
    from projects.fraud_anomaly_detection.graph.build import build_store

    def row(adv, user, ts, fraud=0):
        return {
            "advance_id": adv, "user_id": user, "feature_as_of_ts": pd.Timestamp(ts),
            "device_id": "dZ", "bank_account_key": None, "persistent_account_id": None,
            "phone_key": None, "address_key": None, "email_key": None, "ip_address": None,
            "loan_amount": 50.0, "is_fraud": fraud, "label_gross_dpd45": fraud,
            "label_mature_d45": 1, "is_neobank_high_risk_institution": 0,
            "expected_dpd45_date": (pd.Timestamp(ts) + pd.Timedelta(days=10))
            if fraud else pd.NaT,
        }

    df = pd.DataFrame([
        row("x1", "uA", "2026-01-01 10:00", fraud=1),  # matures Jan 11
        row("x2", "uA", "2026-01-20 10:00"),           # own seed must NOT count
        row("x3", "uB", "2026-01-20 11:00"),           # uA's seed DOES count
    ])
    store = tmp_path / "mini.duckdb"
    build_store(df, store, source_label="mini")
    out = _by_advance(leakfree_features(store, degree_cap=None))
    assert (out.loc["x2"].nb_comp, out.loc["x2"].nb_d1) == (0, 0)  # self-excluded
    assert (out.loc["x3"].nb_comp, out.loc["x3"].nb_d1) == (1, 1)  # ring signal


def test_degree_cap_screens_hub_from_the_replay(toy_store):
    out = _by_advance(leakfree_features(toy_store, degree_cap=4))
    # dH carries 5 users -> capped out of the view; hub advances see no graph.
    row = out.loc["a07"]
    assert (row.comp_users, row.comp_types) == (1, 0)
