"""Query helpers: proximity, components, rings, projection, hub report."""

import pytest

pytest.importorskip("duckdb")  # project deps: uv sync --group fraud
pytest.importorskip("igraph")

from projects.fraud_anomaly_detection.graph.load import load_graph  # noqa: E402
from projects.fraud_anomaly_detection.graph.queries import (  # noqa: E402
    components,
    hub_report,
    near_flagged,
    ppr_suspicion,
    project_users,
    ring,
)

pytestmark = pytest.mark.unit


@pytest.fixture()
def toy_graph(toy_store):
    return load_graph(toy_store, node_attrs=("is_fraud",), scenarios=False)


def test_near_flagged_user_hops(toy_graph):
    out = near_flagged(toy_graph, flag="is_fraud", max_hops=3)
    hops = dict(zip(out["user_id"], out["hops"]))
    # u1 is the only fraud seed. u2 shares d1 (1 user-hop); u3 reaches u1
    # via b1-u2-d1 (2 user-hops). u4..u9 are unreachable -> absent.
    assert hops == {"u2": 1, "u3": 2}
    assert dict(zip(out["user_id"], out["nearest_flagged"])) == {"u2": "u1", "u3": "u1"}


def test_near_flagged_respects_max_hops(toy_graph):
    out = near_flagged(toy_graph, flag="is_fraud", max_hops=1)
    assert set(out["user_id"]) == {"u2"}


def test_components_census(toy_graph):
    out = components(toy_graph, flag="is_fraud")
    by_size = out.sort_values("n_users", ascending=False).reset_index(drop=True)
    assert by_size.loc[0, "n_users"] == 5            # hub component (dH)
    assert by_size.loc[0, "n_types"] == 1
    ring_row = by_size[by_size["n_users"] == 3].iloc[0]  # u1-u2-u3 via d1+b1
    assert ring_row["n_types"] == 2                  # multi-type: the v1 discriminator
    assert ring_row["n_flagged"] == 1                # u1
    assert (by_size["n_users"] == 1).sum() == 1      # u4+d4


def test_ring_extraction(toy_graph):
    sub = ring(toy_graph, "u1", hops=1)
    names = set(sub.vs["name"])
    assert names == {"user:u1", "device:d1", "user:u2"}


def test_project_users_weights(toy_store):
    out = project_users(toy_store, degree_cap=None)
    pairs = {tuple(sorted((a, b))): (s, t) for a, b, s, t
             in zip(out["user_a"], out["user_b"], out["n_shared"], out["n_types"])}
    assert pairs[("u1", "u2")] == (1, 1)   # share d1 only
    assert pairs[("u2", "u3")] == (1, 1)   # share b1 only
    assert len([p for p in pairs if "u5" in p or "u6" in p]) == 4 + 3  # hub pairs for u5/u6


def test_project_users_cap_kills_hub_pairs(toy_store):
    out = project_users(toy_store, degree_cap=4)
    users_in_pairs = set(out["user_a"]) | set(out["user_b"])
    assert users_in_pairs == {"u1", "u2", "u3"}


def test_hub_report_orders_and_annotates(toy_store):
    out = hub_report(toy_store, top_n=3)
    top = out.iloc[0]
    assert (top["entity_value"], top["n_users"]) == ("dH", 5)
    assert top["fraud_user_rate"] == 0.0
    d1 = out[out["entity_value"] == "d1"].iloc[0]
    assert d1["fraud_user_rate"] == 0.5  # u1 fraud, u2 not


def test_ppr_suspicion_ranks_by_proximity_to_seeds(toy_store):
    g = load_graph(toy_store, scenarios=False)
    out = ppr_suspicion(g, flag="is_fraud")
    s = out.set_index("user_id")["score"]
    # u2 shares d1 with seed u1; u3 reaches u1 only through u2-b1; u4 sits in a
    # disconnected component — suspicion must decay along that ordering
    assert s["u2"] > s["u3"] > s["u4"]
    assert bool(out.set_index("user_id")["seeded"]["u1"])
    assert not bool(out.set_index("user_id")["seeded"]["u2"])


def test_ppr_suspicion_no_seeds_returns_empty(toy_store):
    g = load_graph(toy_store, scenarios=False)
    out = ppr_suspicion(g, flag="is_neobank_high_risk_institution")  # all zero
    assert len(out) == 0


def test_project_users_counts_multi_entity_weights(tmp_path):
    import pandas as pd

    from projects.fraud_anomaly_detection.graph.build import build_store

    rows = []
    for adv, dev, phone in [("x1", "dX", "pX"), ("x2", "dX", "pX")]:
        rows.append({
            "advance_id": adv, "user_id": "ua" if adv == "x1" else "ub",
            "feature_as_of_ts": pd.Timestamp("2026-01-01 10:00"),
            "device_id": dev, "bank_account_key": None,
            "persistent_account_id": None, "phone_key": phone,
            "address_key": None, "email_key": None, "ip_address": None,
            "loan_amount": 50.0, "is_fraud": 0, "label_gross_dpd45": 0,
            "label_mature_d45": 1, "is_neobank_high_risk_institution": 0,
        })
    store = tmp_path / "mini.duckdb"
    build_store(pd.DataFrame(rows), store, source_label="mini")
    out = project_users(store, degree_cap=None)
    [(a, b, n_shared, n_types)] = list(
        zip(out["user_a"], out["user_b"], out["n_shared"], out["n_types"]))
    assert {a, b} == {"ua", "ub"}
    assert (n_shared, n_types) == (2, 2)  # share device dX AND phone pX


def test_project_users_rejects_bad_layers(toy_store):
    with pytest.raises(ValueError, match="at least one"):
        project_users(toy_store, layers=())
    with pytest.raises(ValueError, match="unknown layer"):
        hub_report(toy_store, layers=("device", "passport"))
