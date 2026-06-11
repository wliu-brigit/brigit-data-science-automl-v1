"""Analysis views: layers, caps, as-of, metadata join, dynamic scenario overlay."""

import textwrap

import pandas as pd
import pytest

pytest.importorskip("duckdb")  # project deps: uv sync --group fraud
pytest.importorskip("igraph")

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


def _users(g):
    return {v["raw_id"] for v in g.vs if v["kind"] == "user"}


def _entities(g):
    return {v["raw_id"] for v in g.vs if v["kind"] != "user"}


def test_default_load_shape(toy_store):
    g = load_graph(toy_store, scenarios=False)
    assert _users(g) == {f"u{i}" for i in range(1, 10)}  # ALL users, singletons too
    assert _entities(g) == {"d1", "d4", "dH", "b1"}
    assert g.ecount() == 11


def test_layer_selection(toy_store):
    g = load_graph(toy_store, layers=("bank",), scenarios=False)
    assert _entities(g) == {"b1"}
    assert g.ecount() == 2


def test_unknown_layer_raises(toy_store):
    with pytest.raises(ValueError, match="unknown layer"):
        load_graph(toy_store, layers=("device", "passport"), scenarios=False)


def test_degree_cap_drops_hub_keeps_users(toy_store):
    g = load_graph(toy_store, degree_cap=4, scenarios=False)
    assert "dH" not in _entities(g)
    assert {"u5", "u6", "u7", "u8", "u9"} <= _users(g)  # users stay (as singletons)


def test_as_of_excludes_future_edges(toy_store):
    g = load_graph(toy_store, as_of=pd.Timestamp("2026-01-15"), scenarios=False)
    etypes = {e["etype"] for e in g.es}
    assert "bank" in etypes
    bank_edges = [e for e in g.es if e["etype"] == "bank"]
    assert len(bank_edges) == 1  # a04 (2026-02-01) excluded


def test_window_slices_edges(toy_store):
    g = load_graph(
        toy_store, layers=("device",),
        window=(pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-09 23:00")),
        scenarios=False,
    )
    assert g.ecount() == 6  # hub edges a06..a10 + the a11 parallel edge


def test_parallel_edges_kept(toy_store):
    g = load_graph(toy_store, scenarios=False)
    u1, d1 = g.vs.find(name="user:u1"), g.vs.find(name="device:d1")
    assert len(g.es.select(_between=([u1.index], [d1.index]))) == 2


def test_node_attrs_joined_from_snapshot(toy_store):
    g = load_graph(toy_store, node_attrs=("is_fraud",), scenarios=False)
    assert g.vs.find(name="user:u1")["is_fraud"] == 1
    assert g.vs.find(name="user:u2")["is_fraud"] == 0


def test_scenario_overlay_matches_engine(toy_store, tmp_path):
    register = tmp_path / "register.yaml"
    register.write_text(textwrap.dedent(TOY_REGISTER))
    g = load_graph(toy_store, register_path=register)
    flagged = {v["raw_id"] for v in g.vs if v["kind"] == "user" and v["scenario_big_loan"]}
    assert flagged == {"u2"}  # only u2 has loan_amount > 100
    assert g.vs.find(name="user:u2")["scenario_any"]


def test_missing_node_attr_raises(toy_store):
    with pytest.raises(ValueError, match="node_attrs not found"):
        load_graph(toy_store, node_attrs=("no_such_column",), scenarios=False)


def test_degree_cap_is_per_view_not_global(toy_store):
    # dH has 5 users globally but only 3 within this window; cap=3 keeps it.
    g = load_graph(
        toy_store, layers=("device",), degree_cap=3,
        window=(pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-07 23:00")),
        scenarios=False,
    )
    assert "dH" in {v["raw_id"] for v in g.vs if v["kind"] != "user"}


def test_custom_base_missing_users_get_falsy_flags(toy_store):
    import duckdb as _duckdb

    with _duckdb.connect(str(toy_store), read_only=True) as con:
        base = con.execute(
            "SELECT * FROM advances WHERE user_id <> 'u9'").df()
    g = load_graph(toy_store, base=base, node_attrs=("is_fraud",), scenarios=False)
    u9 = g.vs.find(name="user:u9")
    assert not bool(u9["is_fraud"])  # absent from base -> falsy, never NaN-truthy
