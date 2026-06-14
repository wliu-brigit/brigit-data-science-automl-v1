import pandas as pd

from projects.fraud_anomaly_detection.codex_poc.control.plug_fact_store import (
    PlugFactStore,
)


def _facts():
    return pd.DataFrame(
        [
            {
                "entity_type": "bank",
                "entity_value": "acctA",
                "support": 3,
                "mature_users": 3,
                "bad_users": 2,
                "coverage": 2,
                "dpd45_precision": 2 / 3,
                "innocents": 1,
            }
        ]
    )


def test_plug_fact_store_roundtrips_latest_snapshot(tmp_path):
    store = PlugFactStore(tmp_path / "facts.duckdb")

    snapshot_id = store.write_snapshot(
        "run-1",
        data_version="sample",
        selection_users=pd.Series(["u2", "u1"]),
        facts=_facts(),
    )

    latest = store.latest_snapshot()
    facts = store.read_latest()
    assert latest["snapshot_id"] == snapshot_id
    assert latest["refresh_key"] == "run-1"
    assert facts.iloc[0]["snapshot_id"] == snapshot_id
    assert facts.iloc[0]["entity_type"] == "bank"
    assert facts.iloc[0]["coverage"] == 2


def test_plug_fact_store_requires_fact_contract(tmp_path):
    store = PlugFactStore(tmp_path / "facts.duckdb")

    try:
        store.write_snapshot(
            "run-1",
            data_version="sample",
            selection_users=pd.Series(["u1"]),
            facts=pd.DataFrame([{"entity_type": "bank"}]),
        )
    except KeyError as exc:
        assert "entity_value" in str(exc)
    else:
        raise AssertionError("expected malformed candidate facts to fail")
