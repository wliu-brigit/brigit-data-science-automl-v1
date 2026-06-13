from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.finding_store import FindingStore


def _fs(method="scenario:s", users=("u1",)):
    return FindingSet(method, "v1", [Finding(u) for u in users])


def test_write_then_read_latest_roundtrips(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot("2026-06-13", data_version="v3", finding_sets=[_fs(users=["u1", "u2"])])

    latest = store.read_latest()

    assert set(latest["user_id"]) == {"u1", "u2"}
    assert latest.iloc[0]["refresh_key"] == "2026-06-13"
    assert latest.iloc[0]["data_version"] == "v3"


def test_unchanged_snapshot_is_trimmed(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot("2026-06-13", data_version="v3", finding_sets=[_fs(users=["u1"])])

    wrote = store.write_snapshot("2026-06-14", data_version="v3", finding_sets=[_fs(users=["u1"])])

    assert wrote is False
    assert store.refresh_keys() == ["2026-06-13"]


def test_changed_snapshot_is_kept(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot("2026-06-13", data_version="v3", finding_sets=[_fs(users=["u1"])])

    wrote = store.write_snapshot("2026-06-14", data_version="v3", finding_sets=[_fs(users=["u1", "u2"])])

    assert wrote is True
    assert store.refresh_keys() == ["2026-06-13", "2026-06-14"]
