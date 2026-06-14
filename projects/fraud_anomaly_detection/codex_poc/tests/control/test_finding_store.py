from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.finding_store import FindingStore


def _fs(method="scenario:s", users=("u1",)):
    return FindingSet(method, "v1", [Finding(u) for u in users])


def _empty_fs(method="scenario:s", version="v1"):
    return FindingSet(method, version, [])


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


def test_logic_version_change_is_kept_even_when_findings_match(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot("2026-06-13", data_version="v3", finding_sets=[_fs(users=["u1"])])
    changed_version = FindingSet("scenario:s", "v2", [Finding("u1")])

    wrote = store.write_snapshot("2026-06-14", data_version="v3", finding_sets=[changed_version])

    assert wrote is True
    latest = store.read_latest()
    assert set(latest["method_version"]) == {"v2"}


def test_empty_finding_snapshot_is_recorded_and_trimmed(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")

    wrote = store.write_snapshot(
        "2026-06-13",
        data_version="v3",
        finding_sets=[_empty_fs()],
    )
    wrote_same = store.write_snapshot(
        "2026-06-14",
        data_version="v3",
        finding_sets=[_empty_fs()],
    )

    assert wrote is True
    assert wrote_same is False
    assert store.refresh_keys() == ["2026-06-13"]
    assert store.read_latest().empty
