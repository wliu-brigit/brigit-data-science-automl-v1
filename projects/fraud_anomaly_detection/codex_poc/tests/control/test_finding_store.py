from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.codex_poc.control.finding_store import FindingStore


def _fs(method="scenario:s", users=("u1",)):
    return FindingSet(method, "v1", [Finding(u) for u in users])


def _empty_fs(method="scenario:s", version="v1"):
    return FindingSet(method, version, [])


def _metadata(method="scenario:s", version="v1"):
    return MethodMetadata(
        name=method,
        version=version,
        method_type="scenario",
        time_semantics="production_safe",
        promotion_tier="plug_candidate",
        enforcement_projection="scenario_rule",
        params={"scenario_name": method.removeprefix("scenario:")},
    )


def test_write_then_read_latest_roundtrips(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot(
        "2026-06-13",
        data_version="v3",
        finding_sets=[_fs(users=["u1", "u2"])],
        method_metadata=[_metadata()],
    )

    latest = store.read_latest()

    assert set(latest["user_id"]) == {"u1", "u2"}
    assert latest.iloc[0]["refresh_key"] == "2026-06-13"
    assert latest.iloc[0]["data_version"] == "v3"
    assert latest.iloc[0]["snapshot_id"] == store.latest_snapshot()["snapshot_id"]


def test_unchanged_snapshot_is_trimmed(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot(
        "2026-06-13",
        data_version="v3",
        finding_sets=[_fs(users=["u1"])],
        method_metadata=[_metadata()],
    )

    wrote = store.write_snapshot(
        "2026-06-14",
        data_version="v3",
        finding_sets=[_fs(users=["u1"])],
        method_metadata=[_metadata()],
    )

    assert wrote is False
    assert store.refresh_keys() == ["2026-06-13"]


def test_changed_snapshot_is_kept(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot(
        "2026-06-13",
        data_version="v3",
        finding_sets=[_fs(users=["u1"])],
        method_metadata=[_metadata()],
    )

    wrote = store.write_snapshot(
        "2026-06-14",
        data_version="v3",
        finding_sets=[_fs(users=["u1", "u2"])],
        method_metadata=[_metadata()],
    )

    assert wrote is True
    assert store.refresh_keys() == ["2026-06-13", "2026-06-14"]


def test_logic_version_change_is_kept_even_when_findings_match(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot(
        "2026-06-13",
        data_version="v3",
        finding_sets=[_fs(users=["u1"])],
        method_metadata=[_metadata()],
    )
    changed_version = FindingSet("scenario:s", "v2", [Finding("u1")])

    wrote = store.write_snapshot(
        "2026-06-14",
        data_version="v3",
        finding_sets=[changed_version],
        method_metadata=[_metadata(version="v2")],
    )

    assert wrote is True
    latest = store.read_latest()
    assert set(latest["method_version"]) == {"v2"}


def test_empty_finding_snapshot_is_recorded_and_trimmed(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")

    wrote = store.write_snapshot(
        "2026-06-13",
        data_version="v3",
        finding_sets=[_empty_fs()],
        method_metadata=[_metadata()],
    )
    wrote_same = store.write_snapshot(
        "2026-06-14",
        data_version="v3",
        finding_sets=[_empty_fs()],
        method_metadata=[_metadata()],
    )

    assert wrote is True
    assert wrote_same is False
    assert store.refresh_keys() == ["2026-06-13"]
    assert store.read_latest().empty


def test_same_refresh_key_reads_only_latest_snapshot(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")
    store.write_snapshot(
        "daily",
        data_version="v3",
        finding_sets=[_fs(users=["u1"])],
        method_metadata=[_metadata()],
    )

    store.write_snapshot(
        "daily",
        data_version="v3",
        finding_sets=[_fs(users=["u2"])],
        method_metadata=[_metadata()],
    )

    assert store.refresh_keys() == ["daily", "daily"]
    assert set(store.read_latest()["user_id"]) == {"u2"}


def test_method_metadata_snapshot_is_persisted(tmp_path):
    store = FindingStore(tmp_path / "findings.duckdb")

    store.write_snapshot(
        "2026-06-13",
        data_version="v3",
        finding_sets=[_fs(users=["u1"])],
        method_metadata=[_metadata()],
    )

    metadata = store.read_latest_method_metadata()
    assert metadata.iloc[0]["method"] == "scenario:s"
    assert metadata.iloc[0]["method_type"] == "scenario"
    assert metadata.iloc[0]["time_semantics"] == "production_safe"
    assert metadata.iloc[0]["promotion_tier"] == "plug_candidate"
    assert metadata.iloc[0]["enforcement_projection"] == "scenario_rule"
    assert metadata.iloc[0]["params_json"] == '{"scenario_name": "s"}'
