from projects.fraud_anomaly_detection.neo4j_codex.control.report_store import ReportStore


def test_report_store_roundtrips_latest_daily_report(tmp_path):
    store = ReportStore(tmp_path / "reports.duckdb")
    report = {"discovery": {"union": {"n_users": 3}}, "plug": {"burned_key_count": 1}}

    store.write_report("2026-06-13", data_version="sample", report=report)

    latest = store.read_latest()
    assert latest["refresh_key"] == "2026-06-13"
    assert latest["data_version"] == "sample"
    assert latest["report"] == report


def test_report_store_reads_latest_inserted_report(tmp_path):
    store = ReportStore(tmp_path / "reports.duckdb")

    store.write_report("2026-06-13", data_version="sample", report={"run": 1})
    store.write_report("2026-06-14", data_version="sample", report={"run": 2})

    latest = store.read_latest()
    assert latest["refresh_key"] == "2026-06-14"
    assert latest["report"] == {"run": 2}
