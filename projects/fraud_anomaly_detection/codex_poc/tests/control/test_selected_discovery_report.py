from pathlib import Path

import pytest

from projects.fraud_anomaly_detection.codex_poc.control.selected_discovery_report import (
    SelectedReportConfig,
    generate_selected_discovery_report,
)

SAMPLE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample store not built")
def test_selected_discovery_report_is_repeatable(tmp_path):
    report = generate_selected_discovery_report(
        SelectedReportConfig(
            store=SAMPLE,
            out_dir=tmp_path,
            refresh_key="selected_report_test",
        )
    )

    assert Path(report["paths"]["markdown"]).exists()
    assert Path(report["paths"]["json"]).exists()
    assert report["final_discovery"]["scenario_union_users"] == 1024
    assert report["final_discovery"]["final_union_users"] >= 1024
    assert report["plug"]["candidate_keys"] > 0
    assert report["plug"]["burned_keys"] > 0
    assert report["selected_graph_rows"][0]["selected?"] == "yes"
