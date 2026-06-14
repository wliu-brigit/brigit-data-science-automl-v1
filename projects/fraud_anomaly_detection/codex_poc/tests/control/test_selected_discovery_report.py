from pathlib import Path

import pytest

from projects.fraud_anomaly_detection.codex_poc.control.selected_discovery_report import (
    SelectedReportConfig,
    _graph_row,
    generate_selected_discovery_report,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.selection import (
    SelectionRow,
)

SAMPLE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")


def test_graph_row_exposes_metadata_without_sample_store():
    metadata = MethodMetadata(
        name="graph:test",
        version="v1",
        method_type="graph",
        time_semantics="snapshot_review",
        promotion_tier="review_queue",
        enforcement_projection="entity_key",
        params={"display_name": "test"},
    )
    row = _graph_row(
        SelectionRow(
            name="graph:test",
            users=frozenset({"u1"}),
            total={"users": 1, "dpd45_user_rate": 0.5},
            net_new_users=frozenset({"u1"}),
            net={"users": 1, "dpd45_user_rate": 0.5},
            marginal_users=frozenset({"u1"}),
            marginal={"users": 1, "dpd45_user_rate": 0.5},
            selected=False,
            reason="promotion_tier",
            metadata=metadata,
        )
    )

    assert row == {
        "graph method": "graph:test",
        "display name": "test",
        "method version": "v1",
        "method type": "graph",
        "time semantics": "snapshot_review",
        "promotion tier": "review_queue",
        "enforcement projection": "entity_key",
        "total users / DPD45": "1 / 50.0%",
        "net-new beyond scenarios / DPD45": "1 / 50.0%",
        "marginal after dedupe / DPD45": "1 / 50.0%",
        "selected?": "no",
        "reason": "promotion_tier",
    }


def test_selected_discovery_report_runs_on_tiny_store(tiny_store, tmp_path):
    report = generate_selected_discovery_report(
        SelectedReportConfig(
            store=tiny_store,
            out_dir=tmp_path,
            refresh_key="tiny_selected_report",
            graph_min_marginal_users=1,
        )
    )

    assert Path(report["paths"]["markdown"]).exists()
    assert Path(report["paths"]["json"]).exists()
    assert report["scenario_rows"][0]["scenario method"].startswith("scenario:")
    assert report["scenario_rows"][0]["method version"] == report["scenario_version"]
    assert report["selected_graph_rows"] == []
    assert any(row["reason"] == "promotion_tier" for row in report["excluded_graph_rows"])


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
    assert report["final_discovery"]["final_union_users"] == 1024
    assert report["plug"]["candidate_keys"] > 0
    assert report["plug"]["burned_keys"] > 0
    assert report["selected_graph_rows"] == []
    assert report["scenario_rows"][0]["scenario method"].startswith("scenario:")
    assert report["scenario_rows"][0]["method version"] == report["scenario_version"]
    assert report["scenario_rows"][0]["time semantics"] == "production_safe"
    assert "candidate_facts" in report["plug"]
    assert "reason" in report["excluded_graph_rows"][0]
    assert any(
        row["reason"] == "promotion_tier"
        for row in report["excluded_graph_rows"]
    )
