from pathlib import Path

import duckdb
import pandas as pd
import pytest

from projects.fraud_anomaly_detection.neo4j_codex.control.control_loop_report import (
    ControlLoopReportConfig,
    _graph_row,
    generate_control_loop_report,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.selection import (
    SelectionRow,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.graph.methods import (
    Neo4jGraphDiscovery,
)

SAMPLE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")
RING_ACCOUNT_REUSE_GRAPH_USERS = [
    "1f3d302e-f10e-4b5d-ae57-af3749e616b0",
    "4498c60a-2e3d-44ee-a55c-b876b549e6a2",
    "71dea5e8-6b06-40e5-8033-3d9876aafd61",
    "7b9cf08f-ee09-48f1-b250-4e72289ca06b",
    "7fcefe39-633d-4c24-87d0-030592930304",
    "8158e022-0486-4869-9a4c-40380c1c62ec",
    "8d700dd3-4add-4bcb-a1b0-138b986c5a3d",
    "9b3830b4-4504-4c40-a9d7-b2f3a4afca90",
    "a98d7a82-bead-4e8e-aa87-49024a7a6d87",
    "c668acfb-7030-4654-9dd6-aab5a8b1f17d",
    "cabaa79f-4581-4407-bac0-60c21b590cd3",
    "ce5c9955-8613-4580-b2a2-7f6cff0e66b6",
    "e813fedc-d47e-41a5-b66c-369ae9e05af1",
]


class FakeNeo4jRunner:
    def __init__(
        self,
        rows_by_method: dict[str, list[dict]],
        *,
        state_rows_by_method: dict[str, list[dict]] | None = None,
    ) -> None:
        self.rows_by_method = rows_by_method
        self.state_rows_by_method = state_rows_by_method or {}

    def run(self, query: str, params: dict) -> list[dict]:
        rows = self.state_rows_by_method if params.get("as_of") else self.rows_by_method
        return rows.get(str(params["method_name"]), [])


def fake_graph_discovery(
    rows_by_method: dict[str, list[dict]],
    *,
    state_rows_by_method: dict[str, list[dict]] | None = None,
) -> Neo4jGraphDiscovery:
    return Neo4jGraphDiscovery(
        runner=FakeNeo4jRunner(
            rows_by_method,
            state_rows_by_method=state_rows_by_method,
        )
    )


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
        "status": "review_only",
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


def test_control_loop_report_runs_on_tiny_store(tiny_store, tmp_path):
    report = generate_control_loop_report(
        ControlLoopReportConfig(
            store=tiny_store,
            out_dir=tmp_path,
            refresh_key="tiny_selected_report",
            graph_min_marginal_users=1,
        ),
        graph_discovery=fake_graph_discovery(
            {"graph:scenario_neighborhood:ring_account_reuse": [{"user_id": "u3"}]}
        ),
    )

    assert Path(report["paths"]["markdown"]).exists()
    assert Path(report["paths"]["json"]).exists()
    assert report["scenario_rows"][0]["scenario method"].startswith("scenario:")
    assert report["scenario_rows"][0]["method version"] == report["scenario_version"]
    assert report["graph_status_counts"]["review_only"] > 0
    assert report["review_graph_net_new_users"] > 0
    assert {row["status"] for row in report["graph_rows"]} == {"review_only"}
    assert report["selected_graph_rows"] == []
    assert any(row["reason"] == "promotion_tier" for row in report["excluded_graph_rows"])


def test_control_loop_report_filters_graph_rows_by_status(tiny_store, tmp_path):
    report = generate_control_loop_report(
        ControlLoopReportConfig(
            store=tiny_store,
            out_dir=tmp_path,
            refresh_key="tiny_selected_report",
            graph_min_marginal_users=1,
            include_statuses=frozenset({"promoted_to_plug_derivation"}),
        ),
        graph_discovery=fake_graph_discovery(
            {"graph:scenario_neighborhood:ring_account_reuse": [{"user_id": "u3"}]}
        ),
    )

    assert report["graph_status_counts"]["review_only"] > 0
    assert report["graph_rows"] == []
    assert report["selected_graph_rows"] == []
    assert report["excluded_graph_rows"] == []


def test_control_loop_report_state_a_uses_asof_discovery(tmp_path):
    store = tmp_path / "leak.duckdb"
    advances = pd.DataFrame(
        {
            "advance_id": ["a1", "a2", "a3", "a4"],
            "user_id": ["u1", "u2", "u_future", "u_future"],
            "is_fraud": [False, False, False, False],
            "label_gross_dpd45": [True, True, False, True],
            "label_mature_d45": [True, True, True, True],
            "feature_as_of_ts": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2026-01-05", "2026-03-01"]
            ),
            "identity_created_time": pd.to_datetime(
                ["2026-01-01", "2026-01-02", "2025-12-01", "2026-03-01"]
            ),
            "loan_amount": [150.0, 120.0, 50.0, 200.0],
            "prior_advances_on_bank_account_7d": [1, 1, 0, 1],
            "users_on_bank_account_72h": [2, 2, 0, 0],
            "users_on_persistent_account_id_72h": [2, 2, 0, 0],
            "is_joint": [0, 0, 0, 0],
            "users_on_device_id_72h": [0, 0, 0, 0],
        }
    )
    edges = pd.DataFrame(
        {
            "advance_id": ["a1", "a2", "a3", "a4"],
            "user_id": ["u1", "u2", "u_future", "u_future"],
            "entity_type": ["bank", "bank", "bank", "bank"],
            "entity_value": ["acctA", "acctA", "acctB", "acctB"],
            "ts": advances["feature_as_of_ts"],
            "source": ["advance"] * 4,
        }
    )
    with duckdb.connect(str(store)) as con:
        con.register("advances_df", advances)
        con.register("edges_df", edges)
        con.execute("CREATE TABLE advances AS SELECT * FROM advances_df")
        con.execute("CREATE TABLE edges AS SELECT * FROM edges_df")
        con.execute(
            """
            CREATE TABLE users AS
            SELECT DISTINCT CAST(user_id AS VARCHAR) AS user_id
            FROM advances_df
            """
        )

    report = generate_control_loop_report(
        ControlLoopReportConfig(
            store=store,
            out_dir=tmp_path,
            refresh_key="leak_report",
            graph_min_marginal_users=1,
        ),
        graph_discovery=fake_graph_discovery({}),
    )

    assert report["final_discovery"]["final_union_users"] == 3
    assert report["final_discovery"]["state_a_final_union_users"] == 2


@pytest.mark.skipif(not SAMPLE.exists(), reason="sample store not built")
def test_control_loop_report_is_repeatable(tmp_path):
    report = generate_control_loop_report(
        ControlLoopReportConfig(
            store=SAMPLE,
            out_dir=tmp_path,
            refresh_key="selected_report_test",
        ),
        graph_discovery=fake_graph_discovery(
            {
                "graph:scenario_neighborhood:ring_account_reuse": [
                    {"user_id": user_id} for user_id in RING_ACCOUNT_REUSE_GRAPH_USERS
                ]
            }
        ),
    )

    assert Path(report["paths"]["markdown"]).exists()
    assert Path(report["paths"]["json"]).exists()
    assert report["final_discovery"]["scenario_union_users"] == 1024
    assert report["final_discovery"]["final_union_users"] == 1024
    assert report["plug"]["candidate_keys"] > 0
    assert report["plug"]["burned_keys"] > 0
    assert report["review_graph_net_new_users"] > 0
    assert report["graph_status_counts"]["review_only"] > 0
    assert any(
        row["graph method"] == "graph:scenario_neighborhood:ring_account_reuse"
        and row["status"] == "review_only"
        and row["net-new beyond scenarios / DPD45"] == "13 / 84.6%"
        for row in report["graph_rows"]
    )
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
