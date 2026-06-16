"""Neo4j mirror export should support the active control-report workflow."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pytest

from projects.fraud_anomaly_detection.neo4j_codex.neo4j_mirror import export as mirror


SAMPLE_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")
MIRROR_DIR = Path("projects/fraud_anomaly_detection/neo4j_codex/neo4j_mirror")


def test_ids_are_namespaced_for_neo4j_import() -> None:
    assert mirror.user_node_id("u1") == "user:u1"
    assert mirror.entity_node_id("bank", "abc") == "entity:bank:abc"
    assert mirror.scenario_node_id("ring_identity_burst") == "scenario:ring_identity_burst"


def test_sample_export_bundle_contains_active_graph_report_shape(tmp_path: Path) -> None:
    result = mirror.export_bundle(SAMPLE_STORE, tmp_path, max_edges=250)

    expected = {
        "users.csv",
        "entities.csv",
        "scenarios.csv",
        "used_device_rels.csv",
        "used_bank_account_rels.csv",
        "used_persistent_account_rels.csv",
        "used_phone_rels.csv",
        "used_address_rels.csv",
        "scenario_match_rels.csv",
        "summary.md",
        "neo4j_admin_import.sh",
    }
    assert expected.issubset({p.name for p in result.files})
    assert not (tmp_path / "clusters.csv").exists()
    assert not (tmp_path / "cluster_member_rels.csv").exists()

    users_header = (tmp_path / "users.csv").read_text().splitlines()[0]
    assert "userNodeId:ID(User-ID)" in users_header
    assert "first_seen_ts" in users_header
    assert "scenario_any:boolean" in users_header
    assert "is_fraud:boolean" in users_header
    users_csv = (tmp_path / "users.csv").read_text()
    assert "False" not in users_csv
    assert "True" not in users_csv
    assert "true" in users_csv

    rels = (tmp_path / "scenario_match_rels.csv").read_text().splitlines()
    assert rels[0] == ":START_ID(User-ID),:END_ID(Scenario-ID),matched:boolean"
    assert len(rels) > 1

    device_rels = (tmp_path / "used_device_rels.csv").read_text().splitlines()
    assert device_rels[0].endswith("last_ts:localdatetime")
    assert "n_events:int" in device_rels[0]
    assert "T" in device_rels[1].split(",")[-1]

    summary = (tmp_path / "summary.md").read_text()
    assert "control report" in summary
    assert "ReviewCluster" not in summary


def test_user_nodes_carry_advance_outcome_counts(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    mirror.export_bundle(SAMPLE_STORE, tmp_path, max_edges=250)

    users = pd.read_csv(tmp_path / "users.csv")
    assert "n_mature_advances:int" in users.columns
    assert "n_bad_advances:int" in users.columns

    with duckdb.connect(str(SAMPLE_STORE), read_only=True) as con:
        n_mature, n_bad = con.execute(
            """
            SELECT count(*) FILTER (WHERE label_mature_d45),
                   count(*) FILTER (WHERE label_mature_d45 AND label_gross_dpd45)
            FROM advances
            """
        ).fetchone()
    assert users["n_mature_advances:int"].sum() == n_mature
    assert users["n_bad_advances:int"].sum() == n_bad
    assert (users["n_bad_advances:int"] <= users["n_mature_advances:int"]).all()
    assert (users["n_mature_advances:int"] <= users["n_advances:int"]).all()

    bad_flag = users["label_gross_dpd45:boolean"].astype(str).str.lower() == "true"
    assert (bad_flag == (users["n_bad_advances:int"] > 0)).all()


def test_user_rate_stays_queryable_without_baking_strict_dpd45_definition(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    mirror.export_bundle(SAMPLE_STORE, tmp_path, max_edges=250)

    users = pd.read_csv(tmp_path / "users.csv")
    assert "bad_advance_rate:float" in users.columns
    observable = users[users["n_mature_advances:int"] > 0]
    expected = observable["n_bad_advances:int"] / observable["n_mature_advances:int"]
    assert (observable["bad_advance_rate:float"] - expected).abs().max() < 1e-5


def test_cypher_playbook_covers_scenario_ego_entity_and_gds_workflows(tmp_path: Path) -> None:
    mirror.write_cypher_playbook(tmp_path)

    files = {p.name for p in (tmp_path / "cypher").iterdir()}
    assert {
        "00_scenario_overview.cypher",
        "01_scenario_neighborhood.cypher",
        "02_user_ego_ring.cypher",
        "03_entity_drilldown.cypher",
        "04_gds_component_and_pagerank.cypher",
    }.issubset(files)

    scenario_query = (tmp_path / "cypher" / "01_scenario_neighborhood.cypher").read_text()
    assert "MATCHED_SCENARIO" in scenario_query
    assert "candidate_user" in scenario_query

    ego_query = (tmp_path / "cypher" / "02_user_ego_ring.cypher").read_text()
    assert "MATCH path =" in ego_query
    assert "USED_BANK_ACCOUNT" in ego_query

    entity_query = (tmp_path / "cypher" / "03_entity_drilldown.cypher").read_text()
    assert "$entity_type" in entity_query
    assert "$entity_value" in entity_query

    gds_query = (tmp_path / "cypher" / "04_gds_component_and_pagerank.cypher").read_text()
    assert "gds.wcc" in gds_query
    assert "gds.pageRank" in gds_query


def test_docker_setup_script_documents_rebuild_import_and_start() -> None:
    script = (MIRROR_DIR / "scripts" / "setup_neo4j.sh").read_text()

    assert "neo4j_mirror.export" in script
    assert "neo4j-admin database import full" in script
    assert "--nodes=User=" in script
    assert "--nodes=Scenario=" in script
    assert "--nodes=ReviewCluster=" not in script
    assert "--relationships=USED_DEVICE=" in script
    assert "--relationships=USED_BANK_ACCOUNT=" in script
    assert "--relationships=MATCHED_SCENARIO=" in script
    assert "--relationships=IN_REVIEW_CLUSTER=" not in script
    assert "docker run" in script
    assert "7474:7474" in script


def test_active_mirror_export_does_not_import_python_graph_backend() -> None:
    text = (MIRROR_DIR / "export.py").read_text()

    assert "igraph" not in text
    assert "projects.fraud_anomaly_detection.graph.load" not in text
    assert "projects.fraud_anomaly_detection.graph.discover" not in text

