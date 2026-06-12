"""Codex Neo4j mirror POC: import bundle should serve analyst workflows."""

from __future__ import annotations

from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")

from projects.fraud_anomaly_detection.codex_poc import export_neo4j_mirror as mirror


SAMPLE_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")
POC_DIR = Path("projects/fraud_anomaly_detection/codex_poc")


def test_ids_are_namespaced_for_neo4j_import() -> None:
    assert mirror.user_node_id("u1") == "user:u1"
    assert mirror.entity_node_id("bank", "abc") == "entity:bank:abc"
    assert mirror.scenario_node_id("ring_identity_burst") == "scenario:ring_identity_burst"


def test_sample_export_bundle_contains_graph_scenarios_and_user_story(tmp_path: Path) -> None:
    result = mirror.export_bundle(SAMPLE_STORE, tmp_path, max_edges=250)

    expected = {
        "users.csv",
        "entities.csv",
        "clusters.csv",
        "scenarios.csv",
        "used_device_rels.csv",
        "used_bank_account_rels.csv",
        "cluster_member_rels.csv",
        "scenario_match_rels.csv",
        "summary.md",
    }
    assert expected.issubset({p.name for p in result.files})
    assert not (tmp_path / "touched_rels.csv").exists()

    users_header = (tmp_path / "users.csv").read_text().splitlines()[0]
    assert "userNodeId:ID(User-ID)" in users_header
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

    clusters_header = (tmp_path / "clusters.csv").read_text().splitlines()[0]
    assert "clusterNodeId:ID(Cluster-ID)" in clusters_header
    assert "dpd45_user_rate:float" in clusters_header

    members = (tmp_path / "cluster_member_rels.csv").read_text().splitlines()
    assert members[0] == ":START_ID(User-ID),:END_ID(Cluster-ID),member:boolean"
    assert len(members) > 1

    summary = (tmp_path / "summary.md").read_text()
    assert "User Story" in summary
    assert "Suspicious Clusters -> Ring" in summary
    assert "Performance Gate" in summary


def test_cypher_playbook_covers_cluster_scenario_ego_and_gds_workflows(tmp_path: Path) -> None:
    mirror.write_cypher_playbook(tmp_path)

    files = {p.name for p in (tmp_path / "cypher").iterdir()}
    assert {
        "00_top_suspicious_clusters.cypher",
        "01_cluster_ring_view.cypher",
        "01_scenario_overview.cypher",
        "02_scenario_to_users.cypher",
        "03_user_ego_ring.cypher",
        "04_gds_component_and_pagerank.cypher",
        "05_entity_drilldown.cypher",
        "06_discovery_candidates.cypher",
    }.issubset(files)

    cluster_query = (tmp_path / "cypher" / "00_top_suspicious_clusters.cypher").read_text()
    assert "ReviewCluster" in cluster_query
    assert "dpd45_user_rate" in cluster_query

    cluster_ring = (tmp_path / "cypher" / "01_cluster_ring_view.cypher").read_text()
    assert "IN_REVIEW_CLUSTER" in cluster_ring
    assert "USED_DEVICE" in cluster_ring

    scenario_query = (tmp_path / "cypher" / "02_scenario_to_users.cypher").read_text()
    assert "neighbor_users" in scenario_query
    assert "fraud_neighbors" in scenario_query

    ego_query = (tmp_path / "cypher" / "03_user_ego_ring.cypher").read_text()
    assert "MATCH path =" in ego_query
    assert "USED_BANK_ACCOUNT" in ego_query

    gds_query = (tmp_path / "cypher" / "04_gds_component_and_pagerank.cypher").read_text()
    assert "gds.wcc" in gds_query
    assert "gds.pageRank" in gds_query

    entity_query = (tmp_path / "cypher" / "05_entity_drilldown.cypher").read_text()
    assert "$entity_type" in entity_query
    assert "$entity_value" in entity_query

    discovery_query = (tmp_path / "cypher" / "06_discovery_candidates.cypher").read_text()
    assert "candidate_user" in discovery_query
    assert "shared_entity_types" in discovery_query


def test_docker_setup_script_documents_rebuild_import_and_start() -> None:
    script = (POC_DIR / "scripts" / "setup_neo4j.sh").read_text()

    assert "export_neo4j_mirror" in script
    assert "neo4j-admin database import full" in script
    assert "--nodes=User=" in script
    assert "--nodes=ReviewCluster=" in script
    assert "--relationships=USED_DEVICE=" in script
    assert "--relationships=USED_BANK_ACCOUNT=" in script
    assert "--relationships=IN_REVIEW_CLUSTER=" in script
    assert "docker run" in script
    assert "7474:7474" in script


def test_how_to_use_guide_explains_relationship_meanings() -> None:
    guide = (POC_DIR / "HOW_TO_USE_NEO4J.md").read_text()

    assert "MATCHED_SCENARIO" in guide
    assert "USED_DEVICE" in guide
    assert "Suspicious Clusters -> Ring" in guide
    assert "Discovery" in guide
