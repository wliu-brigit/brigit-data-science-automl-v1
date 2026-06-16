from pathlib import Path


def test_neo4j_codex_readme_documents_control_extension_workflow():
    text = Path("projects/fraud_anomaly_detection/neo4j_codex/README.md").read_text()

    assert "control_loop_report" in text
    assert "run_skeleton" not in text
    assert "neo4j_mirror/scripts/setup_neo4j.sh" in text
    assert "neo4j_codex/archived" not in text
    assert "scenario" in text and "graph" in text and "plug" in text and "holdout" in text
    assert "review_only" in text
    assert "promoted_to_plug_derivation" in text
    assert "outside_discovery" in text
    assert "method metadata" in text
    assert "promotion_tier" in text
    assert "--include-status" in text
    assert "uv run --with neo4j --group fraud" in text
    assert "NEO4J_PASSWORD" in text
    assert "control/graph/methods.py" in text
    assert "graph.load" not in text
    assert "graph.discover" not in text


def test_handsoff_points_to_active_neo4j_mirror_setup():
    text = Path("docs/handsoff.md").read_text()

    assert "neo4j_mirror/scripts/setup_neo4j.sh" in text
    assert "neo4j_codex/archived/scripts/setup_neo4j.sh" not in text
    assert "neo4j_mirror/" in text
