from pathlib import Path


def test_neo4j_codex_readme_documents_control_extension_workflow():
    text = Path("projects/fraud_anomaly_detection/neo4j_codex/README.md").read_text()

    assert "control_loop_report" in text
    assert "run_skeleton" not in text
    assert "scenario" in text and "graph" in text and "plug" in text and "holdout" in text
    assert "review_only" in text
    assert "promoted_to_plug_derivation" in text
    assert "outside_discovery" in text
    assert "method metadata" in text
    assert "promotion_tier" in text
    assert "--include-status" in text
