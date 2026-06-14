from pathlib import Path


def test_codex_poc_readme_documents_control_extension_workflow():
    text = Path("projects/fraud_anomaly_detection/codex_poc/README.md").read_text()

    assert "control/discovery/catalog.py" in text
    assert "ScenarioMethod(\"ring_account_reuse\")" in text
    assert "ResidualRingMethod" in text
    assert "run_skeleton" in text
    assert "discovery" in text and "finding_store" in text and "plug" in text and "holdout" in text
    assert "state_a_backtest" in text
    assert "holdout_backtest" in text
    assert "outside_discovery" in text
    assert "reports_db" in text
    assert "method metadata" in text
    assert "promotion_tier" in text
    assert "Disable a method" in text
    assert "selected-discovery report uses reusable selection" in text
