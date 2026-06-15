import importlib.util


def test_legacy_run_skeleton_module_is_removed():
    assert (
        importlib.util.find_spec("projects.fraud_anomaly_detection.neo4j_codex.control.run")
        is None
    )
