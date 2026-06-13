import pandas as pd

from projects.fraud_anomaly_detection.codex_poc.control import monitoring


def test_holdout_effect_counts_prevention_and_leakage(tiny_store):
    burned = pd.DataFrame({"entity_type": ["bank"], "entity_value": ["acctA"]})
    held = ["u3", "u4"]

    report = monitoring.holdout_effect(tiny_store, burned, held)

    assert report["prevented_bad"] == 1
    assert report["innocents_blocked"] == 0
    assert report["leaked_bad"] == 0
