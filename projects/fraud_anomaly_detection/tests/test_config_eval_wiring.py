"""config.py EVAL: all model metrics residual-masked; matched rows only in scenario_identified."""

import pytest

from projects.fraud_anomaly_detection.config import EVAL

pytestmark = pytest.mark.unit


def test_primary_is_residual_gross_dpd45_average_precision():
    # The scenarios absorb the heuristic's top band, so the proxy label is
    # dead in the residual — the primary scores against the real outcome
    # (gross DPD45, the standard early-default ruler).
    assert EVAL.primary_name == "residual_gross_dpd45_average_precision"


def test_all_model_metrics_are_residual_and_scenario_report_is_wired():
    names = [metric.resolved_name() for metric in EVAL.metrics]
    assert names == [
        "residual_gross_dpd45_average_precision",
        "residual_never_paid_average_precision",
        "residual_precision_recall_at_depth",
        "residual_band_report",
        "residual_early_default_capture",
        "scenario_identified",
    ]
