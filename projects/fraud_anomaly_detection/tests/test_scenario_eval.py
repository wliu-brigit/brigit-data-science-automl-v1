"""Scenario-aware eval: ResidualOnly masking and the ScenarioIdentified report.

Model-performance metrics are computed only on rows no scenario matched
(arithmetically as if matched rows were never in the test set); the matched
rows surface exclusively through ScenarioIdentified, as rule outcomes.
"""

import pandas as pd
import pytest

from projects.fraud_anomaly_detection.eval.metrics import (
    PrecisionRecallAtDepth,
    ResidualOnly,
    ScenarioIdentified,
)
from projects.fraud_anomaly_detection.scenarios import SCENARIOS_VERSION
from projects.fraud_anomaly_detection.tests.test_scenarios import make_frame

pytestmark = pytest.mark.unit


@pytest.fixture
def frame():
    """4 rows: 2 matched by draft ring_account_reuse (rows 0-1), 2 residual (rows 2-3)."""
    rows = [
        # matched, mature, never-paid (the rule's catch)
        make_frame(advance_id="m0", label_gross_dpd45=1, label_repaid_current_snapshot=0, label_mature_d45=1, is_fraud=0),
        # matched, mature, repaid (a rule false positive)
        make_frame(advance_id="m1", label_gross_dpd45=1, label_repaid_current_snapshot=1, label_mature_d45=1, is_fraud=0),
        # residual positive, mature never-paid
        make_frame(advance_id="r0", loan_amount=50.0, label_gross_dpd45=1, label_repaid_current_snapshot=0, label_mature_d45=1, is_fraud=1),
        # residual negative, immature
        make_frame(advance_id="r1", prior_advances_on_bank_account_7d=0, label_gross_dpd45=0, label_repaid_current_snapshot=0, label_mature_d45=0, is_fraud=0),
    ]
    return pd.concat(rows, ignore_index=True)


SCORES = [0.9, 0.8, 0.7, 0.6]


def test_residual_only_excludes_matched_rows_from_inner_metric(frame):
    metric = ResidualOnly(PrecisionRecallAtDepth(depths=(0.5,)))
    [record] = metric.compute(frame, SCORES, "is_fraud")
    # population is the 2 residual rows only; top 50% = r0 (score 0.7) -> the positive
    assert record["n_reviewed"] == 1
    assert record["true_positives"] == 1
    assert record["precision"] == 1.0


def test_residual_only_resolved_name_and_required_columns(frame):
    metric = ResidualOnly(PrecisionRecallAtDepth())
    assert metric.resolved_name() == "residual_precision_recall_at_depth"
    # carries the scenario trigger columns so validation checks the frame
    assert "feature_as_of_ts" in metric.required_columns
    assert "identity_created_time" in metric.required_columns


def test_scenario_identified_reports_matched_rows_as_rule_outcomes(frame):
    report = ScenarioIdentified().compute(frame, SCORES, "is_fraud")
    assert report["scenarios_version"] == SCENARIOS_VERSION
    assert report["n_rows"] == 4
    assert report["n_residual"] == 2
    rar = next(s for s in report["scenarios"] if s["name"] == "ring_account_reuse")
    assert rar["name"] == "ring_account_reuse"
    assert rar["status"] == "draft"
    assert rar["tier"] == "block"
    assert rar["n"] == 2
    assert rar["n_mature"] == 2
    assert rar["n_never_paid"] == 1
    assert rar["never_paid_rate"] == pytest.approx(0.5)


def test_scenario_identified_handles_no_matches():
    df = pd.concat(
        [
            make_frame(advance_id="r0", loan_amount=50.0, label_gross_dpd45=0, label_repaid_current_snapshot=0, label_mature_d45=1, is_fraud=0),
        ],
        ignore_index=True,
    )
    report = ScenarioIdentified().compute(df, [0.5], "is_fraud")
    assert report["n_residual"] == 1
    rar = next(s for s in report["scenarios"] if s["name"] == "ring_account_reuse")
    assert rar["n"] == 0
    assert rar["never_paid_rate"] is None
