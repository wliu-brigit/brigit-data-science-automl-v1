import pytest

from automl.experiment.views.types import ComparisonResult, LeaderboardData, MetricDelta
from automl.trial.types import TrialDetails, TrialSummary

pytestmark = pytest.mark.unit


def test_leaderboard_data_from_dict_loads_trial_summaries():
    data = LeaderboardData.from_dict(
        {
            "metric": "test.auc",
            "experiment_id": "baseline",
            "rows": [{"run_id": "run-1", "status": "FINISHED"}],
            "n_unscored": 2,
        }
    )

    assert data.metric == "test.auc"
    assert data.n_unscored == 2
    assert isinstance(data.rows[0], TrialSummary)


def test_comparison_result_from_dict_loads_details_and_metric_deltas():
    result = ComparisonResult.from_dict(
        {
            "run_ids": ["run-a", "run-b"],
            "runs": [{"run_id": "run-a"}, {"run_id": "run-b"}],
            "metric_deltas": [{"metric": "test.auc", "value_a": 0.7, "value_b": 0.8, "delta": 0.1}],
        }
    )

    assert isinstance(result.runs[0], TrialDetails)
    assert result.metric_deltas == (
        MetricDelta(metric="test.auc", value_a=0.7, value_b=0.8, delta=0.1),
    )
