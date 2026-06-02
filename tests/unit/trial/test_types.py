import pytest

from automl.eval import EvalResult
from automl.trial.types import (
    ArtifactRef,
    ParentExperimentRef,
    TrialDetails,
    TrialStatus,
    TrialSummary,
)

pytestmark = pytest.mark.unit


def test_trial_summary_from_dict_normalizes_status_and_unknown_fields():
    summary = TrialSummary.from_dict(
        {
            "schema_version": 1,
            "run_id": "run-1",
            "slug": "baseline",
            "strategy": "logistic",
            "status": "FINISHED",
            "primary_metric_name": "auc",
            "primary_metric_value": 0.71,
            "trial_number": "3",
            "hypothesis": "numeric baseline",
            "training_origin": "automl",
            "training_time_s": "12.5",
            "n_features": "42",
            "ignored_newer_field": "ok",
        }
    )

    assert summary.status is TrialStatus.FINISHED
    assert summary.trial_number == 3
    assert summary.training_time_s == 12.5
    assert summary.n_features == 42


def test_trial_details_from_dict_loads_nested_artifacts_and_eval_results():
    details = TrialDetails.from_dict(
        {
            "run_id": "run-1",
            "status": "FAILED",
            "params": {"C": "1.0"},
            "metrics": {"test.auc": 0.4},
            "tags": {"automl.trial.strategy": "tree"},
            "artifacts": [{"path": "eval/test/results.json", "file_size": 123}],
            "evaluations": [
                {
                    "label": "test",
                    "eval_dataset_id": "eval_abc",
                    "eval_dataset_kind": "split_view",
                    "predictions_uri": "",
                    "predictions_manifest_uri": "",
                    "augmentations_used": [],
                    "primary": "auc",
                    "metrics": [{"name": "auc", "value": 0.4}],
                    "computed_at": "2026-05-28T00:00:00Z",
                }
            ],
        }
    )

    assert details.status is TrialStatus.FAILED
    assert details.artifacts == (ArtifactRef(path="eval/test/results.json", file_size=123),)
    assert isinstance(details.evaluations[0], EvalResult)


def test_trial_details_none_vs_loaded_empty_evaluations():
    cheap = TrialDetails(run_id="run-1", evaluations=None)
    loaded_empty = TrialDetails(run_id="run-1", evaluations=())

    assert cheap.evaluations is None
    assert loaded_empty.evaluations == ()


def test_parent_experiment_ref_round_trips_full_name():
    parent = ParentExperimentRef.from_dict(
        {
            "mlflow_experiment_id": "42",
            "mlflow_experiment_name": "qa/dry_run/home_credit/baseline",
            "dry_run": True,
            "project_name": "home_credit",
            "experiment_id": "baseline",
        }
    )

    assert parent.mlflow_experiment_id == "42"
    assert parent.dry_run is True
