from datetime import UTC, datetime
from pathlib import Path

import pytest

from tests.e2e._gates import requires_live_e2e

from automl.data import materialize
from automl.experiment import compare, leaderboard
from automl.experiment import delete as delete_experiment
from automl.experiment.views.queries import recent_failures, strategies_attempted
from automl.experiment.views.summary import build_summary
from automl.mlflow import client as mlflow_client
from automl.mlflow import trial as mlflow_trial
from automl.project import clear_session, use_project
from automl.runner import run_trial
from automl.trial import TrialStatus, load_model, show_trial
from automl.utils.io import gcs

pytestmark = [pytest.mark.e2e, pytest.mark.qa]


@requires_live_e2e("experiment trial cleanup")
def test_experiment_trial_cleanup_gate():
    repo_root = Path(__file__).resolve().parents[2]
    stamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    namespace = f"qa-experiment-trial-cleanup-{stamp}"
    experiment_id = f"experiment-trial-cleanup-{stamp}"
    active = use_project(
        "example_homecredit",
        repo_root=repo_root,
        namespace=namespace,
        experiment_id=experiment_id,
    )
    try:
        materialize(session=active)
        first = run_trial("example_homecredit", session=active)
        second = run_trial("example_homecredit", session=active)
        assert first.status == TrialStatus.FINISHED.value
        assert second.status == TrialStatus.FINISHED.value

        with pytest.raises(RuntimeError):
            with mlflow_trial.active(
                slug="experiment_trial_cleanup_failed", strategy="forced_failure"
            ):
                raise RuntimeError("experiment trial cleanup forced failure")
        with mlflow_trial.active(slug="experiment_trial_cleanup_unscored", strategy="unscored"):
            pass

        board = leaderboard(session=active, n=10)
        assert board.rows
        assert (
            board.metric
            == f"eval.{active.config.require_run_config().eval_split}.{active.config.primary_metric}"
        )
        assert first.run_id in {row.run_id for row in board.rows}
        assert second.run_id in {row.run_id for row in board.rows}
        assert board.n_unscored >= 2

        comparison = compare([first.run_id, second.run_id], session=active)
        assert comparison.metric_deltas
        assert [run.run_id for run in comparison.runs] == [first.run_id, second.run_id]

        details = show_trial(first.run_id, session=active)
        assert details.evaluations
        assert load_model(first.run_id, session=active) is not None

        summary = build_summary(session=active)
        assert summary["trial_count"] >= 4
        assert recent_failures(session=active)
        assert strategies_attempted(session=active)

        route_prefix = (
            f"gs://{active.config.gcs_bucket}/{active.config.gcs_prefix}/"
            f"{namespace}/{active.project_name}/{experiment_id}/"
        )
        assert gcs.list_blob_names(route_prefix)
        report = delete_experiment(experiment_id, apply=True, session=active)
        assert report.applied is True
        assert gcs.list_blob_names(route_prefix) == []
        assert (
            mlflow_client.raw()
            .get_experiment_by_name(f"{namespace}/{active.project_name}/{experiment_id}")
            .lifecycle_stage
            == "deleted"
        )
    finally:
        clear_session()
