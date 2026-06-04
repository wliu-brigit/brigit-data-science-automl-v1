from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.e2e._gates import requires_live_e2e

from automl.data import load_dataset_by_id, materialize
from automl.eval import (
    Auc,
    EvalSpec,
    LogLoss,
    Metric,
    ThresholdSweep,
    evaluate,
    prepare_eval_augmentation,
    prepare_eval_dataset,
)
from automl.mlflow import client as mlflow_client
from automl.mlflow import tags
from automl.mlflow.trial import artifacts
from automl.project import clear_session, use_project
from automl.runner import run_trial
from automl.trial import TrialStatus
from automl.utils.io import gcs

pytestmark = [pytest.mark.e2e, pytest.mark.qa]


class WeightedMeanScore(Metric):
    name = "weighted_mean_score"
    required_columns = ("risk_weight",)
    required_augmentations = ("risk_weight",)

    def compute(self, df, y_pred, target_col):
        del target_col
        return float((df["risk_weight"] * y_pred).mean())


@requires_live_e2e("eval dataset")
def test_eval_dataset_breadth_external_gate():
    repo_root = Path(__file__).resolve().parents[2]
    active = use_project("example_homecredit", repo_root=repo_root)
    try:
        loaded = materialize(session=active)
        result = run_trial("example_homecredit", session=active)

        assert result.status == TrialStatus.FINISHED.value
        assert result.run_id

        eval_slice = load_dataset_by_id(
            loaded.id,
            split_name=active.config.require_run_config().eval_split,
            session=active,
        )
        external_frame = eval_slice.df.reset_index(drop=True).copy()
        eval_dataset, _ = prepare_eval_dataset(
            session=active,
            kind="external",
            frame=external_frame,
            target_col=active.config.target_column,
            unique_key=loaded.dataset.unique_key,
            provenance={"source": "eval_dataset_e2e"},
        )
        unique_key = list(loaded.dataset.unique_key)
        augmentation_frame = external_frame.loc[:, unique_key].copy()
        augmentation_frame["risk_weight"] = (
            pd.Series(range(len(augmentation_frame)), dtype="float64") + 1.0
        )
        augmentation, _ = prepare_eval_augmentation(
            session=active,
            eval_dataset_id=eval_dataset.id,
            frame=augmentation_frame,
            name="risk_weight",
        )

        external_result = evaluate(
            session=active,
            model_run_id=result.run_id,
            eval_dataset_id=eval_dataset.id,
            eval_spec=EvalSpec(
                primary=Auc(),
                metrics=[
                    -LogLoss(),
                    ThresholdSweep(thresholds=[0.3, 0.5, 0.7]),
                    WeightedMeanScore(),
                ],
            ),
            label="external_augmented",
            set_as_primary_label=True,
        )
        run = mlflow_client.raw().get_run(result.run_id)

        assert external_result.eval_dataset_kind == "external"
        assert artifacts.load_eval(result.run_id, "external_augmented") == external_result
        assert artifacts.load_predictions(result.run_id, "external_augmented").frame.shape[
            0
        ] == len(external_frame)
        assert ("external_augmented", eval_dataset.id) in artifacts.list_eval(result.run_id)
        index = artifacts.load_eval_index(result.run_id)
        assert index.primary_label == "external_augmented"
        assert "external_augmented" in [entry.label for entry in index.evaluations]
        assert run.data.metrics["eval.external_augmented.auc"] >= 0.0
        assert "eval.external_augmented.negative_log_loss" in run.data.metrics
        assert "eval.external_augmented.weighted_mean_score" in run.data.metrics
        assert "eval.external_augmented.threshold_sweep" not in run.data.metrics
        assert "auc" not in run.data.metrics
        assert any(record["name"] == "threshold_sweep" for record in external_result.metrics)
        for uri in (
            eval_dataset.manifest_gcs_uri,
            eval_dataset.data_gcs_uri,
            augmentation.manifest_gcs_uri,
            augmentation.data_gcs_uri,
            run.data.tags[tags.eval_predictions_uri("external_augmented")],
        ):
            assert uri is not None
            assert uri.startswith("gs://")
            assert gcs.blob_exists(uri)
        _assert_trial_artifact_exists(
            result.run_id,
            run.data.tags[tags.eval_uri("external_augmented")],
        )
        _assert_trial_artifact_exists(
            result.run_id,
            run.data.tags[tags.eval_predictions_manifest_uri("external_augmented")],
        )
    finally:
        clear_session()


def _assert_trial_artifact_exists(run_id: str, artifact_path: str) -> None:
    local_path = mlflow_client.raw().download_artifacts(run_id, artifact_path)
    assert Path(local_path).exists()
