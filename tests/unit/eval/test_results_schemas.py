import pandas as pd
import pytest

from automl.eval import EvalResult
from automl.eval import results as schemas

pytestmark = pytest.mark.unit


def test_eval_result_uses_report_shape_and_omits_cached():
    result = EvalResult(
        label="external_augmented",
        eval_dataset_id="v1_abcdef12",
        eval_dataset_kind="external",
        predictions_uri="gs://bucket/eval/external_augmented/predictions.parquet",
        predictions_manifest_uri="gs://bucket/eval/external_augmented/predictions.json",
        augmentations_used=({"name": "risk_weight", "hash8": "12345678"},),
        primary="auc",
        metrics=(
            {"name": "auc", "value": 0.91, "augmentations": []},
            {
                "name": "threshold_sweep",
                "value": [{"threshold": 0.5, "precision": 1.0}],
                "augmentations": [],
            },
        ),
        computed_at="2026-05-27T00:00:00+00:00",
        cached=True,
    )

    payload = result.to_dict()
    restored = EvalResult.from_dict({**payload, "future": "ignored"})

    assert "cached" not in payload
    assert payload["metrics"][0]["name"] == "auc"
    assert restored.metrics[1]["value"][0]["threshold"] == 0.5
    assert restored.cached is False


def test_eval_result_constructor_normalizes_metrics_to_tuple():
    result = EvalResult(
        label="holdout",
        eval_dataset_id="ev_1",
        eval_dataset_kind="external",
        predictions_uri="",
        predictions_manifest_uri="",
        augmentations_used=[],
        primary="auc",
        metrics=[{"name": "auc", "value": 1.0, "augmentations": []}],
        computed_at="2026-05-27T00:00:00+00:00",
    )

    assert result.metrics == ({"name": "auc", "value": 1.0, "augmentations": []},)
    assert result.augmentations_used == ()


def test_eval_index_round_trips_entries_and_primary_label():
    index = schemas.EvalIndex(
        primary_label="external_augmented",
        evaluations=(
            schemas.EvalIndexEntry(
                label="external_augmented",
                eval_dataset_id="v1_abcdef12",
                kind="external",
                report_path="eval/external_augmented/results.json",
                eval_dataset_manifest_uri="gs://bucket/eval/datasets/v1_abcdef12/manifest.json",
                predictions_uri="gs://bucket/eval/external_augmented/predictions.parquet",
                predictions_manifest_uri="gs://bucket/eval/external_augmented/predictions.json",
                augmentations_used=({"name": "risk_weight", "hash8": "12345678"},),
                computed_at="2026-05-27T00:00:00+00:00",
            ),
        ),
    )

    restored = schemas.EvalIndex.from_dict({**index.to_dict(), "future": "ignored"})

    assert restored.primary_label == "external_augmented"
    assert restored.evaluations[0].label == "external_augmented"
    assert restored.evaluations[0].augmentations_used == (
        {"name": "risk_weight", "hash8": "12345678"},
    )


def test_predictions_manifest_and_frame_round_trip():
    frame = pd.DataFrame({"row_id": [1, 2], "y_pred": [0.2, 0.8]})
    predictions = schemas.Predictions(
        trial_run_id="run-1",
        eval_dataset_id="v1_abcdef12",
        eval_dataset_kind="external",
        label="external_augmented",
        unique_key=("row_id",),
        frame=frame,
        augmentations_used=({"name": "risk_weight", "hash8": "12345678"},),
        written_at="2026-05-27T00:00:00+00:00",
    )

    manifest = predictions.manifest_dict()
    restored = schemas.Predictions.from_parts({**manifest, "future": "ignored"}, frame)

    assert "frame" not in manifest
    assert "columns" not in manifest
    assert manifest["row_count"] == 2
    assert manifest["unique_key"] == ["row_id"]
    assert restored.frame.equals(frame)
    assert restored.augmentations_used == ({"name": "risk_weight", "hash8": "12345678"},)
