import pandas as pd
import pytest

import automl.eval.base as eval_base
import automl.eval.metrics as builtins
from automl.eval import Auc, EvalSpec
from automl.eval.base import Metric

pytestmark = pytest.mark.unit


class NeedsWeight(Metric):
    name = "weighted_mean_score"
    required_columns = ("risk_weight",)
    required_augmentations = ("risk_weight",)

    def compute(self, df, y_pred, target_col):
        del target_col
        return float((df["risk_weight"] * y_pred).mean())


def test_metric_alias_sign_required_augmentations_and_scalar_records():
    df = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "target": [0, 0, 1, 1],
        }
    )
    augmentation = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "risk_weight": [1.0, 2.0, 3.0, 4.0],
        }
    )
    y_pred = pd.Series([0.1, 0.2, 0.8, 0.9])

    spec = EvalSpec(primary={"custom_auc": Auc()}, metrics=[-builtins.LogLoss(), NeedsWeight()])
    report = spec.evaluate(
        df,
        y_pred,
        "target",
        augmentation_frames={"risk_weight": augmentation},
        unique_key=("row_id",),
    )

    assert spec.metrics[0].resolved_name() == "custom_auc"
    assert spec.primary_name == "custom_auc"
    assert spec.required_columns() == ("risk_weight",)
    assert spec.required_augmentations() == ("risk_weight",)
    assert report["primary"] == "custom_auc"
    assert report["metrics"] == [
        {"name": "custom_auc", "value": pytest.approx(1.0), "augmentations": []},
        {
            "name": "negative_log_loss",
            "value": pytest.approx(-0.164252033486018),
            "augmentations": [],
        },
        {
            "name": "weighted_mean_score",
            "value": pytest.approx(1.625),
            "augmentations": ["risk_weight"],
        },
    ]

    assert eval_base.scalar_metric_records(report) == {
        "custom_auc": pytest.approx(1.0),
        "negative_log_loss": pytest.approx(-0.164252033486018),
        "weighted_mean_score": pytest.approx(1.625),
    }


def test_threshold_sweep_persists_non_scalar_but_is_not_scalar_metric():
    df = pd.DataFrame({"target": [0, 0, 1, 1]})
    y_pred = pd.Series([0.1, 0.4, 0.6, 0.9])

    report = EvalSpec(
        primary=Auc(),
        metrics=[builtins.ThresholdSweep(thresholds=[0.3, 0.5, 0.7])],
    ).evaluate(df, y_pred, "target")

    assert report["metrics"][1] == {
        "name": "threshold_sweep",
        "value": [
            {"threshold": 0.3, "precision": pytest.approx(2 / 3), "recall": pytest.approx(1.0)},
            {"threshold": 0.5, "precision": pytest.approx(1.0), "recall": pytest.approx(1.0)},
            {"threshold": 0.7, "precision": pytest.approx(1.0), "recall": pytest.approx(0.5)},
        ],
        "augmentations": [],
    }
    assert eval_base.scalar_metric_records(report) == {"auc": pytest.approx(1.0)}
    assert not eval_base.is_scalar_value(report["metrics"][1]["value"])


def test_eval_spec_rejects_duplicate_resolved_names_and_non_scalar_primary():
    with pytest.raises(ValueError, match="duplicate metric"):
        EvalSpec(primary={"score": Auc()}, metrics=[{"score": builtins.LogLoss()}])

    with pytest.raises(ValueError, match="primary metric must be a finite scalar"):
        EvalSpec(primary=builtins.ThresholdSweep(thresholds=[0.5])).evaluate(
            pd.DataFrame({"target": [0, 1]}),
            pd.Series([0.2, 0.8]),
            "target",
        )


def test_eval_spec_rejects_missing_required_augmentation_even_if_column_exists():
    frame = pd.DataFrame({"row_id": [1, 2], "target": [0, 1], "risk_weight": [1.0, 2.0]})

    with pytest.raises(ValueError, match="required augmentations missing"):
        EvalSpec(primary=Auc(), metrics=[NeedsWeight()]).evaluate(
            frame,
            pd.Series([0.2, 0.8]),
            "target",
            unique_key=("row_id",),
        )


def test_threshold_sweep_requires_thresholds():
    with pytest.raises(ValueError, match="at least one threshold"):
        builtins.ThresholdSweep(thresholds=[])


def test_average_precision_as_primary_metric():
    # Perfect ranking → AP = 1.0 regardless of prevalence.
    df = pd.DataFrame({"target": [0, 0, 0, 1]})
    perfect = pd.Series([0.1, 0.2, 0.3, 0.9])
    report = EvalSpec(primary=builtins.AveragePrecision()).evaluate(df, perfect, "target")
    assert report["primary"] == "average_precision"
    assert report["metrics"][0]["value"] == pytest.approx(1.0)

    # Uninformative constant score → AP equals base prevalence (sklearn ties → 0.25),
    # unlike ROC-AUC which would sit at 0.5. This is the property we pick it for.
    constant = pd.Series([0.5, 0.5, 0.5, 0.5])
    report = EvalSpec(primary=builtins.AveragePrecision()).evaluate(df, constant, "target")
    assert report["metrics"][0]["value"] == pytest.approx(0.25)
