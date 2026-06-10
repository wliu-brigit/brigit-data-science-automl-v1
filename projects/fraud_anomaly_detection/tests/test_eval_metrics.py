"""Project-owned fraud_anomaly_detection metrics: depth records, bands, outcomes."""

import numpy as np
import pandas as pd
import pytest

from automl.eval import AveragePrecision, EvalSpec
from projects.fraud_anomaly_detection.eval.metrics import (
    BandReport,
    EarlyDefaultCapture,
    PrecisionRecallAtDepth,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def frame():
    # 10 rows, scores descending by row order: row 0 highest.
    return pd.DataFrame(
        {
            "advance_id": [f"a{i}" for i in range(10)],
            "is_fraud": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            "heuristic_fraud_band": [
                "EXTREMELY_LIKELY",
                "LOW",  # high-scored LOW row -> the discovery queue
                "LIKELY",
                "POSSIBLE",
                "LOW",
                "LOW",
                "LOW",
                "LOW",
                "LOW",
                "LOW",
            ],
            "label_gross_dpd45": [1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
            "label_mature_d45": [1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
        }
    )


SCORES = pd.Series([0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])


def test_precision_recall_at_depth_counts_top_rows():
    df = pd.DataFrame({"is_fraud": [1, 0, 1, 0, 0, 0, 0, 0, 0, 0]})
    records = PrecisionRecallAtDepth(depths=(0.2, 0.5)).compute(df, SCORES, "is_fraud")
    # top 20% = rows 0,1 -> 1 of 2 positives caught
    assert records[0] == {
        "depth": 0.2,
        "n_reviewed": 2,
        "true_positives": 1,
        "precision": 0.5,
        "recall": 0.5,
        "fp_per_tp": 1.0,
    }
    # top 50% = rows 0-4 -> both positives caught
    assert records[1]["precision"] == pytest.approx(0.4)
    assert records[1]["recall"] == pytest.approx(1.0)


def test_depth_floor_is_one_row_and_no_positives_yields_none_recall():
    df = pd.DataFrame({"is_fraud": [0, 0, 0]})
    [record] = PrecisionRecallAtDepth(depths=(0.01,)).compute(df, [0.3, 0.2, 0.1], "is_fraud")
    assert record["n_reviewed"] == 1  # rounding floor: always at least one row
    assert record["recall"] is None
    assert record["fp_per_tp"] is None


def test_depths_must_be_fractions():
    with pytest.raises(ValueError, match="fractions"):
        PrecisionRecallAtDepth(depths=(5,))
    with pytest.raises(ValueError, match="at least one"):
        BandReport(depths=())


def test_band_report_capture_and_percentile(frame):
    records = BandReport(depths=(0.2,)).compute(frame, SCORES, "is_fraud")
    by_band = {record["band"]: record for record in records}
    assert list(by_band) == ["LOW", "POSSIBLE", "LIKELY", "EXTREMELY_LIKELY"]
    # top 20% = rows 0 (EXTREMELY_LIKELY) and 1 (LOW)
    assert by_band["EXTREMELY_LIKELY"]["capture_at_0.2"] == pytest.approx(1.0)
    assert by_band["LOW"]["n_in_top_0.2"] == 1  # the discovery row
    assert by_band["LOW"]["capture_at_0.2"] == pytest.approx(1 / 7)
    assert by_band["LIKELY"]["n_in_top_0.2"] == 0
    # row 0 is the highest score -> 100th percentile band member
    assert by_band["EXTREMELY_LIKELY"]["mean_score_percentile"] == pytest.approx(100.0)


def test_band_report_handles_unexpected_band_values(frame):
    frame = frame.assign(heuristic_fraud_band=["NEW_BAND"] + ["LOW"] * 9)
    records = BandReport(depths=(0.2,)).compute(frame, SCORES, "is_fraud")
    bands = [record["band"] for record in records]
    assert bands == ["LOW", "POSSIBLE", "LIKELY", "EXTREMELY_LIKELY", "NEW_BAND"]
    by_band = {record["band"]: record for record in records}
    assert by_band["POSSIBLE"]["n"] == 0
    assert by_band["POSSIBLE"]["mean_score_percentile"] is None


def test_early_default_capture_restricts_to_mature_rows(frame):
    report = EarlyDefaultCapture(depths=(0.5,)).compute(frame, SCORES, "is_fraud")
    # row 9 (dpd45=1) is immature -> excluded from population and denominator
    assert report["n_mature"] == 9
    assert report["n_dpd45"] == 2
    [record] = report["records"]
    # top 50% of the 9 mature rows = rows 0-4 (round(4.5) -> 4? no: max(1, round(9*0.5)) = 4)
    assert record["n_reviewed"] == 4
    assert record["true_positives"] == 2
    assert record["recall"] == pytest.approx(1.0)


def test_early_default_capture_with_no_mature_rows(frame):
    frame = frame.assign(label_mature_d45=0)
    report = EarlyDefaultCapture().compute(frame, SCORES, "is_fraud")
    assert report == {"n_mature": 0, "n_dpd45": 0, "records": []}


def test_gross_dpd45_average_precision_scores_against_outcome():
    from projects.fraud_anomaly_detection.eval.metrics import GrossDpd45AveragePrecision

    df = pd.DataFrame(
        {
            "label_gross_dpd45": [1, 1, 0, 1],
            "label_mature_d45":  [1, 1, 1, 0],
        }
    )
    # mature rows: 0 (dpd45), 1 (dpd45), 2 (clean); row 3 immature -> excluded.
    # Unlike never-paid, a late-but-repaid row still counts as a DPD45 positive.
    metric = GrossDpd45AveragePrecision()
    assert metric.compute(df, [0.9, 0.5, 0.1, 0.99], "is_fraud") == pytest.approx(1.0)
    # ranking a clean row above a dpd45 row drops AP below 1
    assert metric.compute(df, [0.1, 0.5, 0.9, 0.99], "is_fraud") < 1.0


def test_gross_dpd45_average_precision_no_positives_returns_zero():
    from projects.fraud_anomaly_detection.eval.metrics import GrossDpd45AveragePrecision

    df = pd.DataFrame({"label_gross_dpd45": [0, 0], "label_mature_d45": [1, 1]})
    assert GrossDpd45AveragePrecision().compute(df, [0.9, 0.1], "is_fraud") == 0.0


def test_never_paid_average_precision_scores_against_outcome():
    from projects.fraud_anomaly_detection.eval.metrics import NeverPaidAveragePrecision

    df = pd.DataFrame(
        {
            "label_gross_dpd45":            [1, 1, 0, 1],
            "label_repaid_current_snapshot":[0, 1, 0, 0],
            "label_mature_d45":             [1, 1, 1, 0],
        }
    )
    # mature rows: 0 (never-paid), 1 (late-repaid -> negative), 2 (clean);
    # row 3 never-paid but immature -> excluded entirely
    metric = NeverPaidAveragePrecision()
    assert metric.compute(df, [0.9, 0.5, 0.1, 0.99], "is_fraud") == pytest.approx(1.0)
    # ranking the never-paid row last drops AP below 1
    assert metric.compute(df, [0.1, 0.5, 0.9, 0.99], "is_fraud") < 1.0


def test_never_paid_average_precision_no_positives_returns_zero():
    from projects.fraud_anomaly_detection.eval.metrics import NeverPaidAveragePrecision

    df = pd.DataFrame(
        {
            "label_gross_dpd45": [0, 0],
            "label_repaid_current_snapshot": [0, 0],
            "label_mature_d45": [1, 1],
        }
    )
    assert NeverPaidAveragePrecision().compute(df, [0.9, 0.1], "is_fraud") == 0.0


def test_full_eval_spec_integration(frame):
    spec = EvalSpec(
        primary=AveragePrecision(),
        metrics=[PrecisionRecallAtDepth(depths=(0.2,)), BandReport(depths=(0.2,)), EarlyDefaultCapture(depths=(0.5,))],
    )
    report = spec.evaluate(frame, SCORES, "is_fraud")
    assert report["primary"] == "average_precision"
    names = [record["name"] for record in report["metrics"]]
    assert names == [
        "average_precision",
        "precision_recall_at_depth",
        "band_report",
        "early_default_capture",
    ]
    assert isinstance(report["metrics"][0]["value"], float)
