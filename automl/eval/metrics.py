"""Built-in metrics for the eval thin path."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss as sklearn_log_loss
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from automl.eval.base import Metric


class Auc(Metric):
    name = "auc"

    def compute(self, df: pd.DataFrame, y_pred, target_col: str) -> float:
        return float(roc_auc_score(df[target_col], y_pred))


class LogLoss(Metric):
    name = "log_loss"

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> float:
        return float(sklearn_log_loss(df[target_col], y_pred))


class ThresholdSweep(Metric):
    name = "threshold_sweep"

    def __init__(self, *, thresholds: Sequence[float]) -> None:
        if not thresholds:
            raise ValueError("ThresholdSweep requires at least one threshold")
        self.thresholds = tuple(float(threshold) for threshold in thresholds)

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> list[dict[str, float]]:
        y_true = df[target_col]
        scores = np.asarray(y_pred, dtype=float)
        records = []
        for threshold in self.thresholds:
            y_hat = (scores >= threshold).astype(int)
            records.append(
                {
                    "threshold": threshold,
                    "precision": float(precision_score(y_true, y_hat, zero_division=0)),
                    "recall": float(recall_score(y_true, y_hat, zero_division=0)),
                }
            )
        return records


__all__ = ["Auc", "LogLoss", "ThresholdSweep"]
