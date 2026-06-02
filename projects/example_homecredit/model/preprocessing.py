from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class WOEEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, smoothing: float = 0.5) -> None:
        self.smoothing = smoothing

    def fit(self, X, y):
        series = _as_series(X)
        target = pd.Series(y).astype(float).reset_index(drop=True)
        categories = series.fillna("__missing__").astype(str).reset_index(drop=True)
        global_rate = _smoothed_rate(target.sum(), len(target), self.smoothing)
        self.fallback_ = _logit(global_rate)
        self.mapping_: dict[str, float] = {}
        for category, indices in categories.groupby(categories).groups.items():
            values = target.iloc[list(indices)]
            rate = _smoothed_rate(values.sum(), len(values), self.smoothing)
            self.mapping_[str(category)] = _logit(rate)
        return self

    def transform(self, X):
        series = _as_series(X).fillna("__missing__").astype(str)
        encoded = series.map(self.mapping_).fillna(self.fallback_).astype(float)
        return encoded.to_numpy().reshape(-1, 1)


def _as_series(X) -> pd.Series:
    if isinstance(X, pd.Series):
        return X
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError("WOEEncoder expects exactly one input column")
        return X.iloc[:, 0]
    array = np.asarray(X)
    if array.ndim == 2:
        if array.shape[1] != 1:
            raise ValueError("WOEEncoder expects exactly one input column")
        array = array[:, 0]
    return pd.Series(array)


def _smoothed_rate(events: float, total: int, smoothing: float) -> float:
    return float((events + smoothing) / (total + 2 * smoothing))


def _logit(value: float) -> float:
    clipped = min(max(value, 1e-6), 1.0 - 1e-6)
    return math.log(clipped / (1.0 - clipped))


__all__ = ["WOEEncoder"]
