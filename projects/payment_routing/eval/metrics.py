from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from automl.eval import Metric


class ProjectedRevenue(Metric):
    """Average expected routing value from score-weighted amount minus cost."""

    def __init__(self, *, amount_col: str, cost_col: str) -> None:
        self.amount_col = amount_col
        self.cost_col = cost_col
        self.required_columns = (amount_col, cost_col)

    def compute(self, df_test: pd.DataFrame, y_pred: Any, target_col: str) -> float:  # noqa: ARG002
        scores = np.asarray(y_pred, dtype=float)
        amount = df_test[self.amount_col].astype(float).to_numpy()
        cost = df_test[self.cost_col].astype(float).to_numpy()
        return float(np.mean(scores * amount - cost))
