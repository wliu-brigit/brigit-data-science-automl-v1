"""Project decision metrics for the native re-eval (see decision-metric-vocabulary.md)."""
from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from automl.eval import EvalSpec, Metric
from projects.neobank_ncm.analysis import policy, report

# columns the metrics need present in the eval frame (validated by evaluate())
_REQUIRED = (
    "user_id", "day_number", "is_known", "synthetic_score", "v2_score",
    "account_approval_state", "dailyincomemean", "highestpaydepositmean",
    "noactivityrate", policy.PLAID_INFLOW_30D, "loan_amount_max",
    "underwriting_strategy", "first_activation_date",
)


class Day2KnownAuc(Metric):
    name = "day2_known_auc"
    required_columns = ("day_number", "is_known", "went_dpd45")

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> float:
        mask = (df["day_number"] == 2) & df["is_known"] & df[target_col].notna()
        y_pred_s = pd.Series(y_pred, index=df.index)
        return float(roc_auc_score(df.loc[mask, target_col].astype(int), y_pred_s[mask]))


class DecisionReport(Metric):
    name = "decision_report"
    required_columns = _REQUIRED

    def __init__(self, *, headline_scenario: int = 2, provenance: dict | None = None) -> None:
        self._headline = headline_scenario
        self._provenance = provenance

    def compute(self, df: pd.DataFrame, y_pred: Any, target_col: str) -> dict:
        # target_col unused — the decision report always evaluates against went_dpd45
        scored = df.copy()
        scored["v3_score"] = pd.Series(y_pred, index=df.index).to_numpy()
        return report.build_decision_report(
            scored, headline_scenario=self._headline, provenance=self._provenance
        )


def decision_eval_spec(*, headline_scenario: int = 2, provenance: dict | None = None) -> EvalSpec:
    return EvalSpec(
        primary=Day2KnownAuc(),
        metrics=[DecisionReport(headline_scenario=headline_scenario, provenance=provenance)],
    )
