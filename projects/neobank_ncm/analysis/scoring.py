"""Daily-grain feature engineering + chunked model scoring.

Port of cells 4, 5 and 9 of the legacy financial_impact_analysis.ipynb.
The derived-feature formulas are the locked base-table ones with ONE
deliberate difference, copied from the legacy: ``istaxseason`` anchors to
the daily ``snapshot_date``, not the origination date.

Two model adapters score the daily frame:

- ``LegacyArtifactsModel`` — parity mode: replays the production v3 scoring
  path from the legacy artifacts folder (WoE json + transformer.pkl +
  neobank_ncm_model_v3.json).
- ``TrialModel`` — harness mode: any MLflow-logged trial model (pyfunc);
  our models embed their preprocessing, so the adapter only feeds them the
  daily frame with the derived columns added.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCORE_CHUNK = 250_000  # legacy cell 9: bound the transform matrix in RAM

DERIVED_COLS = [
    "balancesdtodailyincomemeanratio",
    "maxnegbalance30dtodailyincomemeanratio",
    "inflowsumtooutflowsumratio14d",
    "netflowtodailyincomemeanratio14d",
    "balancedepletionrate1d",
    "incomebuffertodaystopaydayratio",
    "competitorborrowintensity",
    "istaxseason",
]


def add_daily_derived_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Derived features on the daily grain, in place (legacy cell 5)."""
    d = daily
    eps = 1e-6
    income = d["dailyincomemean"].abs().clip(lower=eps)

    d["balancesdtodailyincomemeanratio"] = d["balancesd"] / income
    d["maxnegbalance30dtodailyincomemeanratio"] = (
        d["maxnegativebalpast30days"].abs() / income
    )

    outflow = d["outflowsum14d"].abs()
    inflow = d["inflowsum14d"].abs()
    d["inflowsumtooutflowsumratio14d"] = inflow / outflow.clip(lower=eps)
    d["netflowtodailyincomemeanratio14d"] = (inflow - outflow) / (income * 14)

    d["balancedepletionrate1d"] = (
        d["balancemeanafterpayday1"] - d["balancemeanafterpayday0"]
    ) / d["highestpaydepositmean"].abs().clip(lower=eps)

    balance_to_income = d["balancemean"] / income
    d["incomebuffertodaystopaydayratio"] = balance_to_income / d["daystopayday"].clip(
        lower=1
    )

    comp = (
        d["davesummarycreditninetydayamount"].fillna(0)
        + d["earninsummarycreditninetydayamount"].fillna(0)
        + d["othercompetitorsummarycreditninetydayamount"].fillna(0)
    )
    d["competitorborrowintensity"] = comp / (income * 90)

    # snapshot-date anchored — the daily pipeline's deliberate difference
    month = pd.to_datetime(d["snapshot_date"]).dt.month
    d["istaxseason"] = month.isin([2, 3, 4]).astype(int)
    return d


def calibration_table(daily: pd.DataFrame) -> pd.DataFrame:
    """Synthetic-score calibration by decile, D2 known users (legacy cell 7).

    Day 2 only, so each user appears once. For a well-calibrated RI model
    mean(synthetic_score) tracks the actual bad rate per decile.
    """
    cal = daily[
        daily["is_known"] & daily["synthetic_score"].notna() & (daily["day_number"] == 2)
    ].copy()
    cal["went_dpd45"] = cal["went_dpd45"].astype(float)
    cal["synthetic_score"] = cal["synthetic_score"].astype(float)
    cal["decile"] = (
        pd.qcut(cal["synthetic_score"], q=10, labels=False, duplicates="drop") + 1
    )
    table = (
        cal.groupby("decile")
        .agg(
            n=("went_dpd45", "count"),
            mean_syn_score=("synthetic_score", "mean"),
            actual_br=("went_dpd45", "mean"),
        )
        .reset_index()
    )
    table["delta"] = table["mean_syn_score"] - table["actual_br"]
    return table


def d2_known_auc(daily: pd.DataFrame) -> dict[str, float]:
    """v3-score AUC on D2 known users (legacy cell 10 sanity check)."""
    from sklearn.metrics import roc_auc_score

    d2 = daily[(daily["day_number"] == 2) & daily["is_known"]].dropna(
        subset=["went_dpd45"]
    )
    return {
        "d2_auc": float(roc_auc_score(d2["went_dpd45"].astype(int), d2["v3_score"])),
        "d2_n": int(len(d2)),
        "d2_bad_rate": float(d2["went_dpd45"].astype(float).mean()),
    }


class LegacyArtifactsModel:
    """The production v3 scoring path, from the legacy artifacts folder."""

    def __init__(self, artifacts_dir: str | Path) -> None:
        import pickle

        import xgboost as xgb

        artifacts = Path(artifacts_dir)
        self.woe_dict = json.loads((artifacts / "bankinstitutionwoe.json").read_text())
        meta = json.loads((artifacts / "features.json").read_text())
        self.feature_cols = (
            meta["NUMERICAL_FEATURES"]
            + meta["CATEGORICAL_FEATURES"]
            + meta["BOOL_FEATURES"]
        )
        with open(artifacts / "transformer.pkl", "rb") as fh:
            self.preprocessor = pickle.load(fh)
        self.model = xgb.XGBClassifier()
        self.model.load_model(str(artifacts / "neobank_ncm_model_v3.json"))

    def score_chunk(self, chunk: pd.DataFrame) -> np.ndarray:
        # legacy normalize_columns: feature columns to UPPER-no-underscore
        feat_set = set(self.feature_cols)
        upper = chunk.rename(
            columns=lambda c: c.upper().replace("_", "")
            if c.upper().replace("_", "") in feat_set
            else c.upper()
        ).copy()
        other = self.woe_dict["OTHER"]
        upper["BANKINSTITUTIONWOE"] = (
            upper["BANKINSTITUTION"]
            .map(lambda x: self.woe_dict.get(x, other) if not pd.isna(x) else other)
            .astype(float)
        )
        X = self.preprocessor.transform(upper)
        return self.model.predict_proba(X)[:, 1]


class TrialModel:
    """An MLflow-logged trial model from this harness.

    Loaded through the harness MLflow seam (which resolves the logged-model
    URI), so it needs a bound session — call inside ``use_project(...)`` /
    ``automl.mlflow.bound_for(...)``.
    """

    def __init__(self, run_id: str) -> None:
        from automl.mlflow.trial.artifacts import load_model

        self.run_id = run_id
        self.model = load_model(run_id)

    def score_chunk(self, chunk: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.model.predict(chunk), dtype=float).reshape(-1)


def score_daily(
    daily: pd.DataFrame,
    model,
    *,
    chunk_size: int = SCORE_CHUNK,
    drop_derived: bool = True,
) -> pd.DataFrame:
    """Add ``v3_score`` to the daily frame (legacy cell 9 semantics).

    Derived features are added in place first, the model scores in chunks,
    and (like the legacy) the derived columns are dropped afterwards unless
    ``drop_derived=False``.
    """
    add_daily_derived_features(daily)
    scores = np.empty(len(daily), dtype=np.float32)
    for start in range(0, len(daily), chunk_size):
        rows = slice(start, start + chunk_size)
        scores[rows] = model.score_chunk(daily.iloc[rows])
    daily["v3_score"] = scores
    if drop_derived:
        daily.drop(
            columns=[c for c in DERIVED_COLS if c in daily.columns], inplace=True
        )
        gc.collect()
    return daily
