"""Baseline trial model: faithful replication of the legacy v3 final model.

Implements the legacy Phase-4 recipe end to end, driven by the locked
decisions in data/legacy/experiment_decisions.json:

- the locked 162-feature set (old experiment-phase names resolved to the
  snapshot's columns via the final notebook's rename map)
- WoE-encoded bankinstitution, fit on known (labeled) train rows only
- soft-label reject inference: unknown rows expand into dual records
  (y=1 weighted synthetic_score / y=0 weighted 1-synthetic_score), sampled
  to the locked 80/20 known/unknown ratio with random_state=42
- median imputation for non-payday numerics (±inf coerced to NaN first),
  NaN passthrough for payday features, one-hot highestpayfrequency
- XGBoost with the locked hyperparameters, n_estimators, and monotone
  constraints; random_state=42 / nthread=4 as in the legacy notebook (the
  harness seed argument is deliberately ignored to stay faithful)

Known deviation from the legacy run: the legacy pipeline downsampled
unknowns to 200K server-side (ORDER BY HASH(entity_id)) before the ratio
sample. Snowflake's HASH is not reproducible in pandas, so this model
samples the ratio target directly from all scored unknowns. Counts match;
the exact row draw differs.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder

from automl.model import BaseModel
from projects.neobank_ncm.model.preprocessing import (
    BankInstitutionWOEEncoder,
    PrefitBankInstitutionWOEEncoder,
)

PROJECT_DIR = Path(__file__).resolve().parents[1]
DECISIONS_PATH = PROJECT_DIR / "data" / "legacy" / "experiment_decisions.json"

TARGET = "went_dpd45"
BANK_COL = "bankinstitution"
WOE_COL = "bankinstitution_woe"
SYNTHETIC_COL = "synthetic_score"

# Final-notebook feature-name migration (experiment-phase name -> updated SA
# spec name), keys/values normalized to lower-no-underscore.
FEAT_RENAME = {
    "balancevoltoincome": "balancesdtodailyincomemeanratio",
    "maxnegbaltoincome": "maxnegbalance30dtodailyincomemeanratio",
    "inflowtooutflowratio14d": "inflowsumtooutflowsumratio14d",
    "netflowtoincome": "netflowtodailyincomemeanratio14d",
    "balancedepletionrate": "balancedepletionrate1d",
    "daystopaydaystress": "incomebuffertodaystopaydayratio",
}

# Missing = no detectable pay cycle = informative; never imputed
# (normalized names; incomebuffertodaystopaydayratio inherits daystopayday's NaN).
PAYDAY_NAN_NORMS = {
    "daystopayday",
    "dayssincepayday",
    "daystoregularpayday",
    "dayssinceregularpayday",
    "highestincometotalobservedpaydays",
    "incomebuffertodaystopaydayratio",
}


def _norm(column: str) -> str:
    return str(column).lower().replace("_", "")


class NeobankNCMReplicationModel(BaseModel):
    """The legacy v3 final model, re-expressed in the harness trial contract."""

    name = "neobank_ncm_v3_replication"

    def __init__(self) -> None:
        self.feature_registry = None
        self.preprocessor = None
        self.model = None
        self.woe_encoder_ = None
        self.input_cols_: list[str] = []
        self.missing_features_: list[str] = []

    # ── fit ──────────────────────────────────────────────────────────────
    def fit(self, df_train: pd.DataFrame, registry=None, seed: int = 0):
        del seed  # locked to random_state=42, as in the legacy notebook
        decisions = json.loads(DECISIONS_PATH.read_text())

        by_norm: dict[str, str] = {}
        for column in df_train.columns:
            by_norm.setdefault(_norm(column), str(column))

        target = by_norm.get(_norm(TARGET), TARGET)
        bank_col = by_norm.get(_norm(BANK_COL))
        syn_col = by_norm.get(_norm(SYNTHETIC_COL))

        # 1. resolve the locked feature set against the snapshot's columns
        num_cols: list[str] = []
        self.missing_features_ = []
        cat_cols = [
            by_norm[_norm(c)] for c in decisions.get("cat_feat_cols", []) if _norm(c) in by_norm
        ]
        for feature in decisions["feature_cols"]:
            normalized = FEAT_RENAME.get(_norm(feature), _norm(feature))
            if normalized == _norm(WOE_COL):
                continue  # generated below from bankinstitution
            resolved = by_norm.get(normalized)
            if resolved is None:
                self.missing_features_.append(feature)
            elif resolved not in cat_cols:
                num_cols.append(resolved)

        # 2. WoE: fit on known (labeled) rows only — legacy semantics. The
        # fitted mapping is mounted into the ColumnTransformer below via the
        # prefit variant (the required-transformer contract checks the entry;
        # ColumnTransformer refitting must not see the synthetic labels).
        known = df_train[df_train[target].notna()]
        unknown = df_train[df_train[target].isna()]
        self.bank_col_ = bank_col
        self.woe_encoder_ = BankInstitutionWOEEncoder()
        if bank_col is not None and len(known):
            self.woe_encoder_.fit(known[bank_col], known[target])
        else:
            # degenerate sample (e.g. the runner's 200-row pre-fit check
            # drawing only unknown rows): empty mapping, everything -> 0.0
            self.woe_encoder_.mapping_ = {}
            self.woe_encoder_.other_woe_ = 0.0

        # 3. soft-label dual records at the locked known/unknown ratio
        ratio = float(decisions["winning_ratio"])
        scored = (
            unknown[unknown[syn_col].notna()] if syn_col is not None else unknown.iloc[0:0]
        )
        if len(known) and len(scored):
            n_syn_target = max(1, int(len(known) * (1 - ratio) / ratio))
            if len(scored) > n_syn_target:
                scored = scored.sample(n_syn_target, random_state=42)
        if len(scored):
            pos = scored.copy()
            pos[target] = 1
            neg = scored.copy()
            neg[target] = 0
            train = pd.concat([known, pos, neg], ignore_index=True)
            weights = np.concatenate(
                [
                    np.ones(len(known)),
                    scored[syn_col].to_numpy(dtype=float),
                    1.0 - scored[syn_col].to_numpy(dtype=float),
                ]
            )
        else:
            train = known
            weights = np.ones(len(known))
        if not len(train):
            raise ValueError("no trainable rows: need labeled rows or scored unknowns")

        # 4. preprocessing — WoE entry (required-transformer contract),
        # median imputation, payday NaN passthrough, OHE
        impute_cols = [c for c in num_cols if _norm(c) not in PAYDAY_NAN_NORMS]
        passthrough_cols = [c for c in num_cols if _norm(c) in PAYDAY_NAN_NORMS]
        transformers = []
        if bank_col is not None:
            transformers.append(
                (
                    "neobank_bankinstitution_woe",
                    PrefitBankInstitutionWOEEncoder.from_fitted(self.woe_encoder_),
                    [bank_col],
                )
            )
        transformers += [
            ("num", SimpleImputer(strategy="median", keep_empty_features=True), impute_cols),
            ("pass", "passthrough", passthrough_cols),
        ]
        if cat_cols:
            transformers.append(
                (
                    "ohe",
                    OneHotEncoder(sparse_output=False, handle_unknown="ignore", dtype=np.float32),
                    cat_cols,
                )
            )
        self.preprocessor = ColumnTransformer(
            transformers, remainder="drop", verbose_feature_names_out=False
        ).set_output(transform="pandas")

        X = self.preprocessor.fit_transform(self._prepare(train, num_cols))
        y = train[target].astype(int)

        # 5. monotone constraints over the model columns (OHE columns get 0)
        params = dict(decisions["winning_params"])
        if decisions.get("use_constraints"):
            monotone = {
                FEAT_RENAME.get(_norm(k), _norm(k)): v
                for k, v in decisions.get("monotone_constraints", {}).items()
            }
            params["monotone_constraints"] = tuple(
                monotone.get(_norm(c), 0) for c in self.preprocessor.get_feature_names_out()
            )

        # 6. the locked estimator (nthread pinned: deterministic hist reductions)
        self.model = xgb.XGBClassifier(
            **params,
            n_estimators=int(decisions.get("best_n_estimators", 500)),
            random_state=42,
            nthread=4,
            tree_method="hist",
        )
        self.model.fit(X, y, sample_weight=weights)

        # contract bookkeeping: feature_cols = the consumed raw input columns
        # (the WoE column is generated inside the preprocessor, so the raw
        # bankinstitution is the declared input), and the model registry's
        # model=True set must exactly match it
        self.num_cols_ = num_cols
        seen: set[str] = set()
        self.input_cols_ = [
            c
            for c in (*num_cols, *cat_cols, *([bank_col] if bank_col is not None else []))
            if not (c in seen or seen.add(c))
        ]
        self.feature_cols = list(self.input_cols_)
        self.feature_registry = copy.deepcopy(registry)
        if self.feature_registry is not None:
            flagged = self.feature_registry.get_by_flag("model")
            if flagged:
                self.feature_registry.set_flag(flagged, "model", False)
            self.feature_registry.set_flag(self.feature_cols, "model", True)
        return self

    # ── transform / predict ──────────────────────────────────────────────
    def _prepare(self, df: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
        # WoE is generated inside the preprocessor entry; here only coerce
        # numerics and squash ±inf (legacy impute_df did the same)
        out = df.copy()
        for column in (c for c in num_cols if c in out.columns):
            out[column] = pd.to_numeric(out[column], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
        return out

    def transform(self, df: pd.DataFrame):
        return self.preprocessor.transform(self._prepare(df, self.num_cols_))

    def _predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def feature_importances(self):
        if self.model is None:
            return None
        names = list(self.preprocessor.get_feature_names_out())
        return dict(zip(names, self.model.feature_importances_.tolist(), strict=False))


MODEL_CLASS = NeobankNCMReplicationModel

__all__ = ["MODEL_CLASS", "NeobankNCMReplicationModel"]
