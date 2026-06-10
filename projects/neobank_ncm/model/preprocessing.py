"""Project-owned preprocessing: the legacy bankinstitution WoE encoder.

Faithful port of fit_woe/apply_woe from the legacy notebooks
(neobank_ncm_model_v3_final.ipynb; same logic in the experiment notebook):

- WoE = log(dist_good / dist_bad) with 0.5 Laplace smoothing on each
  distribution; HIGHER WoE = safer bank.
- Only banks with >= min_obs (30) observations get their own value; sparse,
  unseen, and missing banks all resolve to OTHER.
- OTHER inherits CHIME's WoE (CHIME's bad rate ~ population bad rate), 0.0
  when CHIME itself is absent from the fit data.
- Rows with a missing target are ignored during fit. The legacy WoE was fit
  on the known group only; dropping NaN-target rows reproduces that even
  when the encoder is fit on the mixed known/unknown train frame. When
  fitting on soft-label dual records (where synthetic rows carry y of 0/1),
  fit the encoder on the known rows before expansion instead.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# The mapping the legacy final model shipped with (fit on full-2025 known
# train). Kept beside the encoder because it is part of this preprocessing
# unit: the parity check at warehouse time diffs a freshly fit mapping
# against it, and from_legacy_mapping() replays it exactly.
LEGACY_MAPPING_PATH = Path(__file__).resolve().parent / "bankinstitutionwoe.json"


class BankInstitutionWOEEncoder(BaseEstimator, TransformerMixin):
    """WoE-encode a single categorical column with the legacy v3 semantics."""

    def __init__(
        self,
        min_obs: int = 30,
        smoothing: float = 0.5,
        fallback_category: str = "CHIME",
    ) -> None:
        self.min_obs = min_obs
        self.smoothing = smoothing
        self.fallback_category = fallback_category

    def fit(self, X, y):
        series = _as_series(X).reset_index(drop=True)
        target = pd.to_numeric(pd.Series(y).reset_index(drop=True), errors="coerce")

        # Known-only semantics: a missing target means an unknown-group row.
        labeled = target.notna()
        series = series[labeled]
        target = target[labeled]
        if len(target) == 0:
            raise ValueError(
                "BankInstitutionWOEEncoder.fit received no labeled rows; "
                "fit on rows where the target is present (known group)"
            )

        total_bad = max(float(target.sum()), 1.0)
        total_good = max(float(len(target) - target.sum()), 1.0)

        stats = target.groupby(series).agg(["sum", "count"])
        self.mapping_: dict[str, float] = {}
        self.iv_ = 0.0
        for value, row in stats[stats["count"] >= self.min_obs].iterrows():
            n_bad = float(row["sum"])
            n_good = float(row["count"] - row["sum"])
            dist_bad = (n_bad + self.smoothing) / total_bad
            dist_good = (n_good + self.smoothing) / total_good
            woe = math.log(dist_good / dist_bad)
            self.mapping_[str(value)] = woe
            self.iv_ += (dist_good - dist_bad) * woe
        self.other_woe_ = self.mapping_.get(self.fallback_category, 0.0)
        return self

    def transform(self, X):
        series = _as_series(X)
        encoded = series.map(
            lambda value: self.other_woe_
            if pd.isna(value)
            else self.mapping_.get(str(value), self.other_woe_)
        ).astype(float)
        return encoded.to_numpy().reshape(-1, 1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["bankinstitution_woe"], dtype=object)

    @classmethod
    def from_legacy_mapping(
        cls, path: str | Path = LEGACY_MAPPING_PATH
    ) -> "BankInstitutionWOEEncoder":
        """Build a fitted encoder from an exported WoE mapping, bypassing fit.

        Defaults to the legacy full-2025 artifact — use this to score with
        exactly the production v3 encoding, or to diff a refit against it.
        """
        mapping = {str(k): float(v) for k, v in json.loads(Path(path).read_text()).items()}
        encoder = cls()
        encoder.other_woe_ = mapping.pop("OTHER", mapping.get(encoder.fallback_category, 0.0))
        encoder.mapping_ = mapping
        return encoder


class PrefitBankInstitutionWOEEncoder(BankInstitutionWOEEncoder):
    """A WoE encoder whose mapping is decided before pipeline fit.

    The harness contract requires the project's WoE transformer to appear as
    a named entry inside the model's ColumnTransformer — but ColumnTransformer
    refits (and clones) its entries on whatever frame the model trains on.
    For soft-label dual records that frame carries synthetic 0/1 labels, and
    the legacy WoE must be fit on known rows only. This variant carries a
    mapping fit elsewhere (constructor params survive sklearn clone) and
    makes pipeline fit a no-op re-adoption of that mapping.
    """

    def __init__(
        self,
        mapping: dict[str, float] | None = None,
        other_woe: float = 0.0,
        min_obs: int = 30,
        smoothing: float = 0.5,
        fallback_category: str = "CHIME",
    ) -> None:
        super().__init__(
            min_obs=min_obs, smoothing=smoothing, fallback_category=fallback_category
        )
        self.mapping = mapping
        self.other_woe = other_woe

    @classmethod
    def from_fitted(
        cls, encoder: BankInstitutionWOEEncoder
    ) -> "PrefitBankInstitutionWOEEncoder":
        return cls(
            mapping=dict(encoder.mapping_),
            other_woe=float(encoder.other_woe_),
            min_obs=encoder.min_obs,
            smoothing=encoder.smoothing,
            fallback_category=encoder.fallback_category,
        )

    def fit(self, X, y=None):
        del X, y
        self.mapping_ = dict(self.mapping or {})
        self.other_woe_ = float(self.other_woe)
        return self


def _as_series(X) -> pd.Series:
    if isinstance(X, pd.Series):
        return X
    if isinstance(X, pd.DataFrame):
        if X.shape[1] != 1:
            raise ValueError("BankInstitutionWOEEncoder expects exactly one input column")
        return X.iloc[:, 0]
    array = np.asarray(X)
    if array.ndim == 2:
        if array.shape[1] != 1:
            raise ValueError("BankInstitutionWOEEncoder expects exactly one input column")
        array = array[:, 0]
    return pd.Series(array)


__all__ = ["BankInstitutionWOEEncoder", "PrefitBankInstitutionWOEEncoder"]
