"""Pin the legacy fit_woe/apply_woe semantics of BankInstitutionWOEEncoder."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from projects.neobank_ncm.model.preprocessing import BankInstitutionWOEEncoder


def _fixture_frame() -> tuple[pd.Series, pd.Series]:
    """Hand-countable population.

    CHIME: 40 rows / 4 bad   (>= min_obs, ~population bad rate)
    VARO:  35 rows / 7 bad   (>= min_obs, riskier)
    STEP:  10 rows / 1 bad   (sparse: < min_obs -> no own entry)
    NaN:    5 rows / 1 bad   (missing bank -> no entry, still in totals)

    Labeled totals: n=90, bad=13, good=77.
    """
    banks = ["CHIME"] * 40 + ["VARO"] * 35 + ["STEP"] * 10 + [None] * 5
    target = (
        [1] * 4 + [0] * 36
        + [1] * 7 + [0] * 28
        + [1] * 1 + [0] * 9
        + [1] * 1 + [0] * 4
    )
    return pd.Series(banks), pd.Series(target)


def _woe(n_bad: float, n_good: float, total_bad: float, total_good: float) -> float:
    return math.log(((n_good + 0.5) / total_good) / ((n_bad + 0.5) / total_bad))


def test_fit_matches_legacy_formula():
    banks, target = _fixture_frame()
    encoder = BankInstitutionWOEEncoder().fit(banks, target)

    assert encoder.mapping_["CHIME"] == pytest.approx(_woe(4, 36, 13, 77))
    assert encoder.mapping_["VARO"] == pytest.approx(_woe(7, 28, 13, 77))
    # riskier bank -> lower WoE
    assert encoder.mapping_["VARO"] < encoder.mapping_["CHIME"]


def test_sparse_bank_gets_no_entry_but_counts_in_totals():
    banks, target = _fixture_frame()
    encoder = BankInstitutionWOEEncoder().fit(banks, target)

    assert "STEP" not in encoder.mapping_
    # totals (13 bad / 77 good) include STEP and NaN rows — pinned by the
    # CHIME value above; this asserts the sparse cut is on entries only.
    assert set(encoder.mapping_) == {"CHIME", "VARO"}


def test_other_inherits_chime_and_handles_unseen_and_nan():
    banks, target = _fixture_frame()
    encoder = BankInstitutionWOEEncoder().fit(banks, target)

    assert encoder.other_woe_ == encoder.mapping_["CHIME"]
    # the final notebook's own sanity cell: unseen STEP / direct CHIME / NaN
    out = encoder.transform(pd.Series(["STEP", "CHIME", None]))
    assert out.shape == (3, 1)
    assert out[0, 0] == encoder.other_woe_   # sparse -> OTHER
    assert out[1, 0] == encoder.mapping_["CHIME"]
    assert out[2, 0] == encoder.other_woe_   # NaN -> OTHER


def test_other_defaults_to_zero_without_chime():
    banks = pd.Series(["VARO"] * 40 + ["CURRENT"] * 40)
    target = pd.Series([1] * 8 + [0] * 32 + [1] * 2 + [0] * 38)
    encoder = BankInstitutionWOEEncoder().fit(banks, target)

    assert "CHIME" not in encoder.mapping_
    assert encoder.other_woe_ == 0.0


def test_nan_targets_are_ignored_in_fit():
    banks, target = _fixture_frame()
    # append unknown-group rows: bank present, target missing
    banks_mixed = pd.concat([banks, pd.Series(["VARO"] * 50)], ignore_index=True)
    target_mixed = pd.concat([target.astype(float), pd.Series([np.nan] * 50)], ignore_index=True)

    known_only = BankInstitutionWOEEncoder().fit(banks, target)
    mixed = BankInstitutionWOEEncoder().fit(banks_mixed, target_mixed)

    assert mixed.mapping_ == known_only.mapping_
    assert mixed.other_woe_ == known_only.other_woe_


def test_fit_with_no_labeled_rows_raises():
    with pytest.raises(ValueError, match="no labeled rows"):
        BankInstitutionWOEEncoder().fit(
            pd.Series(["CHIME", "VARO"]), pd.Series([np.nan, np.nan])
        )


def test_sklearn_clone_compatible():
    encoder = BankInstitutionWOEEncoder(min_obs=10, smoothing=1.0, fallback_category="VARO")
    cloned = clone(encoder)

    assert cloned.min_obs == 10
    assert cloned.smoothing == 1.0
    assert cloned.fallback_category == "VARO"


def test_from_legacy_mapping_replays_production_encoding():
    encoder = BankInstitutionWOEEncoder.from_legacy_mapping()

    # the legacy export carries OTHER = CHIME's WoE inside the mapping
    assert encoder.other_woe_ == encoder.mapping_["CHIME"]
    assert "OTHER" not in encoder.mapping_
    assert len(encoder.mapping_) == 27

    out = encoder.transform(pd.Series(["SOFI", "NEVER_SEEN_BANK", None]))
    assert out[0, 0] == pytest.approx(0.39840152345376395)
    assert out[1, 0] == encoder.other_woe_
    assert out[2, 0] == encoder.other_woe_


def test_accepts_single_column_frame_and_2d_array():
    banks, target = _fixture_frame()
    encoder = BankInstitutionWOEEncoder().fit(banks.to_frame("bankinstitution"), target)
    out = encoder.transform(np.array([["CHIME"], ["VARO"]], dtype=object))
    assert out.shape == (2, 1)
    assert out[0, 0] == encoder.mapping_["CHIME"]
