"""Scenario register: draft ring_account_reuse trigger predicate, assign(), residual_mask().

These tests pin the predicate semantics on synthetic frames so register edits
can't silently change what a scenario matches. The empirical check (does the
predicate reproduce the register's validation numbers on the pinned snapshot)
lives outside the unit tier — see the scenario verification analysis.
"""

import pandas as pd
import pytest

from projects.fraud_anomaly_detection.scenarios import (
    SCENARIOS,
    SCENARIOS_VERSION,
    assign,
    residual_mask,
)

pytestmark = pytest.mark.unit


TS = pd.Timestamp("2026-01-15 12:00:00")


def make_frame(**overrides):
    """One-row frame matching the draft ring_account_reuse trigger by default.

    Carries every register trigger column (new scenarios add theirs here with
    a non-matching default, so each scenario's tests stay independent).
    """
    row = {
        "advance_id": "a0",
        "feature_as_of_ts": TS,
        "identity_created_time": TS - pd.Timedelta(hours=12),
        "loan_amount": 150.0,
        "prior_advances_on_bank_account_7d": 2,
        "users_on_bank_account_72h": 0,  # ring_identity_burst: non-matching default
    }
    row.update(overrides)
    return pd.DataFrame([row])


def ring_account_reuse_matches(df) -> bool:
    return bool(assign(df)["scenario_ring_account_reuse"].iloc[0])


def ring_identity_burst_matches(df) -> bool:
    return bool(assign(df)["scenario_ring_identity_burst"].iloc[0])


def test_register_scenarios_and_rubric_fields():
    assert SCENARIOS_VERSION  # non-empty version stamp
    assert [s.name for s in SCENARIOS] == ["ring_account_reuse", "ring_identity_burst"]
    for scenario in SCENARIOS:
        assert scenario.status == "draft"  # not signed off — nothing may read it as law
        assert scenario.tier == "block"
        assert scenario.typology  # anchored to a published typology
        assert scenario.theory  # the behavioral story travels with the code


def test_ring_identity_burst_matches_three_fresh_identities():
    assert ring_identity_burst_matches(make_frame(users_on_bank_account_72h=3))
    assert ring_identity_burst_matches(make_frame(users_on_bank_account_72h=7))


def test_ring_identity_burst_boundary_and_nulls():
    assert not ring_identity_burst_matches(make_frame(users_on_bank_account_72h=2))
    assert not ring_identity_burst_matches(make_frame(users_on_bank_account_72h=None))


def test_ring_account_reuse_matches_fresh_identity_on_account_with_history():
    assert ring_account_reuse_matches(make_frame())


def test_ring_account_reuse_boundary_is_24_hours_inclusive():
    assert ring_account_reuse_matches(make_frame(identity_created_time=TS - pd.Timedelta(hours=24)))
    assert not ring_account_reuse_matches(make_frame(identity_created_time=TS - pd.Timedelta(hours=24, minutes=1)))


def test_ring_account_reuse_requires_each_conjunct():
    # aged identity -> the "day-old user" story collapses
    assert not ring_account_reuse_matches(make_frame(identity_created_time=TS - pd.Timedelta(days=400)))
    # amount at or under $100 -> below the confirmation threshold
    assert not ring_account_reuse_matches(make_frame(loan_amount=100.0))
    # no prior advances on the account -> that's S2's story, not a ring reuse
    assert not ring_account_reuse_matches(make_frame(prior_advances_on_bank_account_7d=0))


def test_ring_account_reuse_nulls_never_match():
    assert not ring_account_reuse_matches(make_frame(identity_created_time=pd.NaT))
    assert not ring_account_reuse_matches(make_frame(loan_amount=None))
    assert not ring_account_reuse_matches(make_frame(prior_advances_on_bank_account_7d=None))


def test_ring_account_reuse_accepts_string_timestamps():
    # Metadata columns may arrive as strings depending on the load path.
    assert ring_account_reuse_matches(
        make_frame(
            feature_as_of_ts=str(TS),
            identity_created_time=str(TS - pd.Timedelta(hours=12)),
        )
    )


def test_assign_adds_flag_columns_without_mutating_input():
    df = pd.concat(
        [make_frame(), make_frame(prior_advances_on_bank_account_7d=0)],
        ignore_index=True,
    )
    before = df.columns.tolist()
    flags = assign(df)
    assert df.columns.tolist() == before  # input untouched
    assert flags["scenario_ring_account_reuse"].tolist() == [True, False]
    assert flags["scenario_any"].tolist() == [True, False]
    assert flags.index.equals(df.index)


def test_residual_mask_is_complement_of_scenario_any():
    df = pd.concat(
        [make_frame(), make_frame(loan_amount=50.0)],
        ignore_index=True,
    )
    mask = residual_mask(df)
    assert mask.tolist() == [False, True]
