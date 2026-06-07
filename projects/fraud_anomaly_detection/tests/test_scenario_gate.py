"""Fit gate: scenario-matched rows never reach model training."""

import pandas as pd
import pytest

from projects.fraud_anomaly_detection.scenarios.gate import gate_fit
from projects.fraud_anomaly_detection.tests.test_scenarios import make_frame

pytestmark = pytest.mark.unit


def test_gate_fit_drops_matched_rows_and_keeps_residual():
    df = pd.concat(
        [
            make_frame(advance_id="matched"),
            make_frame(advance_id="residual_amount", loan_amount=50.0),
            make_frame(advance_id="residual_prior", prior_advances_on_bank_account_7d=0),
        ],
        ignore_index=True,
    )
    gated = gate_fit(df)
    assert gated["advance_id"].tolist() == ["residual_amount", "residual_prior"]
    assert df.shape[0] == 3  # input not mutated


def test_gate_fit_noop_when_nothing_matches():
    df = make_frame(loan_amount=50.0)
    gated = gate_fit(df)
    assert gated.shape == df.shape
