"""Scenario validation: stats computation + write-back into register.yaml doc 2.

register.yaml is one consolidated file with two YAML documents: doc 1 is the
hand-written register (comments preserved — the script never rewrites it),
doc 2 is the machine-owned validation stats.
"""

import pandas as pd
import pytest
import yaml

from projects.fraud_anomaly_detection.scenarios.validation import (
    compute_stats,
    write_stats,
)
from projects.fraud_anomaly_detection.scenarios import SCENARIOS
from projects.fraud_anomaly_detection.tests.test_scenarios import make_frame

pytestmark = pytest.mark.unit


@pytest.fixture
def frame():
    """4 rows: 2 ring_account_reuse-matched (one never-paid, one repaid), 2 residual."""
    rows = [
        make_frame(advance_id="m0", heuristic_fraud_band="EXTREMELY_LIKELY",
                   label_gross_dpd45=1, label_repaid_current_snapshot=0, label_mature_d45=1),
        make_frame(advance_id="m1", heuristic_fraud_band="LOW",
                   label_gross_dpd45=1, label_repaid_current_snapshot=1, label_mature_d45=1),
        make_frame(advance_id="r0", loan_amount=50.0, heuristic_fraud_band="LOW",
                   label_gross_dpd45=0, label_repaid_current_snapshot=0, label_mature_d45=1),
        make_frame(advance_id="r1", prior_advances_on_bank_account_7d=0,
                   heuristic_fraud_band="POSSIBLE",
                   label_gross_dpd45=0, label_repaid_current_snapshot=0, label_mature_d45=0),
    ]
    return pd.concat(rows, ignore_index=True)


def test_compute_stats_per_scenario(frame):
    stats = compute_stats(frame, SCENARIOS)
    assert stats["n_rows"] == 4
    rar = stats["scenarios"]["ring_account_reuse"]
    assert rar["n"] == 2
    assert rar["share"] == pytest.approx(0.5)
    assert rar["n_mature"] == 2
    assert rar["n_never_paid"] == 1
    assert rar["n_resolved"] == 2  # both matched advances reached a verdict (repaid or DPD45)
    assert rar["never_paid_rate"] == pytest.approx(0.5)  # 1 never-paid / 2 resolved
    assert rar["n_dpd45"] == 2  # both matched mature rows hit gross DPD45
    assert rar["dpd45_rate"] == pytest.approx(1.0)
    assert rar["band_distribution"] == {"EXTREMELY_LIKELY": 1, "LOW": 1}


def test_compute_stats_baseline_section(frame):
    stats = compute_stats(frame, SCENARIOS)
    base = stats["baseline"]
    assert base["n_mature"] == 3  # r1 is immature
    # resolved denominator: m0 (never-paid) + m1 (repaid) = 2; r0/r1 unresolved
    assert base["never_paid_rate"] == pytest.approx(0.5)  # 1 never-paid (m0) / 2 resolved
    assert base["dpd45_rate"] == pytest.approx(2 / 3)  # m0 + m1 (matured denom, unchanged)
    assert base["bands"]["LOW"]["n"] == 2
    assert base["bands"]["LOW"]["never_paid_rate"] == pytest.approx(0.0)  # m1 repaid, r0 clean
    assert base["bands"]["EXTREMELY_LIKELY"]["never_paid_rate"] == pytest.approx(1.0)


def test_compute_stats_overall_union_overlap_residual(frame):
    stats = compute_stats(frame, SCENARIOS)
    overall = stats["overall"]
    assert overall["union"]["n"] == 2  # m0 + m1
    assert overall["union"]["never_paid_rate"] == pytest.approx(0.5)
    assert overall["overlap"]["n_multi_matched"] == 0  # single scenario: no overlap possible
    residual = overall["residual"]
    assert residual["n"] == 2  # r0 + r1
    # no resolved rows in residual (r0 is matured-but-current, r1 immature) -> undefined
    assert residual["never_paid_rate"] is None
    # band coverage: E_L fully captured, LOW half, POSSIBLE untouched
    assert residual["bands"]["EXTREMELY_LIKELY"] == {"n_left": 0, "coverage": 1.0}
    assert residual["bands"]["LOW"] == {"n_left": 1, "coverage": 0.5}
    assert residual["bands"]["POSSIBLE"] == {"n_left": 1, "coverage": 0.0}


def test_compute_stats_discovery_is_captured_low_band(frame):
    stats = compute_stats(frame, SCENARIOS)
    discovery = stats["overall"]["discovery"]
    # m1 is the only matched LOW-band row; it repaid -> not never-paid
    assert discovery["n"] == 1
    assert discovery["n_mature"] == 1
    assert discovery["never_paid_rate"] == pytest.approx(0.0)


def test_compute_stats_per_scenario_unique_capture_and_lift(frame):
    stats = compute_stats(frame, SCENARIOS)
    rar = stats["scenarios"]["ring_account_reuse"]
    # only one scenario registered: unique == gross
    assert rar["unique_n"] == 2
    assert rar["unique_never_paid_rate"] == pytest.approx(0.5)
    # 0.5 scenario never-paid rate over 0.5 resolved base rate
    assert rar["lift_vs_base"] == pytest.approx(1.0)


def test_compute_stats_empty_match_yields_none_rates(frame):
    residual_only = frame[frame["loan_amount"] < 100].reset_index(drop=True)
    stats = compute_stats(residual_only, SCENARIOS)
    rar = stats["scenarios"]["ring_account_reuse"]
    assert rar["n"] == 0
    assert rar["never_paid_rate"] is None
    assert rar["band_distribution"] == {}


REGISTER_DOC = (
    "# hand-written comment that must survive\n"
    'version: "test.1"\n'
    "scenarios: []\n"
)


def test_write_stats_appends_machine_doc_preserving_register(tmp_path):
    path = tmp_path / "scenarios.yaml"
    path.write_text(REGISTER_DOC)
    write_stats({"n_rows": 4, "scenarios": {}}, path=path)
    text = path.read_text()
    assert text.startswith(REGISTER_DOC)  # doc 1 untouched, comment intact
    docs = list(yaml.safe_load_all(text))
    assert len(docs) == 2
    assert docs[1]["n_rows"] == 4


def test_write_stats_replaces_existing_machine_doc(tmp_path):
    path = tmp_path / "scenarios.yaml"
    path.write_text(REGISTER_DOC)
    write_stats({"n_rows": 4, "scenarios": {}}, path=path)
    write_stats({"n_rows": 9, "scenarios": {}}, path=path)
    text = path.read_text()
    assert text.startswith(REGISTER_DOC)
    docs = list(yaml.safe_load_all(text))
    assert len(docs) == 2  # replaced, not appended again
    assert docs[1]["n_rows"] == 9
