"""Shared fixtures for the graph-store tests: a hand-checkable toy world."""

import pandas as pd
import pytest

_TS = pd.Timestamp


def _row(advance_id, user_id, ts, device=None, bank=None, persistent=None,
         phone=None, address=None, email=None, ip=None,
         loan_amount=50.0, is_fraud=0):
    return {
        "advance_id": advance_id, "user_id": user_id, "feature_as_of_ts": ts,
        "device_id": device, "bank_account_key": bank,
        "persistent_account_id": persistent, "phone_key": phone,
        "address_key": address, "email_key": email, "ip_address": ip,
        "loan_amount": loan_amount, "is_fraud": is_fraud,
        "label_gross_dpd45": is_fraud, "label_mature_d45": 1,
        "is_neobank_high_risk_institution": 0,
        # bad outcome becomes KNOWN 10 days after the advance (toy maturity)
        "expected_dpd45_date": ts + pd.Timedelta(days=10) if is_fraud else pd.NaT,
    }


@pytest.fixture()
def toy_df():
    rows = [
        _row("a01", "u1", _TS("2026-01-01 10:00"), device="d1", is_fraud=1),
        _row("a02", "u2", _TS("2026-01-02 10:00"), device="d1", loan_amount=150.0),
        _row("a02", "u2", _TS("2026-01-02 10:00"), device="d1", loan_amount=150.0),  # duplicate row
        _row("a03", "u2", _TS("2026-01-03 10:00"), bank="b1", loan_amount=150.0),
        _row("a04", "u3", _TS("2026-02-01 10:00"), bank="b1"),  # late edge (as-of tests)
        _row("a05", "u4", _TS("2026-01-04 10:00"), device="d4"),
        _row("a06", "u5", _TS("2026-01-05 10:00"), device="dH"),
        _row("a07", "u6", _TS("2026-01-06 10:00"), device="dH"),
        _row("a08", "u7", _TS("2026-01-07 10:00"), device="dH"),
        _row("a09", "u8", _TS("2026-01-08 10:00"), device="dH"),
        _row("a10", "u9", _TS("2026-01-09 10:00"), device="dH"),
        _row("a11", "u1", _TS("2026-01-06 11:00"), device="d1"),  # parallel edge u1-d1
        _row("a12", "u1", _TS("2026-01-07 11:00"), device="none"),  # sentinel -> screened
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def toy_store(toy_df, tmp_path):
    from projects.fraud_anomaly_detection.graph.build import build_store

    path = tmp_path / "toy_graph.duckdb"
    build_store(toy_df, path, source_label="toy")
    return path
