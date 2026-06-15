"""Dense-block mining: greedy peeling with log-degree weights (Fraudar-style)."""

import pandas as pd
import pytest

pytest.importorskip("duckdb")  # project deps: uv sync --group fraud

from projects.fraud_anomaly_detection.graph.build import build_store  # noqa: E402
from projects.fraud_anomaly_detection.graph.dense import dense_blocks  # noqa: E402

pytestmark = pytest.mark.unit


def _row(adv, user, device=None, bank=None, phone=None):
    return {
        "advance_id": adv, "user_id": user,
        "feature_as_of_ts": pd.Timestamp("2026-01-01 10:00"),
        "device_id": device, "bank_account_key": bank,
        "persistent_account_id": None, "phone_key": phone,
        "address_key": None, "email_key": None, "ip_address": None,
        "loan_amount": 50.0, "is_fraud": 0, "label_gross_dpd45": 0,
        "label_mature_d45": 1, "is_neobank_high_risk_institution": 0,
        "expected_dpd45_date": pd.NaT,
    }


@pytest.fixture()
def planted_store(tmp_path):
    """A 4-user x 3-entity dense block, a 12-user popular device (camouflage),
    and sparse background pairs."""
    rows = []
    # the planted ring: r1..r4 all share device dR, bank bR, phone pR
    for i in range(1, 5):
        rows.append(_row(f"r{i}", f"ring{i}", device="dR", bank="bR", phone="pR"))
    # a popular device (NAT-ish): 12 users share dPOP and nothing else
    for i in range(1, 13):
        rows.append(_row(f"p{i}", f"pop{i}", device="dPOP"))
    # sparse background: disjoint user pairs sharing one device each
    for i in range(1, 7):
        rows.append(_row(f"s{i}a", f"bg{i}a", device=f"dS{i}"))
        rows.append(_row(f"s{i}b", f"bg{i}b", device=f"dS{i}"))
    store = tmp_path / "planted.duckdb"
    build_store(pd.DataFrame(rows), store, source_label="planted")
    return store


def test_top_block_recovers_planted_ring(planted_store):
    out = dense_blocks(planted_store, top_k=1)
    assert len(out) == 1
    top = out.iloc[0]
    assert set(top.user_ids.split(",")) == {"ring1", "ring2", "ring3", "ring4"}
    assert top.n_types == 3  # device AND bank AND phone web the block


def test_popular_entity_does_not_dominate(planted_store):
    # log-degree weighting: the 12-user single-entity star must not outrank
    # the multi-entity ring (the Fraudar camouflage-resistance property)
    out = dense_blocks(planted_store, top_k=2, min_users=2)
    assert "ring1" in out.iloc[0].user_ids
    assert "pop1" not in out.iloc[0].user_ids


def test_peel_and_repeat_returns_disjoint_blocks(planted_store):
    out = dense_blocks(planted_store, top_k=3, min_users=2)
    seen: set[str] = set()
    for users in out.user_ids:
        members = set(users.split(","))
        assert not (members & seen)  # blocks never share users
        seen |= members


def test_min_users_floor(planted_store):
    out = dense_blocks(planted_store, top_k=5, min_users=5)
    for users in out.user_ids:
        assert len(users.split(",")) >= 5
