import pandas as pd

from projects.fraud_anomaly_detection.codex_poc.control import plug
from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig


def test_candidate_stats_compute_precision_and_coverage(tiny_store):
    discovered = pd.Series(["u1", "u2", "u3"])

    stats = plug.candidate_stats(tiny_store, discovered)

    acct_a = stats[(stats.entity_type == "bank") & (stats.entity_value == "acctA")].iloc[0]
    assert acct_a.support == 3
    assert acct_a.dpd45_precision == 1.0
    assert acct_a.coverage == 3
    assert acct_a.innocents == 0


def test_qualify_filters_by_config_over_stats(tiny_store):
    discovered = pd.Series(["u1", "u2", "u3"])
    stats = plug.candidate_stats(tiny_store, discovered)

    keys = plug.qualify(
        stats, ControlConfig(min_support=3, min_coverage=2, block_tier_precision=0.8)
    )

    assert ("bank", "acctA") in set(zip(keys.entity_type, keys.entity_value))
    none = plug.qualify(stats, ControlConfig(block_tier_precision=1.01))
    assert len(none) == 0


def test_candidate_stats_can_be_restricted_to_state_a_users(tiny_store):
    discovered = pd.Series(["u1", "u2"])

    stats = plug.candidate_stats(tiny_store, discovered, eligible_users=["u1", "u2", "u5"])

    acct_a = stats[(stats.entity_type == "bank") & (stats.entity_value == "acctA")].iloc[0]
    assert acct_a.support == 2
    assert acct_a.coverage == 2


def test_candidate_stats_extracts_only_keys_touched_by_discovery(tiny_store):
    discovered = pd.Series(["u1", "u2"])

    stats = plug.candidate_stats(tiny_store, discovered, eligible_users=["u1", "u2", "u5"])

    assert ("bank", "acctA") in set(zip(stats.entity_type, stats.entity_value))
    assert ("device", "devX") not in set(zip(stats.entity_type, stats.entity_value))
