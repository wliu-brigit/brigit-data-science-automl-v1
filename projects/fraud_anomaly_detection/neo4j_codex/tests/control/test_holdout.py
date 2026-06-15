from projects.fraud_anomaly_detection.neo4j_codex.control import holdout
from projects.fraud_anomaly_detection.neo4j_codex.control.config import ControlConfig


def test_split_partitions_users_by_cutoff(tiny_store):
    split = holdout.two_state_split(tiny_store, ControlConfig(holdout_days=30))

    assert set(split.state_a_users) == {"u1", "u2", "u5"}
    assert set(split.holdout_users) == {"u3", "u4"}
    assert split.cutoff < split.newest
