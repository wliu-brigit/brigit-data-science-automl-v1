import pytest

from projects.fraud_anomaly_detection.codex_poc.control.discovery.graph_screen_catalog import (
    default_graph_screen_specs,
)


def test_default_graph_screen_specs_are_snapshot_review_only():
    specs = default_graph_screen_specs(["ring_account_reuse"])
    by_name = {spec.name: spec for spec in specs}

    assert not by_name[
        "high_risk_entity_members_scenario_fraud_seed"
    ].metadata.plug_eligible
    assert not by_name["residual_ring_members"].metadata.plug_eligible
    assert {
        spec.metadata.promotion_tier for spec in specs
    } == {"review_queue"}
    assert (
        by_name["scenario_neighborhood:ring_account_reuse"].metadata.params["scenario_name"]
        == "ring_account_reuse"
    )


def test_graph_screen_spec_candidates_use_canonical_metadata_name():
    spec = default_graph_screen_specs([])[0]
    candidate = spec.candidate({"u1"})

    assert candidate.name == spec.metadata.name
    assert candidate.metadata.params["display_name"] == spec.name
    assert candidate.users == frozenset({"u1"})


def test_graph_screen_spec_params_are_immutable():
    spec = default_graph_screen_specs(["ring_account_reuse"])[-1]

    with pytest.raises(TypeError):
        spec.params["scenario_name"] = "changed"
