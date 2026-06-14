from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery.selection import (
    DiscoveryCandidate,
    SelectionRule,
    select_candidates,
)


def _outcome_factory(bad_users: set[str]):
    def outcome(users: set[str]) -> dict:
        user_ids = {str(user_id) for user_id in users}
        dpd45_users = user_ids & bad_users
        return {
            "users": len(user_ids),
            "dpd45_users": len(dpd45_users),
            "dpd45_user_rate": len(dpd45_users) / len(user_ids) if user_ids else 0.0,
        }

    return outcome


def test_select_candidates_uses_marginal_net_new_after_baseline_and_dedupe():
    baseline = {"u1", "u2"}
    outcome = _outcome_factory({"u3", "u4"})
    metadata = MethodMetadata(
        name="graph:good",
        version="v1",
        method_type="graph",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )
    duplicate = MethodMetadata(
        name="graph:duplicate",
        version="v1",
        method_type="graph",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    result = select_candidates(
        [
            DiscoveryCandidate("graph:good", {"u2", "u3", "u4"}, metadata),
            DiscoveryCandidate("graph:duplicate", {"u3", "u4", "u5"}, duplicate),
        ],
        baseline_users=baseline,
        outcome_fn=outcome,
        rule=SelectionRule(min_marginal_users=2, min_marginal_dpd45_user_rate=0.5),
    )

    assert [row.name for row in result.selected] == ["graph:good"]
    assert [row.name for row in result.excluded] == ["graph:duplicate"]
    assert result.selected[0].marginal_users == {"u3", "u4"}
    assert result.excluded[0].marginal_users == {"u5"}
    assert result.final_users == {"u1", "u2", "u3", "u4"}


def test_select_candidates_excludes_non_promotable_tiers():
    outcome = _outcome_factory({"u10", "u11"})
    metadata = MethodMetadata(
        name="graph:review",
        version="v1",
        method_type="graph",
        time_semantics="snapshot_review",
        promotion_tier="review_queue",
        enforcement_projection="entity_key",
    )

    result = select_candidates(
        [DiscoveryCandidate("graph:review", {"u10", "u11"}, metadata)],
        baseline_users=set(),
        outcome_fn=outcome,
        rule=SelectionRule(min_marginal_users=1, min_marginal_dpd45_user_rate=0.5),
    )

    assert result.selected == []
    assert result.excluded[0].name == "graph:review"
    assert result.excluded[0].reason == "promotion_tier"
    assert result.final_users == set()
