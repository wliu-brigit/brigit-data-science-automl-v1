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

    assert result.selected == ()
    assert result.excluded[0].name == "graph:review"
    assert result.excluded[0].reason == "promotion_tier"
    assert result.final_users == set()


def test_discovery_candidate_requires_canonical_metadata_name():
    metadata = MethodMetadata(
        name="graph:canonical",
        version="v1",
        method_type="graph",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    try:
        DiscoveryCandidate("display-only", {"u1"}, metadata)
    except ValueError as exc:
        assert "metadata.name" in str(exc)
    else:
        raise AssertionError("expected mismatched candidate name to fail")


def test_select_candidates_preserves_input_order_for_metric_ties():
    outcome = _outcome_factory({"u1", "u2", "u3", "u4"})
    first = MethodMetadata(
        name="graph:first",
        version="v1",
        method_type="graph",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )
    second = MethodMetadata(
        name="graph:second",
        version="v1",
        method_type="graph",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    result = select_candidates(
        [
            DiscoveryCandidate("graph:first", {"u1", "u2"}, first),
            DiscoveryCandidate("graph:second", {"u3", "u4"}, second),
        ],
        baseline_users=set(),
        outcome_fn=outcome,
        rule=SelectionRule(min_marginal_users=1, min_marginal_dpd45_user_rate=0.5),
    )

    assert [row.name for row in result.selected] == ["graph:first", "graph:second"]


def test_select_candidates_rejects_duplicate_candidate_names():
    outcome = _outcome_factory({"u1"})
    metadata = MethodMetadata(
        name="graph:duplicate",
        version="v1",
        method_type="graph",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    try:
        select_candidates(
            [
                DiscoveryCandidate("graph:duplicate", {"u1"}, metadata),
                DiscoveryCandidate("graph:duplicate", {"u2"}, metadata),
            ],
            baseline_users=set(),
            outcome_fn=outcome,
            rule=SelectionRule(min_marginal_users=1, min_marginal_dpd45_user_rate=0.5),
        )
    except ValueError as exc:
        assert "Duplicate discovery candidate name" in str(exc)
    else:
        raise AssertionError("expected duplicate candidate names to fail")


def test_select_candidates_fails_when_outcome_contract_is_missing_required_keys():
    metadata = MethodMetadata(
        name="graph:bad_outcome",
        version="v1",
        method_type="graph",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    try:
        select_candidates(
            [DiscoveryCandidate("graph:bad_outcome", {"u1"}, metadata)],
            baseline_users=set(),
            outcome_fn=lambda users: {"users": len(users)},
            rule=SelectionRule(min_marginal_users=1, min_marginal_dpd45_user_rate=0.5),
        )
    except KeyError as exc:
        assert "dpd45_user_rate" in str(exc)
    else:
        raise AssertionError("expected malformed outcome data to fail")


def test_selection_results_do_not_expose_mutable_user_sets():
    outcome = _outcome_factory({"u1"})
    metadata = MethodMetadata(
        name="graph:immutable",
        version="v1",
        method_type="graph",
        time_semantics="leakfree_asof",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )

    result = select_candidates(
        [DiscoveryCandidate("graph:immutable", {"u1"}, metadata)],
        baseline_users=set(),
        outcome_fn=outcome,
        rule=SelectionRule(min_marginal_users=1, min_marginal_dpd45_user_rate=0.5),
    )

    assert isinstance(result.final_users, frozenset)
    assert isinstance(result.selected, tuple)
    assert isinstance(result.excluded, tuple)
    assert isinstance(result.selected[0].users, frozenset)
    assert isinstance(result.selected[0].marginal_users, frozenset)
