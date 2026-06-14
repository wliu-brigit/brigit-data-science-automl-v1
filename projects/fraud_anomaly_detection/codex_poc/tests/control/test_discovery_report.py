from projects.fraud_anomaly_detection.codex_poc.control.contract import Finding, FindingSet
from projects.fraud_anomaly_detection.codex_poc.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.codex_poc.control.discovery_report import summarize_discovery


def _metadata(name: str, version: str, method_type: str) -> MethodMetadata:
    return MethodMetadata(
        name=name,
        version=version,
        method_type=method_type,  # type: ignore[arg-type]
        time_semantics="production_safe",
        promotion_tier="plug_candidate",
        enforcement_projection="entity_key",
    )


def test_summarize_discovery_reports_methods_and_deduped_union(tiny_store):
    scenario = FindingSet("scenario:test", "s1", [Finding("u1"), Finding("u2")])
    graph = FindingSet("graph:test", "g1", [Finding("u2"), Finding("u3")])

    report = summarize_discovery(
        tiny_store,
        [scenario, graph],
        method_metadata=[
            _metadata("scenario:test", "s1", "scenario"),
            _metadata("graph:test", "g1", "graph"),
        ],
    )

    assert report["methods"][0]["method"] == "scenario:test"
    assert report["methods"][0]["method_type"] == "scenario"
    assert report["methods"][1]["method_type"] == "graph"
    assert report["methods"][0]["n_users"] == 2
    assert report["methods"][0]["outcomes"]["n_dpd45_advances"] == 2
    assert report["union"]["n_users"] == 3
    assert report["union"]["outcomes"]["dpd45_advance_rate"] == 1.0
    assert report["attribution"] == {
        "by_method_type": {
            "graph": 2,
            "scenario": 2,
        },
        "multi_type_users": 1,
    }


def test_summarize_discovery_can_restrict_to_holdout_users(tiny_store):
    scenario = FindingSet("scenario:test", "s1", [Finding("u1"), Finding("u3")])
    graph = FindingSet("graph:test", "g1", [Finding("u4")])

    report = summarize_discovery(
        tiny_store,
        [scenario, graph],
        method_metadata=[
            _metadata("scenario:test", "s1", "scenario"),
            _metadata("graph:test", "g1", "graph"),
        ],
        eligible_users=["u3", "u4"],
    )

    assert report["union"]["n_users"] == 2
    assert report["attribution"] == {
        "by_method_type": {
            "graph": 1,
            "scenario": 1,
        },
        "multi_type_users": 0,
    }
