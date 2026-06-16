from __future__ import annotations

from typing import Mapping

import pytest

from projects.fraud_anomaly_detection.neo4j_codex.control.graph.methods import (
    Neo4jGraphDiscovery,
    default_neo4j_graph_methods,
)


class RecordingRunner:
    def __init__(self, rows_by_method: Mapping[str, list[dict]]) -> None:
        self.rows_by_method = rows_by_method
        self.calls: list[tuple[str, dict]] = []

    def run(self, query: str, params: Mapping[str, object]) -> list[dict]:
        call_params = dict(params)
        self.calls.append((query, call_params))
        return self.rows_by_method.get(str(call_params["method_name"]), [])


def test_default_neo4j_graph_methods_are_review_screens():
    methods = default_neo4j_graph_methods(["ring_account_reuse"])
    by_name = {method.name: method for method in methods}

    assert "graph:scenario_neighborhood:ring_account_reuse" in by_name
    assert "MATCHED_SCENARIO" in by_name[
        "graph:scenario_neighborhood:ring_account_reuse"
    ].cypher
    assert "gds.pageRank.stream" in by_name["graph:suspicion_queue_top200"].cypher
    assert {
        method.metadata.promotion_tier for method in methods
    } == {"review_queue"}
    assert not any(method.metadata.plug_eligible for method in methods)


def test_gds_projection_queries_do_not_redeclare_graph_name():
    methods = default_neo4j_graph_methods([])
    gds_queries = [
        method.cypher
        for method in methods
        if "gds.graph.project" in method.cypher
    ]

    assert gds_queries
    for query in gds_queries:
        for line in query.splitlines():
            if "YIELD graphName" in line:
                assert " AS " in line


def test_fraud_neighbour_query_uses_bounded_expansion_not_variable_path():
    by_name = {
        method.name: method
        for method in default_neo4j_graph_methods([])
    }

    query = by_name["graph:fraud_neighbours_hops2"].cypher

    assert "*1.." not in query
    assert "UNION" in query


def test_graph_queries_filter_users_by_asof_presence():
    methods = default_neo4j_graph_methods(["ring_account_reuse"])

    assert all("first_seen_ts <= localdatetime($as_of)" in method.cypher for method in methods)


def test_neo4j_graph_discovery_returns_contract_candidates():
    runner = RecordingRunner(
        {
            "graph:scenario_neighborhood:ring_account_reuse": [
                {"user_id": "u3", "score": 2.0, "shared_entities": 2}
            ],
            "graph:suspicion_queue_top200": [
                {"user_id": "u4", "score": 0.42}
            ],
        }
    )
    discovery = Neo4jGraphDiscovery(runner=runner)

    candidates = discovery.run(["ring_account_reuse"], as_of="2026-02-01T00:00:00")
    by_name = {candidate.name: candidate for candidate in candidates}

    assert by_name["graph:scenario_neighborhood:ring_account_reuse"].users == frozenset({"u3"})
    assert by_name["graph:suspicion_queue_top200"].users == frozenset({"u4"})
    assert all(candidate.metadata.method_type == "graph" for candidate in candidates)
    assert all(call[1]["as_of"] == "2026-02-01T00:00:00" for call in runner.calls)


def test_neo4j_graph_discovery_rejects_rows_without_user_id():
    runner = RecordingRunner({"graph:residual_ring_members": [{"score": 1.0}]})
    discovery = Neo4jGraphDiscovery(runner=runner)

    with pytest.raises(ValueError, match="user_id"):
        discovery.run([])
