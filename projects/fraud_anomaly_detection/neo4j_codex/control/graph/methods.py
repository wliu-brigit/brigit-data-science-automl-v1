"""Neo4j/Cypher/GDS graph discovery method registry."""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd

from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.graph_screen_catalog import (
    GraphScreenSpec,
    default_graph_screen_specs,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.metadata import (
    MethodMetadata,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.discovery.selection import (
    DiscoveryCandidate,
)
from projects.fraud_anomaly_detection.neo4j_codex.control.graph.client import (
    GraphQueryRunner,
)

REL_PATTERN = (
    "USED_DEVICE|USED_BANK_ACCOUNT|USED_PERSISTENT_ACCOUNT|USED_PHONE|USED_ADDRESS"
)
GDS_REL_PROJECTION = """
{
  USED_DEVICE: {orientation: 'UNDIRECTED'},
  USED_BANK_ACCOUNT: {orientation: 'UNDIRECTED'},
  USED_PERSISTENT_ACCOUNT: {orientation: 'UNDIRECTED'},
  USED_PHONE: {orientation: 'UNDIRECTED'},
  USED_ADDRESS: {orientation: 'UNDIRECTED'}
}
"""


@dataclass(frozen=True)
class Neo4jGraphMethod:
    """One Neo4j-backed graph discovery screen."""

    spec: GraphScreenSpec
    cypher: str
    params: Mapping[str, object] = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.spec.metadata.name

    @property
    def metadata(self) -> MethodMetadata:
        return self.spec.metadata

    def run(
        self,
        runner: GraphQueryRunner,
        *,
        as_of: str | None,
    ) -> DiscoveryCandidate:
        params = {
            "method_name": self.name,
            "as_of": as_of,
            **self.params,
        }
        rows = runner.run(self.cypher, params)
        users: set[str] = set()
        for row in rows:
            if "user_id" not in row:
                raise ValueError(f"Neo4j graph method {self.name!r} returned a row without user_id")
            users.add(str(row["user_id"]))
        return DiscoveryCandidate(name=self.name, users=users, metadata=self.metadata)


class Neo4jGraphDiscovery:
    """Runs the active Neo4j graph discovery registry."""

    def __init__(
        self,
        runner: GraphQueryRunner,
        methods: Iterable[Neo4jGraphMethod] | None = None,
    ) -> None:
        self.runner = runner
        self._methods = tuple(methods) if methods is not None else None

    def run(
        self,
        scenario_names: Iterable[str],
        *,
        as_of: str | pd.Timestamp | None = None,
    ) -> list[DiscoveryCandidate]:
        methods = self._methods or default_neo4j_graph_methods(scenario_names)
        as_of_value = _as_neo4j_localdatetime(as_of)
        return [method.run(self.runner, as_of=as_of_value) for method in methods]


def default_neo4j_graph_methods(
    scenario_names: Iterable[str],
) -> list[Neo4jGraphMethod]:
    """Return Neo4j-backed graph discovery screens for the report."""
    specs = {spec.name: spec for spec in default_graph_screen_specs(scenario_names)}
    methods = [
        Neo4jGraphMethod(
            specs["residual_ring_members"],
            _residual_ring_members_cypher(),
            {"graph_name": "fraud_residual_ring_members", "min_users": 3, "min_types": 2},
        ),
        Neo4jGraphMethod(
            specs["suspicion_queue_top200"],
            _suspicion_queue_cypher(),
            {"graph_name": "fraud_suspicion_queue", "top_n": 200},
        ),
        Neo4jGraphMethod(
            specs["fraud_neighbours_hops2"],
            _fraud_neighbours_cypher(),
            {"max_hops": 2},
        ),
        Neo4jGraphMethod(
            specs["high_risk_entity_members_scenario_fraud_seed"],
            _high_risk_entity_members_cypher(),
            {"min_entity_users": 3, "max_entity_users": 50},
        ),
        Neo4jGraphMethod(
            specs["multi_witness_neighbors_scenario_fraud_seed"],
            _multi_witness_neighbors_cypher(),
            {"min_shared_types": 2},
        ),
    ]
    methods.extend(
        Neo4jGraphMethod(
            specs[f"scenario_neighborhood:{scenario_name}"],
            _scenario_neighborhood_cypher(),
            {"scenario_name": scenario_name},
        )
        for scenario_name in scenario_names
    )
    return methods


def _as_neo4j_localdatetime(value: str | pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%S.%f")


def _relationship_asof(alias: str) -> str:
    return f"($as_of IS NULL OR {alias}.last_ts <= localdatetime($as_of))"


def _user_asof(alias: str) -> str:
    return f"($as_of IS NULL OR {alias}.first_seen_ts <= localdatetime($as_of))"


def _scenario_neighborhood_cypher() -> str:
    return f"""
    MATCH (:Scenario {{name: $scenario_name}})<-[:MATCHED_SCENARIO]-(seed:User)
          -[r:{REL_PATTERN}]->(entity:Entity)<-[r2:{REL_PATTERN}]-(candidate:User)
    WHERE candidate <> seed
      AND {_user_asof("seed")}
      AND {_user_asof("candidate")}
      AND coalesce(candidate.scenario_any, false) = false
      AND coalesce(candidate.is_fraud, false) = false
      AND {_relationship_asof("r")}
      AND {_relationship_asof("r2")}
    WITH candidate,
         count(DISTINCT seed) AS seed_users,
         count(DISTINCT entity) AS shared_entities,
         collect(DISTINCT entity.entity_type) AS shared_entity_types
    RETURN candidate.user_id AS user_id,
           seed_users + shared_entities AS score,
           seed_users AS seed_users,
           shared_entities AS shared_entities,
           shared_entity_types AS shared_entity_types
    ORDER BY score DESC, shared_entities DESC, user_id
    """


def _high_risk_entity_members_cypher() -> str:
    return f"""
    MATCH (member:User)-[r:{REL_PATTERN}]->(entity:Entity)
    WHERE {_user_asof("member")}
      AND {_relationship_asof("r")}
    WITH entity,
         count(DISTINCT member) AS entity_users,
         count(DISTINCT CASE WHEN coalesce(member.is_fraud, false) THEN member END) AS fraud_users,
         count(DISTINCT CASE WHEN coalesce(member.scenario_any, false) THEN member END) AS scenario_users
    WHERE entity_users >= $min_entity_users
      AND entity_users <= $max_entity_users
      AND (fraud_users >= 1 OR scenario_users >= 2)
    MATCH (candidate:User)-[r2:{REL_PATTERN}]->(entity)
    WHERE coalesce(candidate.scenario_any, false) = false
      AND coalesce(candidate.is_fraud, false) = false
      AND {_user_asof("candidate")}
      AND {_relationship_asof("r2")}
    RETURN DISTINCT candidate.user_id AS user_id,
           fraud_users + scenario_users AS score,
           entity.entity_type AS entity_type,
           entity.entity_value AS entity_value,
           entity_users AS entity_users,
           fraud_users AS fraud_users,
           scenario_users AS scenario_users
    ORDER BY score DESC, entity_users DESC, user_id
    """


def _multi_witness_neighbors_cypher() -> str:
    return f"""
    MATCH (candidate:User)-[r:{REL_PATTERN}]->(entity:Entity)<-[r2:{REL_PATTERN}]-(seed:User)
    WHERE candidate <> seed
      AND {_user_asof("candidate")}
      AND {_user_asof("seed")}
      AND coalesce(candidate.scenario_any, false) = false
      AND coalesce(candidate.is_fraud, false) = false
      AND (coalesce(seed.scenario_any, false) = true OR coalesce(seed.is_fraud, false) = true)
      AND {_relationship_asof("r")}
      AND {_relationship_asof("r2")}
    WITH candidate,
         collect(DISTINCT entity.entity_type) AS shared_entity_types,
         count(DISTINCT entity) AS shared_entities,
         count(DISTINCT seed) AS seed_users
    WHERE size(shared_entity_types) >= $min_shared_types
    RETURN candidate.user_id AS user_id,
           size(shared_entity_types) + shared_entities + seed_users AS score,
           shared_entity_types AS shared_entity_types,
           shared_entities AS shared_entities,
           seed_users AS seed_users
    ORDER BY size(shared_entity_types) DESC, shared_entities DESC, seed_users DESC, user_id
    """


def _fraud_neighbours_cypher() -> str:
    return f"""
    CALL () {{
      MATCH (seed:User)-[r1:{REL_PATTERN}]->(entity:Entity)<-[r2:{REL_PATTERN}]-(candidate:User)
      WHERE candidate <> seed
        AND {_user_asof("seed")}
        AND {_user_asof("candidate")}
        AND coalesce(seed.is_fraud, false) = true
        AND coalesce(candidate.scenario_any, false) = false
        AND coalesce(candidate.is_fraud, false) = false
        AND {_relationship_asof("r1")}
        AND {_relationship_asof("r2")}
      RETURN candidate.user_id AS user_id, 1 AS user_hops

      UNION

      MATCH (seed:User)-[r1:{REL_PATTERN}]->(entity:Entity)<-[r2:{REL_PATTERN}]-(mid:User)
      WHERE mid <> seed
        AND {_user_asof("seed")}
        AND {_user_asof("mid")}
        AND coalesce(seed.is_fraud, false) = true
        AND {_relationship_asof("r1")}
        AND {_relationship_asof("r2")}
      WITH DISTINCT mid
      MATCH (mid)-[r3:{REL_PATTERN}]->(entity2:Entity)<-[r4:{REL_PATTERN}]-(candidate:User)
      WHERE candidate <> mid
        AND {_user_asof("candidate")}
        AND coalesce(candidate.scenario_any, false) = false
        AND coalesce(candidate.is_fraud, false) = false
        AND {_relationship_asof("r3")}
        AND {_relationship_asof("r4")}
      RETURN candidate.user_id AS user_id, 2 AS user_hops
    }}
    WITH user_id, min(user_hops) AS user_hops
    WHERE user_hops <= $max_hops
    RETURN user_id AS user_id,
           1.0 / user_hops AS score,
           user_hops AS user_hops
    ORDER BY user_hops ASC, user_id
    """


def _suspicion_queue_cypher() -> str:
    return f"""
    WITH $graph_name AS graphName
    CALL (graphName) {{
      CALL gds.graph.drop(graphName, false) YIELD graphName AS droppedGraphName
      RETURN count(droppedGraphName) AS droppedCount
    }}
    WITH graphName
    CALL gds.graph.project(graphName, ['User', 'Entity'], {GDS_REL_PROJECTION})
    YIELD graphName AS projectedGraphName
    WITH projectedGraphName AS graphName
    CALL gds.pageRank.stream(graphName, {{maxIterations: 20, dampingFactor: 0.85}})
    YIELD nodeId, score
    WITH graphName, gds.util.asNode(nodeId) AS candidate, score
    WHERE candidate:User
      AND {_user_asof("candidate")}
      AND coalesce(candidate.scenario_any, false) = false
      AND coalesce(candidate.is_fraud, false) = false
      AND score > 0
    WITH graphName,
         collect({{user_id: candidate.user_id, score: score}})[0..$top_n] AS rows
    CALL (graphName) {{
      CALL gds.graph.drop(graphName, false) YIELD graphName AS droppedGraphName
      RETURN count(droppedGraphName) AS droppedCount
    }}
    WITH rows
    UNWIND rows AS row
    RETURN row.user_id AS user_id, row.score AS score
    ORDER BY score DESC, user_id
    """


def _residual_ring_members_cypher() -> str:
    return f"""
    WITH $graph_name AS graphName
    CALL (graphName) {{
      CALL gds.graph.drop(graphName, false) YIELD graphName AS droppedGraphName
      RETURN count(droppedGraphName) AS droppedCount
    }}
    WITH graphName
    CALL gds.graph.project(graphName, ['User', 'Entity'], {GDS_REL_PROJECTION})
    YIELD graphName AS projectedGraphName
    WITH projectedGraphName AS graphName
    CALL gds.wcc.stream(graphName)
    YIELD nodeId, componentId
    WITH graphName, componentId, collect(gds.util.asNode(nodeId)) AS nodes
    WITH graphName,
         componentId,
         [n IN nodes WHERE n:User] AS users,
         [n IN nodes WHERE n:Entity] AS entities
    WITH graphName,
         componentId,
         users,
         entities,
         [u IN users WHERE coalesce(u.scenario_any, false) = true
                      AND {_user_asof("u")}] AS flagged_users,
         [u IN users WHERE coalesce(u.scenario_any, false) = false
                      AND {_user_asof("u")}] AS residual_users
    UNWIND entities AS entity
    WITH graphName,
         componentId,
         users,
         flagged_users,
         residual_users,
         collect(DISTINCT entity.entity_type) AS entity_types
    WHERE size(users) >= $min_users
      AND size(entity_types) >= $min_types
      AND size(flagged_users) > 0
    UNWIND residual_users AS candidate
    WITH graphName,
         collect({{
             user_id: candidate.user_id,
             score: size(flagged_users),
             comp_id: componentId,
             ring_users: size(users),
             ring_types: size(entity_types),
             ring_flagged: size(flagged_users),
             entity_types: entity_types
         }}) AS rows
    CALL (graphName) {{
      CALL gds.graph.drop(graphName, false) YIELD graphName AS droppedGraphName
      RETURN count(droppedGraphName) AS droppedCount
    }}
    WITH rows
    UNWIND rows AS row
    RETURN row.user_id AS user_id,
           row.score AS score,
           row.comp_id AS comp_id,
           row.ring_users AS ring_users,
           row.ring_types AS ring_types,
           row.ring_flagged AS ring_flagged,
           row.entity_types AS entity_types
    ORDER BY ring_flagged DESC, ring_types DESC, ring_users DESC, user_id
    """
