"""Run Neo4j-native fraud discovery experiments against the disposable mirror.

This is intentionally a POC-side analysis runner. It does not edit the core
DuckDB graph store or scenario register. It treats Neo4j as a rebuildable mirror
and asks: which graph-native queues produce useful residual review candidates?

Run from the repo root while the local Neo4j mirror is running:

    uv run --with neo4j --group fraud python \
      -m projects.fraud_anomaly_detection.codex_poc.neo4j_discovery_experiments

Slow optional signal:

    uv run --with neo4j --group fraud python \
      -m projects.fraud_anomaly_detection.codex_poc.neo4j_discovery_experiments \
      --include-slow
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd
from neo4j import GraphDatabase

from projects.fraud_anomaly_detection.scenarios import SCENARIOS, assign


DEFAULT_URI = "bolt://localhost:7687"
DEFAULT_USER = "neo4j"
DEFAULT_PASSWORD = "fraudpocpass"
DEFAULT_OUT = Path("projects/fraud_anomaly_detection/codex_poc/out/discovery")
DEFAULT_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")
GRAPH_NAME = "fraud_discovery"

REL_PATTERN = (
    "USED_DEVICE|USED_BANK_ACCOUNT|USED_PERSISTENT_ACCOUNT|"
    "USED_PHONE|USED_ADDRESS"
)


@dataclass(frozen=True)
class QueueSpec:
    name: str
    description: str
    query: str
    params: dict[str, object] | None = None
    requires_gds: bool = False
    slow: bool = False


def _records_to_df(records: Iterable[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame([dict(r) for r in records])


def run_df(session, query: str, params: dict[str, object] | None = None) -> pd.DataFrame:
    result = session.run(query, params or {})
    return _records_to_df(result.data())


def ensure_indexes(session) -> None:
    statements = [
        "CREATE INDEX user_id IF NOT EXISTS FOR (u:User) ON (u.user_id)",
        "CREATE INDEX entity_key IF NOT EXISTS FOR (e:Entity) ON (e.entity_type, e.entity_value)",
        "CREATE INDEX scenario_name IF NOT EXISTS FOR (s:Scenario) ON (s.name)",
        "CREATE INDEX cluster_id IF NOT EXISTS FOR (c:ReviewCluster) ON (c.cluster_id)",
    ]
    for stmt in statements:
        session.run(stmt).consume()


def ensure_gds_graph(session) -> pd.DataFrame:
    session.run("CALL gds.graph.drop($name, false) YIELD graphName RETURN graphName", name=GRAPH_NAME).consume()
    return run_df(
        session,
        f"""
        CALL gds.graph.project(
          $name,
          ['User', 'Entity'],
          {{
            USED_DEVICE: {{orientation: 'UNDIRECTED'}},
            USED_BANK_ACCOUNT: {{orientation: 'UNDIRECTED'}},
            USED_PERSISTENT_ACCOUNT: {{orientation: 'UNDIRECTED'}},
            USED_PHONE: {{orientation: 'UNDIRECTED'}},
            USED_ADDRESS: {{orientation: 'UNDIRECTED'}}
          }}
        )
        YIELD graphName, nodeCount, relationshipCount, projectMillis
        RETURN graphName, nodeCount, relationshipCount, projectMillis
        """,
        {"name": GRAPH_NAME},
    )


def graph_inventory(session) -> pd.DataFrame:
    return run_df(
        session,
        """
        MATCH (u:User)
        RETURN count(u) AS users,
               sum(CASE WHEN u.scenario_any THEN 1 ELSE 0 END) AS scenario_users,
               sum(CASE WHEN u.is_fraud THEN 1 ELSE 0 END) AS fraud_users,
               sum(CASE WHEN u.label_gross_dpd45 THEN 1 ELSE 0 END) AS dpd45_users,
               sum(CASE WHEN u.scenario_any = false THEN 1 ELSE 0 END) AS residual_users,
               sum(CASE WHEN u.scenario_any = false AND u.is_fraud = false THEN 1 ELSE 0 END) AS residual_unknown_users,
               sum(CASE WHEN u.scenario_any = false AND u.label_gross_dpd45 THEN 1 ELSE 0 END) AS residual_dpd45_users
        """,
    )


def scenario_inventory(session) -> pd.DataFrame:
    return run_df(
        session,
        """
        MATCH (s:Scenario)<-[:MATCHED_SCENARIO]-(u:User)
        RETURN s.name AS scenario,
               count(DISTINCT u) AS users,
               sum(CASE WHEN u.label_gross_dpd45 THEN 1 ELSE 0 END) AS dpd45_users,
               sum(CASE WHEN u.is_fraud THEN 1 ELSE 0 END) AS fraud_users
        ORDER BY users DESC, scenario
        """,
    )


def scenario_mirror_validation(session, store: Path) -> pd.DataFrame:
    """Validate Neo4j scenario flags against the canonical DuckDB assignment."""
    rows: list[dict[str, object]] = []
    if not store.exists():
        return pd.DataFrame(
            [
                {
                    "scenario": "__store_missing__",
                    "duckdb_users": None,
                    "neo4j_relationship_users": None,
                    "neo4j_property_users": None,
                    "matches_relationship": False,
                    "matches_property": False,
                    "note": f"store not found: {store}",
                }
            ]
        )

    with duckdb.connect(str(store), read_only=True) as con:
        base = con.execute("SELECT * FROM advances").df()
    assigned = assign(base)
    expected = pd.DataFrame({"user_id": base["user_id"].astype(str)})
    for scenario in SCENARIOS:
        expected[scenario.name] = assigned[f"scenario_{scenario.name}"].fillna(False).astype(bool)
    expected["scenario_any"] = assigned["scenario_any"].fillna(False).astype(bool)
    expected = expected.groupby("user_id").max()

    for scenario in SCENARIOS:
        prop = f"scenario_{scenario.name}"
        neo_rel = run_df(
            session,
            """
            MATCH (s:Scenario {name: $scenario})<-[:MATCHED_SCENARIO]-(u:User)
            RETURN count(DISTINCT u) AS users
            """,
            {"scenario": scenario.name},
        )
        neo_prop = run_df(
            session,
            """
            MATCH (u:User)
            WHERE u[$prop] = true
            RETURN count(DISTINCT u) AS users
            """,
            {"prop": prop},
        )
        duck_users = int(expected[scenario.name].sum())
        rel_users = int(neo_rel.iloc[0]["users"]) if len(neo_rel) else 0
        prop_users = int(neo_prop.iloc[0]["users"]) if len(neo_prop) else 0
        rows.append(
            {
                "scenario": scenario.name,
                "duckdb_users": duck_users,
                "neo4j_relationship_users": rel_users,
                "neo4j_property_users": prop_users,
                "matches_relationship": duck_users == rel_users,
                "matches_property": duck_users == prop_users,
                "note": "mirror validation only; 72h semantics come from scenario feature columns, not graph-window recomputation",
            }
        )

    neo_any = run_df(
        session,
        "MATCH (u:User) WHERE u.scenario_any = true RETURN count(DISTINCT u) AS users",
    )
    duck_any = int(expected["scenario_any"].sum())
    neo_any_users = int(neo_any.iloc[0]["users"]) if len(neo_any) else 0
    rows.append(
        {
            "scenario": "scenario_any",
            "duckdb_users": duck_any,
            "neo4j_relationship_users": None,
            "neo4j_property_users": neo_any_users,
            "matches_relationship": None,
            "matches_property": duck_any == neo_any_users,
            "note": "union of registered scenarios",
        }
    )
    return pd.DataFrame(rows)


def queue_specs(include_slow: bool) -> list[QueueSpec]:
    rel = REL_PATTERN
    specs = [
        QueueSpec(
            name="baseline_residual",
            description="All non-scenario users. This is the denominator every residual queue must beat.",
            query="""
            MATCH (candidate:User)
            WHERE candidate.scenario_any = false AND candidate.is_fraud = false
            RETURN candidate.user_id AS candidate_user,
                   0.0 AS score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any
            ORDER BY candidate.user_id
            """,
        ),
        QueueSpec(
            name="shared_risky_neighbors",
            description=(
                "Residual users sharing any entity with scenario/fraud/DPD45 users. "
                "This is the direct Cypher version of guilt-by-association."
            ),
            query=f"""
            MATCH (candidate:User)-[:{rel}]->(e:Entity)<-[:{rel}]-(seed:User)
            WHERE candidate <> seed
              AND candidate.scenario_any = false
              AND candidate.is_fraud = false
              AND (seed.scenario_any = true OR seed.is_fraud = true OR seed.label_gross_dpd45 = true)
            WITH candidate,
                 count(DISTINCT e) AS shared_entities,
                 count(DISTINCT e.entity_type) AS shared_type_count,
                 collect(DISTINCT e.entity_type)[0..10] AS shared_entity_types,
                 count(DISTINCT seed) AS risky_neighbors,
                 count(DISTINCT CASE WHEN seed.is_fraud THEN seed END) AS fraud_neighbors,
                 count(DISTINCT CASE WHEN seed.scenario_any THEN seed END) AS scenario_neighbors,
                 count(DISTINCT CASE WHEN seed.label_gross_dpd45 THEN seed END) AS dpd45_neighbors
            RETURN candidate.user_id AS candidate_user,
                   fraud_neighbors * 5 + scenario_neighbors * 2 + dpd45_neighbors + shared_type_count * 0.25 AS score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any,
                   shared_entities, shared_type_count, shared_entity_types,
                   risky_neighbors, fraud_neighbors, scenario_neighbors, dpd45_neighbors
            ORDER BY score DESC, fraud_neighbors DESC, scenario_neighbors DESC, dpd45_neighbors DESC, shared_entities DESC
            LIMIT 1000
            """,
        ),
        QueueSpec(
            name="multi_witness_neighbors",
            description=(
                "Residual users linked to risky users through at least two independent entity types. "
                "This is closer to a scenario-ready pattern than a single shared entity."
            ),
            query=f"""
            MATCH (candidate:User)-[:{rel}]->(e:Entity)<-[:{rel}]-(seed:User)
            WHERE candidate <> seed
              AND candidate.scenario_any = false
              AND candidate.is_fraud = false
              AND (seed.scenario_any = true OR seed.is_fraud = true OR seed.label_gross_dpd45 = true)
            WITH candidate,
                 count(DISTINCT e) AS shared_entities,
                 count(DISTINCT e.entity_type) AS shared_type_count,
                 collect(DISTINCT e.entity_type)[0..10] AS shared_entity_types,
                 count(DISTINCT seed) AS risky_neighbors,
                 count(DISTINCT CASE WHEN seed.is_fraud THEN seed END) AS fraud_neighbors,
                 count(DISTINCT CASE WHEN seed.scenario_any THEN seed END) AS scenario_neighbors,
                 count(DISTINCT CASE WHEN seed.label_gross_dpd45 THEN seed END) AS dpd45_neighbors
            WHERE shared_type_count >= 2
            RETURN candidate.user_id AS candidate_user,
                   fraud_neighbors * 5 + scenario_neighbors * 2 + dpd45_neighbors + shared_type_count AS score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any,
                   shared_entities, shared_type_count, shared_entity_types,
                   risky_neighbors, fraud_neighbors, scenario_neighbors, dpd45_neighbors
            ORDER BY score DESC, shared_type_count DESC, fraud_neighbors DESC, scenario_neighbors DESC, dpd45_neighbors DESC
            LIMIT 1000
            """,
        ),
        QueueSpec(
            name="rarity_weighted_neighbors",
            description=(
                "Residual users connected to risky users through scarce entities. "
                "Each entity is weighted by 1/log(degree+5) to discount shared infrastructure."
            ),
            query=f"""
            MATCH (candidate:User)-[:{rel}]->(e:Entity)<-[:{rel}]-(seed:User)
            WHERE candidate <> seed
              AND candidate.scenario_any = false
              AND candidate.is_fraud = false
              AND (seed.scenario_any = true OR seed.is_fraud = true OR seed.label_gross_dpd45 = true)
            WITH candidate, e, collect(DISTINCT seed) AS seeds
            MATCH (e)<-[:{rel}]-(all_user:User)
            WITH candidate, e, seeds, count(DISTINCT all_user) AS entity_degree
            WITH candidate,
                 count(DISTINCT e) AS shared_entities,
                 count(DISTINCT e.entity_type) AS shared_type_count,
                 collect(DISTINCT e.entity_type)[0..10] AS shared_entity_types,
                 sum(1.0 / log(entity_degree + 5.0)) AS rarity_score,
                 sum(CASE WHEN any(s IN seeds WHERE s.is_fraud) THEN 1.0 / log(entity_degree + 5.0) ELSE 0.0 END) AS fraud_weight,
                 sum(CASE WHEN any(s IN seeds WHERE s.scenario_any) THEN 1.0 / log(entity_degree + 5.0) ELSE 0.0 END) AS scenario_weight,
                 sum(CASE WHEN any(s IN seeds WHERE s.label_gross_dpd45) THEN 1.0 / log(entity_degree + 5.0) ELSE 0.0 END) AS dpd45_weight,
                 min(entity_degree) AS min_entity_degree,
                 max(entity_degree) AS max_entity_degree
            RETURN candidate.user_id AS candidate_user,
                   fraud_weight * 5 + scenario_weight * 2 + dpd45_weight + rarity_score * 0.1 AS score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any,
                   shared_entities, shared_type_count, shared_entity_types,
                   rarity_score, fraud_weight, scenario_weight, dpd45_weight,
                   min_entity_degree, max_entity_degree
            ORDER BY score DESC, fraud_weight DESC, scenario_weight DESC, dpd45_weight DESC, rarity_score DESC
            LIMIT 1000
            """,
        ),
        QueueSpec(
            name="high_risk_entity_members",
            description=(
                "Residual users touching entities whose attached user set is already DPD45/fraud/scenario-heavy. "
                "This is the entity-centric review queue."
            ),
            query=f"""
            MATCH (e:Entity)<-[:{rel}]-(u:User)
            WITH e,
                 count(DISTINCT u) AS entity_users,
                 sum(CASE WHEN u.label_gross_dpd45 THEN 1 ELSE 0 END) AS entity_dpd45_users,
                 sum(CASE WHEN u.is_fraud THEN 1 ELSE 0 END) AS entity_fraud_users,
                 sum(CASE WHEN u.scenario_any THEN 1 ELSE 0 END) AS entity_scenario_users
            WHERE entity_users >= $min_entity_users AND entity_users <= $max_entity_users
              AND (entity_dpd45_users >= $min_dpd45_users
                   OR entity_fraud_users >= $min_fraud_users
                   OR entity_scenario_users >= $min_scenario_users)
            WITH e, entity_users, entity_dpd45_users, entity_fraud_users, entity_scenario_users,
                 toFloat(entity_dpd45_users) / entity_users AS entity_dpd45_rate,
                 toFloat(entity_fraud_users) / entity_users AS entity_fraud_rate,
                 toFloat(entity_scenario_users) / entity_users AS entity_scenario_rate
            MATCH (e)<-[:{rel}]-(candidate:User)
            WHERE candidate.scenario_any = false AND candidate.is_fraud = false
            WITH candidate,
                 count(DISTINCT e) AS risky_entity_count,
                 collect(DISTINCT e.entity_type)[0..10] AS entity_types,
                 max(entity_dpd45_rate) AS max_entity_dpd45_rate,
                 max(entity_fraud_rate) AS max_entity_fraud_rate,
                 max(entity_scenario_rate) AS max_entity_scenario_rate,
                 sum(entity_dpd45_users) AS touched_entity_dpd45_users,
                 sum(entity_fraud_users) AS touched_entity_fraud_users,
                 sum(entity_scenario_users) AS touched_entity_scenario_users
            RETURN candidate.user_id AS candidate_user,
                   max_entity_fraud_rate * 10 + max_entity_dpd45_rate * 5 + risky_entity_count AS score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any,
                   risky_entity_count, entity_types,
                   max_entity_dpd45_rate, max_entity_fraud_rate, max_entity_scenario_rate,
                   touched_entity_dpd45_users, touched_entity_fraud_users, touched_entity_scenario_users
            ORDER BY score DESC, max_entity_fraud_rate DESC, max_entity_dpd45_rate DESC, risky_entity_count DESC
            LIMIT 1000
            """,
            params={
                "min_entity_users": 3,
                "max_entity_users": 50,
                "min_dpd45_users": 2,
                "min_fraud_users": 1,
                "min_scenario_users": 2,
            },
        ),
        QueueSpec(
            name="gds_wcc_high_risk_components",
            description=(
                "Residual users inside connected components with high DPD45/scenario/fraud concentration."
            ),
            query="""
            CALL gds.wcc.stream($graph_name)
            YIELD nodeId, componentId
            WITH gds.util.asNode(nodeId) AS n, componentId
            WHERE n:User
            WITH componentId,
                 collect(n) AS users,
                 count(n) AS component_users,
                 sum(CASE WHEN n.label_gross_dpd45 THEN 1 ELSE 0 END) AS component_dpd45_users,
                 sum(CASE WHEN n.is_fraud THEN 1 ELSE 0 END) AS component_fraud_users,
                 sum(CASE WHEN n.scenario_any THEN 1 ELSE 0 END) AS component_scenario_users
            WHERE component_users >= 3
              AND component_dpd45_users >= 2
              AND toFloat(component_dpd45_users) / component_users >= 0.5
            UNWIND [u IN users WHERE u.scenario_any = false AND u.is_fraud = false] AS candidate
            RETURN candidate.user_id AS candidate_user,
                   toFloat(component_dpd45_users) / component_users * 10
                     + component_fraud_users * 2 + component_scenario_users AS score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any,
                   componentId AS component_id,
                   component_users, component_dpd45_users, component_fraud_users, component_scenario_users,
                   toFloat(component_dpd45_users) / component_users AS component_dpd45_rate
            ORDER BY score DESC, component_dpd45_rate DESC, component_dpd45_users DESC
            LIMIT 1000
            """,
            params={"graph_name": GRAPH_NAME},
            requires_gds=True,
        ),
        QueueSpec(
            name="gds_labelprop_high_risk_communities",
            description=(
                "Residual users inside label-propagation communities with high DPD45/scenario/fraud concentration."
            ),
            query="""
            CALL gds.labelPropagation.stream($graph_name, {maxIterations: 20})
            YIELD nodeId, communityId
            WITH gds.util.asNode(nodeId) AS n, communityId
            WHERE n:User
            WITH communityId,
                 collect(n) AS users,
                 count(n) AS community_users,
                 sum(CASE WHEN n.label_gross_dpd45 THEN 1 ELSE 0 END) AS community_dpd45_users,
                 sum(CASE WHEN n.is_fraud THEN 1 ELSE 0 END) AS community_fraud_users,
                 sum(CASE WHEN n.scenario_any THEN 1 ELSE 0 END) AS community_scenario_users
            WHERE community_users >= 3
              AND community_dpd45_users >= 2
              AND toFloat(community_dpd45_users) / community_users >= 0.5
            UNWIND [u IN users WHERE u.scenario_any = false AND u.is_fraud = false] AS candidate
            RETURN candidate.user_id AS candidate_user,
                   toFloat(community_dpd45_users) / community_users * 10
                     + community_fraud_users * 2 + community_scenario_users AS score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any,
                   communityId AS community_id,
                   community_users, community_dpd45_users, community_fraud_users, community_scenario_users,
                   toFloat(community_dpd45_users) / community_users AS community_dpd45_rate
            ORDER BY score DESC, community_dpd45_rate DESC, community_dpd45_users DESC
            LIMIT 1000
            """,
            params={"graph_name": GRAPH_NAME},
            requires_gds=True,
        ),
        QueueSpec(
            name="gds_ppr_scenario_fraud_seed",
            description=(
                "Personalized PageRank seeded only from current scenarios and known fraud proxy. "
                "This is the least circular propagation queue."
            ),
            query="""
            MATCH (seed:User)
            WHERE seed.scenario_any = true OR seed.is_fraud = true
            WITH collect(seed) AS seeds
            CALL gds.pageRank.stream($graph_name, {
              sourceNodes: seeds,
              maxIterations: 20,
              dampingFactor: 0.85
            })
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS candidate, score
            WHERE candidate:User AND candidate.scenario_any = false AND candidate.is_fraud = false
            RETURN candidate.user_id AS candidate_user,
                   score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any
            ORDER BY score DESC
            LIMIT 1000
            """,
            params={"graph_name": GRAPH_NAME},
            requires_gds=True,
        ),
        QueueSpec(
            name="gds_ppr_dpd45_seed_upper_bound",
            description=(
                "Personalized PageRank seeded from scenarios, fraud proxy, and DPD45. "
                "This is useful as a case-review queue, but circular if DPD45 is the target."
            ),
            query="""
            MATCH (seed:User)
            WHERE seed.scenario_any = true OR seed.is_fraud = true OR seed.label_gross_dpd45 = true
            WITH collect(seed) AS seeds
            CALL gds.pageRank.stream($graph_name, {
              sourceNodes: seeds,
              maxIterations: 20,
              dampingFactor: 0.85
            })
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS candidate, score
            WHERE candidate:User AND candidate.scenario_any = false AND candidate.is_fraud = false
            RETURN candidate.user_id AS candidate_user,
                   score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any
            ORDER BY score DESC
            LIMIT 1000
            """,
            params={"graph_name": GRAPH_NAME},
            requires_gds=True,
        ),
        QueueSpec(
            name="gds_global_pagerank",
            description=(
                "Global PageRank on the bipartite graph. Included as a negative/control signal."
            ),
            query="""
            CALL gds.pageRank.stream($graph_name, {maxIterations: 20, dampingFactor: 0.85})
            YIELD nodeId, score
            WITH gds.util.asNode(nodeId) AS candidate, score
            WHERE candidate:User AND candidate.scenario_any = false AND candidate.is_fraud = false
            RETURN candidate.user_id AS candidate_user,
                   score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any
            ORDER BY score DESC
            LIMIT 1000
            """,
            params={"graph_name": GRAPH_NAME},
            requires_gds=True,
        ),
        QueueSpec(
            name="gds_node_similarity_to_risky",
            description=(
                "Users most similar to risky users by shared-neighbor Jaccard. "
                "Useful, but notably slower on the sample and risky for full data."
            ),
            query="""
            CALL gds.nodeSimilarity.stream($graph_name, {
              topK: 10,
              similarityCutoff: 0.05,
              degreeCutoff: 1
            })
            YIELD node1, node2, similarity
            WITH gds.util.asNode(node1) AS a, gds.util.asNode(node2) AS b, similarity
            WHERE a:User AND b:User
            WITH CASE WHEN a.scenario_any = false AND a.is_fraud = false THEN a ELSE b END AS candidate,
                 CASE WHEN a.scenario_any = false AND a.is_fraud = false THEN b ELSE a END AS seed,
                 similarity
            WHERE candidate:User
              AND candidate.scenario_any = false
              AND candidate.is_fraud = false
              AND (seed.scenario_any = true OR seed.is_fraud = true OR seed.label_gross_dpd45 = true)
            WITH candidate,
                 max(similarity) AS max_similarity,
                 avg(similarity) AS avg_similarity,
                 count(DISTINCT seed) AS similar_risky_users,
                 sum(CASE WHEN seed.label_gross_dpd45 THEN 1 ELSE 0 END) AS similar_dpd45_users,
                 sum(CASE WHEN seed.is_fraud THEN 1 ELSE 0 END) AS similar_fraud_users,
                 sum(CASE WHEN seed.scenario_any THEN 1 ELSE 0 END) AS similar_scenario_users
            RETURN candidate.user_id AS candidate_user,
                   similar_fraud_users * 5 + similar_scenario_users * 2
                     + similar_dpd45_users + max_similarity AS score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any,
                   max_similarity, avg_similarity, similar_risky_users,
                   similar_dpd45_users, similar_fraud_users, similar_scenario_users
            ORDER BY score DESC, similar_fraud_users DESC, similar_dpd45_users DESC, max_similarity DESC
            LIMIT 1000
            """,
            params={"graph_name": GRAPH_NAME},
            requires_gds=True,
            slow=True,
        ),
    ]
    return [s for s in specs if include_slow or not s.slow]


def evaluate_queue(df: pd.DataFrame, name: str) -> list[dict[str, object]]:
    if "candidate_user" not in df.columns:
        return []
    dedup = df.drop_duplicates("candidate_user", keep="first").reset_index(drop=True)
    if not {"dpd45", "is_fraud", "scenario_any"}.issubset(dedup.columns):
        return []
    rows: list[dict[str, object]] = []
    cuts = [10, 25, 50, 100, 250, 500, 1000, len(dedup)]
    seen_cuts: set[int] = set()
    for cut in cuts:
        cut = min(int(cut), len(dedup))
        if cut <= 0 or cut in seen_cuts:
            continue
        seen_cuts.add(cut)
        sub = dedup.head(cut)
        rows.append(
            {
                "queue": name,
                "top_n": cut if cut < len(dedup) else "all",
                "n_users": len(sub),
                "dpd45_users": int(sub["dpd45"].fillna(False).astype(bool).sum()),
                "dpd45_rate": float(sub["dpd45"].fillna(False).astype(bool).mean()),
                "fraud_users": int(sub["is_fraud"].fillna(False).astype(bool).sum()),
                "fraud_rate": float(sub["is_fraud"].fillna(False).astype(bool).mean()),
                "scenario_users": int(sub["scenario_any"].fillna(False).astype(bool).sum()),
                "scenario_rate": float(sub["scenario_any"].fillna(False).astype(bool).mean()),
            }
        )
    return rows


def truth_table(session) -> pd.DataFrame:
    return run_df(
        session,
        """
        MATCH (u:User)
        RETURN u.user_id AS user_id,
               u.label_gross_dpd45 AS dpd45,
               u.is_fraud AS is_fraud,
               u.scenario_any AS scenario_any
        """,
    )


def cumulative_coverage(
    queue_frames: dict[str, pd.DataFrame],
    truth: pd.DataFrame,
    *,
    top_n: int,
) -> pd.DataFrame:
    truth_by_user = truth.drop_duplicates("user_id").set_index("user_id")
    seen: set[str] = set()
    rows: list[dict[str, object]] = []
    for name, df in queue_frames.items():
        if name == "baseline_residual" or "candidate_user" not in df.columns:
            continue
        ordered_users = [str(u) for u in df["candidate_user"].dropna().drop_duplicates().head(top_n)]
        new_users = [u for u in ordered_users if u not in seen]
        seen.update(new_users)
        sub = truth_by_user.reindex(new_users).dropna(how="all")
        cumulative = truth_by_user.reindex(list(seen)).dropna(how="all")
        rows.append(
            {
                "queue": name,
                "top_n_per_queue": top_n,
                "new_users": len(sub),
                "new_dpd45_users": int(sub["dpd45"].fillna(False).astype(bool).sum()) if len(sub) else 0,
                "new_dpd45_rate": float(sub["dpd45"].fillna(False).astype(bool).mean()) if len(sub) else 0.0,
                "cumulative_users": len(cumulative),
                "cumulative_dpd45_users": int(cumulative["dpd45"].fillna(False).astype(bool).sum()) if len(cumulative) else 0,
                "cumulative_dpd45_rate": float(cumulative["dpd45"].fillna(False).astype(bool).mean()) if len(cumulative) else 0.0,
            }
        )
    return pd.DataFrame(rows)


def top_candidate_evidence(session, queue_frames: dict[str, pd.DataFrame], top_n: int = 100) -> pd.DataFrame:
    users: list[str] = []
    for name, df in queue_frames.items():
        if name == "baseline_residual" or "candidate_user" not in df.columns:
            continue
        users.extend(str(u) for u in df["candidate_user"].dropna().drop_duplicates().head(top_n))
    users = sorted(set(users))
    if not users:
        return pd.DataFrame()
    return run_df(
        session,
        f"""
        UNWIND $users AS user_id
        MATCH (candidate:User {{user_id: user_id}})-[:{REL_PATTERN}]->(e:Entity)<-[:{REL_PATTERN}]-(risk:User)
        WHERE risk <> candidate
          AND (risk.scenario_any = true OR risk.is_fraud = true OR risk.label_gross_dpd45 = true)
        WITH e,
             count(DISTINCT candidate) AS candidate_users,
             count(DISTINCT risk) AS risky_neighbors,
             count(DISTINCT CASE WHEN risk.scenario_any THEN risk END) AS scenario_neighbors,
             count(DISTINCT CASE WHEN risk.is_fraud THEN risk END) AS fraud_neighbors,
             count(DISTINCT CASE WHEN risk.label_gross_dpd45 THEN risk END) AS dpd45_neighbors
        MATCH (e)<-[:{REL_PATTERN}]-(all_user:User)
        WITH e, candidate_users, risky_neighbors, scenario_neighbors, fraud_neighbors, dpd45_neighbors,
             count(DISTINCT all_user) AS entity_degree
        RETURN e.entity_type AS entity_type,
               left(e.entity_value, 64) AS entity_value_sample,
               entity_degree,
               candidate_users,
               risky_neighbors,
               scenario_neighbors,
               fraud_neighbors,
               dpd45_neighbors
        ORDER BY candidate_users DESC, fraud_neighbors DESC, dpd45_neighbors DESC, entity_degree ASC
        LIMIT 200
        """,
        {"users": users},
    )


def scenario_packets(session, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write per-scenario residual queues and scenario evidence summaries."""
    candidate_summaries: list[dict[str, object]] = []
    entity_frames: list[pd.DataFrame] = []
    for scenario in SCENARIOS:
        candidates = run_df(
            session,
            f"""
            MATCH (s:Scenario {{name: $scenario}})<-[:MATCHED_SCENARIO]-(seed:User)-[:{REL_PATTERN}]->(e:Entity)
                  <-[:{REL_PATTERN}]-(candidate:User)
            WHERE candidate <> seed
              AND candidate.scenario_any = false
              AND candidate.is_fraud = false
            WITH candidate,
                 count(DISTINCT seed) AS scenario_neighbors,
                 count(DISTINCT e) AS shared_entities,
                 count(DISTINCT e.entity_type) AS shared_type_count,
                 collect(DISTINCT e.entity_type)[0..10] AS shared_entity_types,
                 count(DISTINCT CASE WHEN seed.label_gross_dpd45 THEN seed END) AS scenario_dpd45_neighbors,
                 count(DISTINCT CASE WHEN seed.is_fraud THEN seed END) AS scenario_fraud_neighbors
            RETURN candidate.user_id AS candidate_user,
                   scenario_fraud_neighbors * 5 + scenario_dpd45_neighbors * 2
                     + scenario_neighbors + shared_type_count AS score,
                   candidate.label_gross_dpd45 AS dpd45,
                   candidate.is_fraud AS is_fraud,
                   candidate.scenario_any AS scenario_any,
                   scenario_neighbors, scenario_dpd45_neighbors, scenario_fraud_neighbors,
                   shared_entities, shared_type_count, shared_entity_types
            ORDER BY score DESC, scenario_fraud_neighbors DESC, scenario_dpd45_neighbors DESC, scenario_neighbors DESC
            LIMIT 500
            """,
            {"scenario": scenario.name},
        )
        candidates.to_csv(out_dir / f"scenario_{scenario.name}_residual_candidates.csv", index=False)
        for row in evaluate_queue(candidates, scenario.name):
            candidate_summaries.append({"scenario": scenario.name, **row})

        entities = run_df(
            session,
            f"""
            MATCH (s:Scenario {{name: $scenario}})<-[:MATCHED_SCENARIO]-(u:User)-[:{REL_PATTERN}]->(e:Entity)
            WITH e,
                 count(DISTINCT u) AS matched_scenario_users,
                 count(DISTINCT CASE WHEN u.label_gross_dpd45 THEN u END) AS matched_dpd45_users,
                 count(DISTINCT CASE WHEN u.is_fraud THEN u END) AS matched_fraud_users
            MATCH (e)<-[:{REL_PATTERN}]-(all_user:User)
            WITH e, matched_scenario_users, matched_dpd45_users, matched_fraud_users,
                 count(DISTINCT all_user) AS entity_degree,
                 count(DISTINCT CASE WHEN all_user.scenario_any = false THEN all_user END) AS residual_users_on_entity,
                 count(DISTINCT CASE WHEN all_user.scenario_any = false AND all_user.label_gross_dpd45 THEN all_user END) AS residual_dpd45_users_on_entity
            RETURN $scenario AS scenario,
                   e.entity_type AS entity_type,
                   left(e.entity_value, 64) AS entity_value_sample,
                   entity_degree,
                   matched_scenario_users,
                   matched_dpd45_users,
                   matched_fraud_users,
                   residual_users_on_entity,
                   residual_dpd45_users_on_entity
            ORDER BY matched_fraud_users DESC, matched_dpd45_users DESC, matched_scenario_users DESC, entity_degree ASC
            LIMIT 50
            """,
            {"scenario": scenario.name},
        )
        entity_frames.append(entities)

    summary = pd.DataFrame(candidate_summaries)
    entity_evidence = pd.concat(entity_frames, ignore_index=True) if entity_frames else pd.DataFrame()
    summary.to_csv(out_dir / "scenario_residual_candidate_summary.csv", index=False)
    entity_evidence.to_csv(out_dir / "scenario_entity_evidence.csv", index=False)
    return summary, entity_evidence


def _fmt_rate(value: object) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return f"{float(value):.1%}"


def write_markdown(
    out_dir: Path,
    *,
    inventory: pd.DataFrame,
    scenarios: pd.DataFrame,
    scenario_validation: pd.DataFrame,
    gds_projection: pd.DataFrame,
    queue_meta: pd.DataFrame,
    summary: pd.DataFrame,
    coverage_100: pd.DataFrame,
    coverage_all: pd.DataFrame,
    evidence_entities: pd.DataFrame,
    scenario_candidate_summary: pd.DataFrame,
    scenario_entity_evidence: pd.DataFrame,
) -> Path:
    path = out_dir / "summary.md"
    inv = inventory.iloc[0].to_dict() if len(inventory) else {}
    lines: list[str] = [
        "# Neo4j Discovery Experiment Summary",
        "",
        "This is a sample-data workflow artifact. Treat rates as queue diagnostics, not production findings.",
        "",
        "## Inventory",
        "",
        f"- Users: {int(inv.get('users', 0)):,}",
        f"- Scenario users: {int(inv.get('scenario_users', 0)):,}",
        f"- Fraud-proxy users: {int(inv.get('fraud_users', 0)):,}",
        f"- DPD45 users: {int(inv.get('dpd45_users', 0)):,}",
        f"- Non-scenario users: {int(inv.get('residual_users', 0)):,}",
        f"- Non-scenario, non-fraud users: {int(inv.get('residual_unknown_users', 0)):,}",
        f"- Non-scenario DPD45 users: {int(inv.get('residual_dpd45_users', 0)):,}",
        "",
        "## Scenario Coverage",
        "",
    ]
    if len(scenarios):
        lines.extend(_markdown_table(scenarios))
    else:
        lines.append("_No scenarios found._")
    lines.extend(["", "## Scenario Mirror Validation", ""])
    if len(scenario_validation):
        validation_view = scenario_validation.copy()
        lines.extend(_markdown_table(validation_view))
    else:
        lines.append("_No validation rows._")
    lines.extend(["", "## GDS Projection", ""])
    if len(gds_projection):
        lines.extend(_markdown_table(gds_projection))
    else:
        lines.append("_GDS projection was not created._")
    lines.extend(["", "## Queue Results", ""])
    if len(summary):
        view = summary.copy()
        view["dpd45_rate"] = view["dpd45_rate"].map(_fmt_rate)
        view["fraud_rate"] = view["fraud_rate"].map(_fmt_rate)
        view["scenario_rate"] = view["scenario_rate"].map(_fmt_rate)
        lines.extend(_markdown_table(view))
    else:
        lines.append("_No queue summaries._")
    lines.extend(["", "## Net-New Coverage By Method", ""])
    lines.append("Cumulative queue coverage using top 100 rows per method:")
    lines.append("")
    if len(coverage_100):
        view = coverage_100.copy()
        view["new_dpd45_rate"] = view["new_dpd45_rate"].map(_fmt_rate)
        view["cumulative_dpd45_rate"] = view["cumulative_dpd45_rate"].map(_fmt_rate)
        lines.extend(_markdown_table(view))
    else:
        lines.append("_No coverage rows._")
    lines.extend(["", "Cumulative queue coverage using all emitted rows per method:", ""])
    if len(coverage_all):
        view = coverage_all.copy()
        view["new_dpd45_rate"] = view["new_dpd45_rate"].map(_fmt_rate)
        view["cumulative_dpd45_rate"] = view["cumulative_dpd45_rate"].map(_fmt_rate)
        lines.extend(_markdown_table(view))
    else:
        lines.append("_No coverage rows._")
    lines.extend(["", "## Evidence Entities Across Queue Heads", ""])
    if len(evidence_entities):
        lines.extend(_markdown_table(evidence_entities.head(25)))
    else:
        lines.append("_No shared evidence entities found._")
    lines.extend(["", "## Scenario-First Review Packets", ""])
    lines.append("Residual candidate quality by scenario-specific neighborhood:")
    lines.append("")
    if len(scenario_candidate_summary):
        view = scenario_candidate_summary.copy()
        view["dpd45_rate"] = view["dpd45_rate"].map(_fmt_rate)
        view["fraud_rate"] = view["fraud_rate"].map(_fmt_rate)
        view["scenario_rate"] = view["scenario_rate"].map(_fmt_rate)
        lines.extend(_markdown_table(view))
    else:
        lines.append("_No scenario residual candidates._")
    lines.extend(["", "Top scenario evidence entities:", ""])
    if len(scenario_entity_evidence):
        lines.extend(_markdown_table(scenario_entity_evidence.head(30)))
    else:
        lines.append("_No scenario evidence entities._")
    lines.extend(["", "## Queue Definitions", ""])
    if len(queue_meta):
        for row in queue_meta.itertuples(index=False):
            lines.append(f"- `{row.name}`: {row.description} Runtime: {row.runtime_seconds:.2f}s; rows: {row.rows}.")
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `baseline_residual` is the non-scenario denominator. A useful queue should beat this DPD45 rate.",
            "- `gds_ppr_scenario_fraud_seed` is the cleanest propagation queue because it does not seed from DPD45.",
            "- `gds_ppr_dpd45_seed_upper_bound` is useful for case review, but circular if DPD45 is the target.",
            "- `gds_global_pagerank` is a control. If it ranks high-degree infrastructure rather than bad outcomes, do not promote it.",
            "- `gds_node_similarity_to_risky` is exploratory and slower; require strong lift before considering it for full data.",
            "- Scenario candidates need interpretable evidence, not only a score. Prefer multi-type or entity-centric queues for rule mining.",
            "- A method is more usable when its queue row includes typed evidence columns that an analyst can drill into.",
            "- Scenario mirror validation must pass before reading queue metrics; otherwise the Neo4j mirror is not measuring the current register.",
        ]
    )
    path.write_text("\n".join(lines) + "\n")
    return path


def _markdown_table(df: pd.DataFrame) -> list[str]:
    if df.empty:
        return ["_Empty._"]
    display = df.copy()
    columns = list(display.columns)
    rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in display.iterrows():
        values = [str(row[col]) for col in columns]
        rows.append("| " + " | ".join(values) + " |")
    return rows


def write_signal_catalog(out_dir: Path) -> Path:
    path = out_dir / "signal_catalog.md"
    path.write_text(
        """# Graph Discovery Signal Catalog

Use this as the working checklist for full-data discovery. The sample runner
implements the first pass of each family that is cheap enough to try now.

## 1. Local Neighbor Propagation

- Shared risky neighbors: candidate shares any entity with a scenario/fraud/DPD45 user.
- Multi-witness risky neighbors: same, but require at least two entity types.
- Rarity-weighted neighbors: discount hubs with `1/log(entity_degree + 5)`.
- Best use: case queue and scenario mining.
- Failure mode: one common entity can create false association; require type diversity or rarity.

## 2. Entity-Centric Risk

- Rank bank/device/phone/address/persistent-account nodes by attached DPD45/fraud/scenario concentration.
- Pull residual users touching high-risk entities.
- Best use: explainable new scenario candidates, because the unit of evidence is tangible.
- Failure mode: high-degree shared infrastructure; cap degree and inspect span/freshness.

## 3. Component And Community Risk

- Weakly connected components find hard graph islands.
- Label propagation finds softer communities inside/around components.
- Best use: executive review and case packaging: “this cluster has N users, X% DPD45, Y scenarios.”
- Failure mode: component membership is not a causal rule; it needs drilldown to typed edges.

## 4. Personalized Ranking

- Personalized PageRank from scenario/fraud seeds is the cleanest graph-native propagation score.
- Personalized PageRank from DPD45 seeds is a review upper bound, but circular for DPD45 validation.
- Best use: ranked residual queue.
- Failure mode: score is less interpretable than a rule; use it to find patterns, not ship directly.

## 5. Global Centrality

- Degree/PageRank can reveal hubs and influential users/entities.
- Best use: negative control and hub inspection.
- Failure mode: often ranks infrastructure, not fraud.

## 6. Similarity And Embeddings

- Node similarity ranks users similar to risky users by shared entities.
- FastRP/KNN can generalize this with embeddings later.
- Best use: research queue after cheaper signals.
- Failure mode: expensive at full scale and harder to explain.

## 7. Rule Mining From Queue Heads

For each queue head, extract:

- Which entity types connect the users?
- Are there repeated scarce resources?
- Are identities fresh or aged?
- Does DPD45/fraud concentration persist after excluding existing scenarios?
- Can the evidence be expressed as a conjunctive scenario with disqualifiers?

Only promote a queue insight into a scenario if it survives leakage/as-of checks
outside this Neo4j mirror.
"""
    )
    return path


def run(args: argparse.Namespace) -> None:
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    driver = GraphDatabase.driver(args.uri, auth=(args.user, args.password))
    queue_meta_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    with driver.session(database=args.database) as session:
        ensure_indexes(session)
        inventory = graph_inventory(session)
        scenarios = scenario_inventory(session)
        scenario_validation = scenario_mirror_validation(session, args.store)
        gds_projection = ensure_gds_graph(session)
        truth = truth_table(session)

        inventory.to_csv(out_dir / "inventory.csv", index=False)
        scenarios.to_csv(out_dir / "scenario_inventory.csv", index=False)
        scenario_validation.to_csv(out_dir / "scenario_mirror_validation.csv", index=False)
        gds_projection.to_csv(out_dir / "gds_projection.csv", index=False)
        truth.to_csv(out_dir / "truth_users.csv", index=False)

        queue_frames: dict[str, pd.DataFrame] = {}
        for spec in queue_specs(args.include_slow):
            started = time.perf_counter()
            df = run_df(session, spec.query, spec.params)
            runtime = time.perf_counter() - started
            df.to_csv(out_dir / f"{spec.name}.csv", index=False)
            queue_frames[spec.name] = df
            evidence_columns = [
                c
                for c in df.columns
                if c not in {"candidate_user", "score", "dpd45", "is_fraud", "scenario_any"}
            ]
            queue_meta_rows.append(
                {
                    "name": spec.name,
                    "description": spec.description,
                    "rows": len(df),
                    "runtime_seconds": round(runtime, 3),
                    "requires_gds": spec.requires_gds,
                    "slow": spec.slow,
                    "evidence_columns": ",".join(evidence_columns),
                    "usable_for_review": bool(evidence_columns) or spec.name.startswith("gds_ppr"),
                }
            )
            summary_rows.extend(evaluate_queue(df, spec.name))

        coverage_100 = cumulative_coverage(queue_frames, truth, top_n=100)
        coverage_all = cumulative_coverage(queue_frames, truth, top_n=1_000_000)
        evidence_entities = top_candidate_evidence(session, queue_frames, top_n=100)
        scenario_candidate_summary, scenario_entity_evidence = scenario_packets(session, out_dir)

    driver.close()

    queue_meta = pd.DataFrame(queue_meta_rows)
    summary = pd.DataFrame(summary_rows)
    if len(summary):
        summary = summary.sort_values(
            ["queue", "top_n"],
            key=lambda col: col.map(lambda x: 10**9 if x == "all" else int(x)) if col.name == "top_n" else col,
            kind="stable",
        )
    queue_meta.to_csv(out_dir / "queue_meta.csv", index=False)
    summary.to_csv(out_dir / "queue_summary.csv", index=False)
    coverage_100.to_csv(out_dir / "coverage_top100_by_method.csv", index=False)
    coverage_all.to_csv(out_dir / "coverage_all_by_method.csv", index=False)
    evidence_entities.to_csv(out_dir / "top_candidate_evidence_entities.csv", index=False)
    write_signal_catalog(out_dir)
    write_markdown(
        out_dir,
        inventory=inventory,
        scenarios=scenarios,
        scenario_validation=scenario_validation,
        gds_projection=gds_projection,
        queue_meta=queue_meta,
        summary=summary,
        coverage_100=coverage_100,
        coverage_all=coverage_all,
        evidence_entities=evidence_entities,
        scenario_candidate_summary=scenario_candidate_summary,
        scenario_entity_evidence=scenario_entity_evidence,
    )
    print(f"wrote Neo4j discovery outputs to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uri", default=DEFAULT_URI)
    parser.add_argument("--user", default=DEFAULT_USER)
    parser.add_argument("--password", default=DEFAULT_PASSWORD)
    parser.add_argument("--database", default="neo4j")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--include-slow", action="store_true", help="include node-similarity experiment")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
