"""Export the sample DuckDB graph store as a rebuildable Neo4j mirror.

This POC currently uses DuckDB as the rebuild source and Neo4j as the graph
analysis mirror. The experiment asks whether Cypher/GDS make fraud discovery
and scenario explanation clearer enough to justify a full-data Neo4j run.

Run from the repo root:

    uv run --group fraud python -m projects.fraud_anomaly_detection.codex_poc.export_neo4j_mirror
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.graph.load import DEFAULT_LAYERS, load_graph
from projects.fraud_anomaly_detection.scenarios import SCENARIOS, assign

DEFAULT_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")
DEFAULT_OUT = Path("projects/fraud_anomaly_detection/codex_poc/out/neo4j")
DEFAULT_CLUSTER_DEGREE_CAP = 20

REL_TYPE_BY_ENTITY = {
    "device": "USED_DEVICE",
    "bank": "USED_BANK_ACCOUNT",
    "persistent": "USED_PERSISTENT_ACCOUNT",
    "phone": "USED_PHONE",
    "address": "USED_ADDRESS",
    "email": "USED_EMAIL",
    "ip": "USED_IP",
}

REL_FILE_BY_ENTITY = {
    "device": "used_device_rels.csv",
    "bank": "used_bank_account_rels.csv",
    "persistent": "used_persistent_account_rels.csv",
    "phone": "used_phone_rels.csv",
    "address": "used_address_rels.csv",
    "email": "used_email_rels.csv",
    "ip": "used_ip_rels.csv",
}

ENTITY_REL_TYPES = tuple(REL_TYPE_BY_ENTITY[layer] for layer in DEFAULT_LAYERS)
ENTITY_REL_PATTERN = "|".join(ENTITY_REL_TYPES)

USER_LABEL_COLS = (
    "is_fraud",
    "label_gross_dpd45",
    "label_mature_d45",
    "is_neobank_high_risk_institution",
)


@dataclass(frozen=True)
class ExportResult:
    out_dir: Path
    files: tuple[Path, ...]
    n_users: int
    n_entities: int
    n_edges: int
    n_scenarios: int
    n_scenario_matches: int
    n_clusters: int


def user_node_id(user_id: object) -> str:
    return f"user:{user_id}"


def entity_node_id(entity_type: object, entity_value: object) -> str:
    return f"entity:{entity_type}:{entity_value}"


def scenario_node_id(name: object) -> str:
    return f"scenario:{name}"


def _read_store(store: Path | str, sql: str, params: Iterable[object] = ()) -> pd.DataFrame:
    with duckdb.connect(str(store), read_only=True) as con:
        return con.execute(sql, list(params)).df()


def _bool_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(False).astype(bool)


def _entity_label(entity_type: str) -> str:
    return {
        "bank": "BankAccount",
        "persistent": "PersistentAccount",
        "device": "Device",
        "phone": "Phone",
        "address": "Address",
        "email": "Email",
        "ip": "IP",
    }.get(entity_type, "Entity")


def _scenario_user_flags(base: pd.DataFrame) -> pd.DataFrame:
    raw = assign(base).copy()
    flags = pd.DataFrame({"user_id": base["user_id"].astype(str)})
    scenario_cols = [s.name for s in SCENARIOS]
    for col in scenario_cols:
        raw_col = f"scenario_{col}"
        flags[col] = raw[raw_col] if raw_col in raw.columns else False
    flags["scenario_any"] = raw["scenario_any"] if "scenario_any" in raw.columns else flags[scenario_cols].any(axis=1)
    per_user = flags.groupby("user_id").max().reset_index()
    _bool_columns(per_user, [*scenario_cols, "scenario_any"])
    return per_user


def _user_nodes(store: Path | str) -> tuple[pd.DataFrame, pd.DataFrame]:
    users = _read_store(store, "SELECT * FROM users ORDER BY user_id")
    base = _read_store(store, "SELECT * FROM advances")
    label_cols = [c for c in USER_LABEL_COLS if c in base.columns]
    labels = (
        base[["user_id", *label_cols]]
        .assign(user_id=lambda df: df["user_id"].astype(str))
        .groupby("user_id", as_index=False)
        .max()
    )
    scenario_flags = _scenario_user_flags(base)
    merged = users.assign(user_id=lambda df: df["user_id"].astype(str))
    merged = merged.merge(labels, on="user_id", how="left")
    merged = merged.merge(scenario_flags, on="user_id", how="left")
    scenario_cols = [s.name for s in SCENARIOS]
    _bool_columns(merged, [*label_cols, *scenario_cols, "scenario_any"])
    merged["userNodeId:ID(User-ID)"] = merged["user_id"].map(user_node_id)
    merged[":LABEL"] = merged.apply(
        lambda r: ";".join(
            label
            for label, keep in (
                ("User", True),
                ("FraudUser", bool(r.get("is_fraud", False))),
                ("ScenarioUser", bool(r.get("scenario_any", False))),
                ("Dpd45User", bool(r.get("label_gross_dpd45", False))),
            )
            if keep
        ),
        axis=1,
    )
    rename = {
        "n_advances": "n_advances:int",
        "is_fraud": "is_fraud:boolean",
        "label_gross_dpd45": "label_gross_dpd45:boolean",
        "label_mature_d45": "label_mature_d45:boolean",
        "is_neobank_high_risk_institution": "is_neobank_high_risk_institution:boolean",
        "scenario_any": "scenario_any:boolean",
    }
    for col in scenario_cols:
        rename[col] = f"scenario_{col}:boolean"
    columns = [
        "userNodeId:ID(User-ID)",
        ":LABEL",
        "user_id",
        "n_advances:int",
        "first_seen_ts",
        "last_seen_ts",
        "identity_created_time",
        "is_fraud:boolean",
        "label_gross_dpd45:boolean",
        "label_mature_d45:boolean",
        "is_neobank_high_risk_institution:boolean",
        "scenario_any:boolean",
        *[f"scenario_{col}:boolean" for col in scenario_cols],
    ]
    out = merged.rename(columns=rename)
    return out[[c for c in columns if c in out.columns]], scenario_flags


def _entity_nodes(store: Path | str) -> pd.DataFrame:
    entities = _read_store(store, "SELECT * FROM entities ORDER BY entity_type, entity_value")
    entities["entityNodeId:ID(Entity-ID)"] = [
        entity_node_id(t, v) for t, v in zip(entities["entity_type"], entities["entity_value"])
    ]
    entities[":LABEL"] = [
        f"Entity;{_entity_label(str(t))}" for t in entities["entity_type"]
    ]
    out = entities.rename(
        columns={
            "n_users": "n_users:int",
            "n_advances": "n_advances:int",
        }
    )
    return out[
        [
            "entityNodeId:ID(Entity-ID)",
            ":LABEL",
            "entity_type",
            "entity_value",
            "n_users:int",
            "n_advances:int",
            "first_seen_ts",
            "last_seen_ts",
        ]
    ]


def _iso_local_datetime(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values).dt.strftime("%Y-%m-%dT%H:%M:%S.%f")


def _typed_entity_relationships(
    store: Path | str,
    layers: tuple[str, ...],
    sources: tuple[str, ...],
    max_edges: int | None,
) -> dict[str, pd.DataFrame]:
    layer_sql = ", ".join("?" for _ in layers)
    source_sql = ", ".join("?" for _ in sources)
    grouped = _read_store(
        store,
        f"""
        SELECT user_id, entity_type, entity_value,
               count(*) AS n_events,
               string_agg(DISTINCT source, '|') AS sources,
               min(ts) AS first_ts,
               max(ts) AS last_ts
        FROM edges
        WHERE entity_type IN ({layer_sql})
          AND source IN ({source_sql})
        GROUP BY 1, 2, 3
        ORDER BY entity_type, user_id, entity_value
        """,
        [*layers, *sources],
    )
    columns = [
        ":START_ID(User-ID)",
        ":END_ID(Entity-ID)",
        "n_events:int",
        "sources",
        "first_ts:localdatetime",
        "last_ts:localdatetime",
    ]
    out: dict[str, pd.DataFrame] = {}
    for layer in layers:
        layer_df = grouped[grouped["entity_type"] == layer].copy()
        if max_edges is not None:
            layer_df = layer_df.head(int(max_edges))
        if layer_df.empty:
            out[REL_FILE_BY_ENTITY[layer]] = pd.DataFrame(columns=columns)
            continue
        out[REL_FILE_BY_ENTITY[layer]] = pd.DataFrame(
            {
                ":START_ID(User-ID)": layer_df["user_id"].map(user_node_id),
                ":END_ID(Entity-ID)": [
                    entity_node_id(t, v)
                    for t, v in zip(layer_df["entity_type"], layer_df["entity_value"])
                ],
                "n_events:int": layer_df["n_events"],
                "sources": layer_df["sources"],
                "first_ts:localdatetime": _iso_local_datetime(layer_df["first_ts"]),
                "last_ts:localdatetime": _iso_local_datetime(layer_df["last_ts"]),
            }
        )
    return out


def _scenario_nodes() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenarioNodeId:ID(Scenario-ID)": scenario_node_id(s.name),
                ":LABEL": "Scenario",
                "name": s.name,
                "title": s.title,
                "tier": s.tier,
                "status": s.status,
                "theory": s.theory,
            }
            for s in SCENARIOS
        ]
    )


def _scenario_relationships(scenario_flags: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for s in SCENARIOS:
        matched = scenario_flags[scenario_flags[s.name].fillna(False)]
        for user_id in matched["user_id"]:
            rows.append(
                {
                    ":START_ID(User-ID)": user_node_id(user_id),
                    ":END_ID(Scenario-ID)": scenario_node_id(s.name),
                    "matched:boolean": True,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[":START_ID(User-ID)", ":END_ID(Scenario-ID)", "matched:boolean"],
    )


def cluster_node_id(cluster_id: object) -> str:
    return f"cluster:{cluster_id}"


def _review_cluster_artifacts(
    store: Path | str,
    layers: tuple[str, ...],
    sources: tuple[str, ...],
    degree_cap: int = DEFAULT_CLUSTER_DEGREE_CAP,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    g = load_graph(store, layers=layers, sources=sources, degree_cap=degree_cap)
    cluster_rows = []
    member_rows = []
    for component_id, members in enumerate(g.connected_components()):
        vertices = [g.vs[i] for i in members]
        users = [v for v in vertices if v["kind"] == "user"]
        entities = [v for v in vertices if v["kind"] != "user"]
        if len(users) < 2:
            continue

        n_mature = sum(bool(v["label_mature_d45"]) for v in users)
        dpd45_users = sum(bool(v["label_gross_dpd45"]) for v in users)
        fraud_users = sum(bool(v["is_fraud"]) for v in users)
        scenario_users = sum(bool(v["scenario_any"]) for v in users)
        entity_types = sorted({str(v["kind"]) for v in entities})
        dpd45_rate = dpd45_users / n_mature if n_mature else 0.0
        scenario_rate = scenario_users / len(users)
        # Review score is intentionally transparent, not a model: outcome
        # concentration first, then scenario overlap, then type diversity.
        review_score = (
            dpd45_rate * min(n_mature, 50)
            + scenario_rate * min(len(users), 50) * 0.5
            + len(entity_types) * 2
            + min(len(users), 25) * 0.1
        )
        labels = ["ReviewCluster"]
        if n_mature and dpd45_rate >= 0.5:
            labels.append("HighDpd45Cluster")
        if scenario_users:
            labels.append("ScenarioCluster")
        cluster_id = str(component_id)
        cluster_rows.append(
            {
                "clusterNodeId:ID(Cluster-ID)": cluster_node_id(cluster_id),
                ":LABEL": ";".join(labels),
                "cluster_id": cluster_id,
                "n_users:int": len(users),
                "n_entities:int": len(entities),
                "n_types:int": len(entity_types),
                "entity_types": "|".join(entity_types),
                "n_mature_users:int": n_mature,
                "dpd45_users:int": dpd45_users,
                "dpd45_user_rate:float": round(dpd45_rate, 6),
                "fraud_users:int": fraud_users,
                "scenario_users:int": scenario_users,
                "scenario_user_rate:float": round(scenario_rate, 6),
                "review_score:float": round(review_score, 6),
            }
        )
        for user in users:
            member_rows.append(
                {
                    ":START_ID(User-ID)": user_node_id(user["raw_id"]),
                    ":END_ID(Cluster-ID)": cluster_node_id(cluster_id),
                    "member:boolean": True,
                }
            )

    clusters = pd.DataFrame(cluster_rows)
    if not clusters.empty:
        clusters = clusters.sort_values(
            ["review_score:float", "dpd45_user_rate:float", "n_users:int"],
            ascending=False,
            kind="stable",
        )
    members = pd.DataFrame(
        member_rows,
        columns=[":START_ID(User-ID)", ":END_ID(Cluster-ID)", "member:boolean"],
    )
    return clusters, members


def write_cypher_playbook(out_dir: Path) -> tuple[Path, ...]:
    cypher_dir = out_dir / "cypher"
    cypher_dir.mkdir(parents=True, exist_ok=True)
    rel_pattern = ENTITY_REL_PATTERN
    gds_rel_projection = ",\n  ".join(
        f"{rel}: {{orientation: 'UNDIRECTED'}}" for rel in ENTITY_REL_TYPES
    )
    files = {
        "00_top_suspicious_clusters.cypher": """// First screen: ranked review clusters, not the whole graph.
// Pick a cluster_id from this table, then run 01_cluster_ring_view.cypher.
MATCH (c:ReviewCluster)
RETURN c.cluster_id AS cluster_id,
       c.review_score AS review_score,
       c.n_users AS n_users,
       c.n_mature_users AS n_mature_users,
       c.dpd45_users AS dpd45_users,
       c.dpd45_user_rate AS dpd45_user_rate,
       c.fraud_users AS fraud_users,
       c.scenario_users AS scenario_users,
       c.entity_types AS entity_types
ORDER BY review_score DESC, dpd45_user_rate DESC, n_users DESC
LIMIT 50;
""",
        "01_cluster_ring_view.cypher": f"""// Visual inspection: suspicious cluster -> users -> typed shared entities.
// Set $cluster_id from 00_top_suspicious_clusters.cypher.
MATCH (c:ReviewCluster {{cluster_id: $cluster_id}})<-[:IN_REVIEW_CLUSTER]-(u:User)
OPTIONAL MATCH path = (u)-[r:{rel_pattern}]->(e:Entity)
RETURN path
LIMIT 1000;
""",
        "01_scenario_overview.cypher": """// Scenario inventory: what exists, how many users match, and what story it tells.
MATCH (s:Scenario)
OPTIONAL MATCH (s)<-[:MATCHED_SCENARIO]-(u:User)
RETURN s.name AS scenario, s.title AS title, s.tier AS tier, s.status AS status,
       count(DISTINCT u) AS matched_users, s.theory AS theory
ORDER BY matched_users DESC, scenario;
""",
        "02_scenario_to_users.cypher": f"""// Scenario overlay: scenario -> users -> typed shared-entity evidence.
// Set $scenario, for example: 'ring_identity_burst'.
MATCH (s:Scenario {{name: $scenario}})<-[:MATCHED_SCENARIO]-(u:User)
OPTIONAL MATCH (u)-[r:{rel_pattern}]->(e:Entity)<-[r2:{rel_pattern}]-(neighbor:User)
WHERE neighbor <> u
WITH s, u,
     count(DISTINCT e) AS connected_entities,
     collect(DISTINCT e.entity_type) AS shared_entity_types,
     count(DISTINCT neighbor) AS neighbor_users,
     count(DISTINCT CASE WHEN neighbor.is_fraud THEN neighbor END) AS fraud_neighbors,
     count(DISTINCT CASE WHEN neighbor.label_gross_dpd45 THEN neighbor END) AS dpd45_neighbors,
     count(DISTINCT CASE WHEN neighbor.scenario_any THEN neighbor END) AS scenario_neighbors
RETURN s.name AS scenario, u.user_id AS user_id, labels(u) AS user_labels,
       u.is_fraud AS is_fraud, u.label_gross_dpd45 AS dpd45,
       connected_entities, shared_entity_types, neighbor_users,
       fraud_neighbors, dpd45_neighbors, scenario_neighbors
ORDER BY fraud_neighbors DESC, scenario_neighbors DESC, neighbor_users DESC
LIMIT 100;
""",
        "03_user_ego_ring.cypher": f"""// Ring view: click a user from a cluster/scenario table, then inspect its ego graph.
// 4 relationship hops = 2 user-hops in the bipartite user<->entity graph.
// Set $user_id to a raw user id.
MATCH path = (u:User {{user_id: $user_id}})-[:{rel_pattern}*1..4]-(n)
RETURN path
LIMIT 500;
""",
        "04_gds_component_and_pagerank.cypher": f"""// Optional GDS workflow after installing/enabling Neo4j Graph Data Science.
// 1) Project the bipartite graph as undirected.
CALL gds.graph.drop('fraud_mirror', false) YIELD graphName
RETURN graphName;

CALL gds.graph.project(
  'fraud_mirror',
  ['User', 'Entity'],
  {{
  {gds_rel_projection}
  }}
);

// 2) Connected components for ring composition.
CALL gds.wcc.stream('fraud_mirror')
YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS n, componentId
WHERE n:User
RETURN componentId, count(*) AS n_users,
       sum(CASE WHEN n.is_fraud THEN 1 ELSE 0 END) AS fraud_users,
       sum(CASE WHEN n.scenario_any THEN 1 ELSE 0 END) AS scenario_users
ORDER BY fraud_users DESC, n_users DESC
LIMIT 50;

// 3) PageRank-style suspicion lens. For true PPR, seed projection/config
// depends on the licensed/installed GDS version; this global rank is only a
// quick sanity check against obvious high-connectivity users/entities.
CALL gds.pageRank.stream('fraud_mirror')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score
WHERE n:User AND coalesce(n.scenario_any, false) = false
RETURN n.user_id AS user_id, n.is_fraud AS is_fraud, n.label_gross_dpd45 AS dpd45, score
ORDER BY score DESC
LIMIT 100;
""",
        "05_entity_drilldown.cypher": f"""// Entity drilldown: answer "this bank/device dot connects which users, and why?"
// Set params, for example:
// :param entity_type => 'bank'
// :param entity_value => '063104626-3158304005'
MATCH (e:Entity {{entity_type: $entity_type, entity_value: $entity_value}})<-[r:{rel_pattern}]-(u:User)
OPTIONAL MATCH (u)-[:MATCHED_SCENARIO]->(s:Scenario)
RETURN e.entity_type AS entity_type, e.entity_value AS entity_value,
       u.user_id AS user_id, labels(u) AS user_labels,
       u.is_fraud AS is_fraud, u.label_gross_dpd45 AS dpd45,
       collect(DISTINCT s.name) AS matched_scenarios
ORDER BY is_fraud DESC, dpd45 DESC, size(matched_scenarios) DESC, user_id
LIMIT 200;
""",
        "06_discovery_candidates.cypher": f"""// Discovery candidates: unscenarioed users touching entities already tied to
// scenario/fraud users. This is the Neo4j version of "show me nearby misses."
MATCH (seed:User)-[r:{rel_pattern}]->(e:Entity)<-[r2:{rel_pattern}]-(candidate:User)
WHERE candidate <> seed
  AND candidate.scenario_any = false
  AND (seed.scenario_any = true OR seed.is_fraud = true OR seed.label_gross_dpd45 = true)
WITH candidate,
     collect(DISTINCT e.entity_type) AS shared_entity_types,
     count(DISTINCT e) AS shared_entities,
     count(DISTINCT seed) AS risky_neighbors,
     count(DISTINCT CASE WHEN seed.is_fraud THEN seed END) AS fraud_neighbors,
     count(DISTINCT CASE WHEN seed.scenario_any THEN seed END) AS scenario_neighbors
RETURN candidate.user_id AS candidate_user, labels(candidate) AS candidate_labels,
       candidate.is_fraud AS is_fraud, candidate.label_gross_dpd45 AS dpd45,
       shared_entity_types, shared_entities, risky_neighbors,
       fraud_neighbors, scenario_neighbors
ORDER BY fraud_neighbors DESC, scenario_neighbors DESC, risky_neighbors DESC, shared_entities DESC
LIMIT 100;
""",
    }
    written = []
    for name, body in files.items():
        path = cypher_dir / name
        path.write_text(body)
        written.append(path)
    return tuple(written)


def _write_import_command(out_dir: Path) -> Path:
    relationship_lines = "\n".join(
        f"  --relationships={REL_TYPE_BY_ENTITY[layer]}={REL_FILE_BY_ENTITY[layer]} \\"
        for layer in DEFAULT_LAYERS
    )
    path = out_dir / "neo4j_admin_import.sh"
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing this file, with Neo4j stopped.
# Database name is arbitrary; use a disposable mirror name for each rebuild.
DB_NAME="${{1:-fraud_mirror}}"

neo4j-admin database import full "$DB_NAME" \\
  --overwrite-destination=true \\
  --nodes=User=users.csv \\
  --nodes=Entity=entities.csv \\
  --nodes=ReviewCluster=clusters.csv \\
  --nodes=Scenario=scenarios.csv \\
{relationship_lines}
  --relationships=IN_REVIEW_CLUSTER=cluster_member_rels.csv \\
  --relationships=MATCHED_SCENARIO=scenario_match_rels.csv
"""
    )
    return path


def _write_neo4j_csv(df: pd.DataFrame, path: Path) -> None:
    out = df.copy()
    for col in out.columns:
        if col.endswith(":boolean"):
            out[col] = out[col].fillna(False).map(lambda v: "true" if bool(v) else "false")
    out.to_csv(path, index=False)


def _write_summary(result: ExportResult, layers: tuple[str, ...], sources: tuple[str, ...]) -> Path:
    path = result.out_dir / "summary.md"
    path.write_text(
        f"""# Neo4j Mirror Export Summary

## User Story

As a fraud analyst, I want to start from a named scenario, see who matched it,
open a user's local ring, and inspect connected entity types and fraud/outcome
composition without writing Python.

## Suspicious Clusters -> Ring

1. Open `00_top_suspicious_clusters.cypher` to pick a review cluster.
2. Run `01_cluster_ring_view.cypher` with `$cluster_id` to inspect it.
3. Use scenarios as a filter/overlay, not as the default visible graph center.
4. If GDS is available, run `04_gds_component_and_pagerank.cypher` for
   component composition and a first graph-native ranking check.

## Exported Shape

- Users: {result.n_users:,}
- Entities: {result.n_entities:,}
- Aggregated user-entity relationships exported: {result.n_edges:,}
- Scenarios: {result.n_scenarios:,}
- MATCHED_SCENARIO relationships: {result.n_scenario_matches:,}
- Review clusters: {result.n_clusters:,}
- Layers: {", ".join(layers)}
- Sources: {", ".join(sources)}

## Performance Gate

This POC should earn continuation only if a sample rebuild plus import is
mechanical, the scenario-to-ring workflow is clearer than the current Python
artifact flow, and graph-native queries/GDS expose useful fraud structure
without loading the whole graph into the browser.

The Browser view should stay bounded: scenario queue -> selected user or
component -> ego/ring view. A 10M+ node full-graph visualization is not a
usable analyst surface.

## Licensing Note

Neo4j Community + GDS Community is the working evaluation path. GDS Community
enforces a maximum concurrency of 4 at runtime. Enterprise/Bloom only need
follow-up if full-data runtime, security, or shared analyst UI requirements
force them.
"""
    )
    return path


def export_bundle(
    store: Path | str = DEFAULT_STORE,
    out_dir: Path | str = DEFAULT_OUT,
    *,
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    sources: tuple[str, ...] = ("advance", "link"),
    max_edges: int | None = None,
) -> ExportResult:
    store = Path(store)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.csv"):
        stale.unlink()

    users, scenario_flags = _user_nodes(store)
    entities = _entity_nodes(store)
    typed_rels = _typed_entity_relationships(store, layers, sources, max_edges)
    clusters, cluster_members = _review_cluster_artifacts(store, layers, sources)
    scenarios = _scenario_nodes()
    scenario_rels = _scenario_relationships(scenario_flags)

    frames = {
        "users.csv": users,
        "entities.csv": entities,
        "clusters.csv": clusters,
        "scenarios.csv": scenarios,
        "cluster_member_rels.csv": cluster_members,
        "scenario_match_rels.csv": scenario_rels,
        **typed_rels,
    }
    files = []
    for name, df in frames.items():
        path = out / name
        _write_neo4j_csv(df, path)
        files.append(path)

    result = ExportResult(
        out_dir=out,
        files=tuple(files),
        n_users=len(users),
        n_entities=len(entities),
        n_edges=sum(len(df) for df in typed_rels.values()),
        n_scenarios=len(scenarios),
        n_scenario_matches=len(scenario_rels),
        n_clusters=len(clusters),
    )
    summary = _write_summary(result, layers, sources)
    import_cmd = _write_import_command(out)
    cypher_files = write_cypher_playbook(out)
    return ExportResult(
        out_dir=out,
        files=(*result.files, summary, import_cmd, *cypher_files),
        n_users=result.n_users,
        n_entities=result.n_entities,
        n_edges=result.n_edges,
        n_scenarios=result.n_scenarios,
        n_scenario_matches=result.n_scenario_matches,
        n_clusters=result.n_clusters,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-edges", type=int, default=None)
    args = parser.parse_args()

    result = export_bundle(args.store, args.out, max_edges=args.max_edges)
    print(f"wrote Neo4j mirror bundle to {result.out_dir}")
    print(f"users={result.n_users:,} entities={result.n_entities:,} "
          f"typed_rels={result.n_edges:,} clusters={result.n_clusters:,} "
          f"scenario_matches={result.n_scenario_matches:,}")
    print(f"start with: {result.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
