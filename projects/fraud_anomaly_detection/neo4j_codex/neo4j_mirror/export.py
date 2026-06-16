"""Export the sample DuckDB graph store as a rebuildable Neo4j mirror.

This module only pours graph facts into Neo4j. Discovery logic lives in
``neo4j_codex.control.graph`` and runs through Cypher/GDS against the mirror.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd

from projects.fraud_anomaly_detection.scenarios import SCENARIOS, assign

DEFAULT_STORE = Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb")
DEFAULT_OUT = Path("projects/fraud_anomaly_detection/neo4j_codex/out/neo4j")
DEFAULT_LAYERS = ("device", "bank", "persistent", "phone", "address")
DEFAULT_SOURCES = ("advance", "link")

REL_TYPE_BY_ENTITY = {
    "device": "USED_DEVICE",
    "bank": "USED_BANK_ACCOUNT",
    "persistent": "USED_PERSISTENT_ACCOUNT",
    "phone": "USED_PHONE",
    "address": "USED_ADDRESS",
}

REL_FILE_BY_ENTITY = {
    "device": "used_device_rels.csv",
    "bank": "used_bank_account_rels.csv",
    "persistent": "used_persistent_account_rels.csv",
    "phone": "used_phone_rels.csv",
    "address": "used_address_rels.csv",
}

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


def _outcome_counts(base: pd.DataFrame) -> pd.DataFrame:
    mature = (
        base["label_mature_d45"].fillna(False).astype(bool)
        if "label_mature_d45" in base.columns
        else pd.Series(False, index=base.index)
    )
    bad = (
        base["label_gross_dpd45"].fillna(False).astype(bool)
        if "label_gross_dpd45" in base.columns
        else pd.Series(False, index=base.index)
    )
    counts = pd.DataFrame(
        {
            "user_id": base["user_id"].astype(str),
            "n_mature_advances": mature.astype(int),
            "n_bad_advances": (mature & bad).astype(int),
        }
    )
    return counts.groupby("user_id", as_index=False).sum()


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
    merged = merged.merge(_outcome_counts(base), on="user_id", how="left")
    for col in ("n_mature_advances", "n_bad_advances"):
        merged[col] = merged[col].fillna(0).astype(int)
    mature = merged["n_mature_advances"]
    merged["bad_advance_rate"] = (
        merged["n_bad_advances"] / mature.where(mature > 0)
    ).round(6)
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
        "n_mature_advances": "n_mature_advances:int",
        "n_bad_advances": "n_bad_advances:int",
        "bad_advance_rate": "bad_advance_rate:float",
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
        "n_mature_advances:int",
        "n_bad_advances:int",
        "bad_advance_rate:float",
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


def _entity_nodes(store: Path | str, layers: tuple[str, ...]) -> pd.DataFrame:
    layer_sql = ", ".join("?" for _ in layers)
    entities = _read_store(
        store,
        f"""
        SELECT *
        FROM entities
        WHERE entity_type IN ({layer_sql})
        ORDER BY entity_type, entity_value
        """,
        layers,
    )
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


def write_cypher_playbook(out_dir: Path) -> tuple[Path, ...]:
    cypher_dir = out_dir / "cypher"
    cypher_dir.mkdir(parents=True, exist_ok=True)
    rel_pattern = "|".join(REL_TYPE_BY_ENTITY[layer] for layer in DEFAULT_LAYERS)
    gds_rel_projection = ",\n  ".join(
        f"{REL_TYPE_BY_ENTITY[layer]}: {{orientation: 'UNDIRECTED'}}"
        for layer in DEFAULT_LAYERS
    )
    files = {
        "00_scenario_overview.cypher": """// Scenario inventory: what exists, how many users match, and what story it tells.
MATCH (s:Scenario)
OPTIONAL MATCH (s)<-[:MATCHED_SCENARIO]-(u:User)
RETURN s.name AS scenario, s.title AS title, s.tier AS tier, s.status AS status,
       count(DISTINCT u) AS matched_users, s.theory AS theory
ORDER BY matched_users DESC, scenario;
""",
        "01_scenario_neighborhood.cypher": f"""// Scenario overlay: scenario -> users -> typed shared-entity evidence.
// Set $scenario, for example: 'ring_identity_burst'.
MATCH (s:Scenario {{name: $scenario}})<-[:MATCHED_SCENARIO]-(seed:User)
      -[r:{rel_pattern}]->(e:Entity)<-[r2:{rel_pattern}]-(candidate:User)
WHERE candidate <> seed
RETURN s.name AS scenario,
       candidate.user_id AS candidate_user,
       labels(candidate) AS candidate_labels,
       candidate.is_fraud AS is_fraud,
       candidate.label_gross_dpd45 AS dpd45,
       count(DISTINCT seed) AS seed_users,
       count(DISTINCT e) AS shared_entities,
       collect(DISTINCT e.entity_type) AS shared_entity_types
ORDER BY seed_users DESC, shared_entities DESC, candidate_user
LIMIT 100;
""",
        "02_user_ego_ring.cypher": f"""// Ring view: inspect a user's local user/entity graph.
// 4 relationship hops = 2 user-hops in the bipartite user<->entity graph.
// Set $user_id to a raw user id.
MATCH path = (u:User {{user_id: $user_id}})-[:{rel_pattern}*1..4]-(n)
RETURN path
LIMIT 500;
""",
        "03_entity_drilldown.cypher": f"""// Entity drilldown: this bank/device/etc connects which users, and why?
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
        "04_gds_component_and_pagerank.cypher": f"""// Optional GDS workflow: components and a first global ranking check.
CALL gds.graph.drop('fraud_mirror', false) YIELD graphName
RETURN graphName;

CALL gds.graph.project(
  'fraud_mirror',
  ['User', 'Entity'],
  {{
  {gds_rel_projection}
  }}
);

CALL gds.wcc.stream('fraud_mirror')
YIELD nodeId, componentId
WITH gds.util.asNode(nodeId) AS n, componentId
WHERE n:User
RETURN componentId, count(*) AS n_users,
       sum(CASE WHEN n.is_fraud THEN 1 ELSE 0 END) AS fraud_users,
       sum(CASE WHEN n.scenario_any THEN 1 ELSE 0 END) AS scenario_users
ORDER BY fraud_users DESC, n_users DESC
LIMIT 50;

CALL gds.pageRank.stream('fraud_mirror')
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS n, score
WHERE n:User
RETURN n.user_id AS user_id, n.is_fraud AS is_fraud, n.label_gross_dpd45 AS dpd45, score
ORDER BY score DESC
LIMIT 100;
""",
    }
    written = []
    for name, body in files.items():
        path = cypher_dir / name
        path.write_text(body)
        written.append(path)
    return tuple(written)


def _write_import_command(out_dir: Path, layers: tuple[str, ...]) -> Path:
    relationship_lines = "\n".join(
        f"  --relationships={REL_TYPE_BY_ENTITY[layer]}={REL_FILE_BY_ENTITY[layer]} \\"
        for layer in layers
    )
    path = out_dir / "neo4j_admin_import.sh"
    path.write_text(
        f"""#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing this file, with Neo4j stopped.
DB_NAME="${{1:-fraud_mirror}}"

neo4j-admin database import full "$DB_NAME" \\
  --overwrite-destination=true \\
  --nodes=User=users.csv \\
  --nodes=Entity=entities.csv \\
  --nodes=Scenario=scenarios.csv \\
{relationship_lines}
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

## Purpose

This mirror supports the fraud-control report. DuckDB remains the sample source
of record; Neo4j is the graph execution surface for Cypher/GDS discovery.

## Exported Shape

- Users: {result.n_users:,}
- Entities: {result.n_entities:,}
- Aggregated user-entity relationships exported: {result.n_edges:,}
- Scenarios: {result.n_scenarios:,}
- MATCHED_SCENARIO relationships: {result.n_scenario_matches:,}
- Layers: {", ".join(layers)}
- Sources: {", ".join(sources)}

## Operator Flow

1. Rebuild the local mirror with `neo4j_mirror/scripts/setup_neo4j.sh`.
2. Run `control.control_loop_report` against `bolt://localhost:7687`.
3. Use the generated Cypher files for manual drilldown only; the report owns
   repeatable discovery evaluation.
"""
    )
    return path


def export_bundle(
    store: Path | str = DEFAULT_STORE,
    out_dir: Path | str = DEFAULT_OUT,
    *,
    layers: tuple[str, ...] = DEFAULT_LAYERS,
    sources: tuple[str, ...] = DEFAULT_SOURCES,
    max_edges: int | None = None,
) -> ExportResult:
    store = Path(store)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob("*.csv"):
        stale.unlink()

    users, scenario_flags = _user_nodes(store)
    entities = _entity_nodes(store, layers)
    typed_rels = _typed_entity_relationships(store, layers, sources, max_edges)
    scenarios = _scenario_nodes()
    scenario_rels = _scenario_relationships(scenario_flags)

    frames = {
        "users.csv": users,
        "entities.csv": entities,
        "scenarios.csv": scenarios,
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
    )
    summary = _write_summary(result, layers, sources)
    import_cmd = _write_import_command(out, layers)
    cypher_files = write_cypher_playbook(out)
    return ExportResult(
        out_dir=out,
        files=(*result.files, summary, import_cmd, *cypher_files),
        n_users=result.n_users,
        n_entities=result.n_entities,
        n_edges=result.n_edges,
        n_scenarios=result.n_scenarios,
        n_scenario_matches=result.n_scenario_matches,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--max-edges", type=int, default=None)
    args = parser.parse_args()

    result = export_bundle(args.store, args.out, max_edges=args.max_edges)
    print(f"wrote Neo4j mirror bundle to {result.out_dir}")
    print(
        f"users={result.n_users:,} entities={result.n_entities:,} "
        f"typed_rels={result.n_edges:,} scenario_matches={result.n_scenario_matches:,}"
    )
    print(f"start with: {result.out_dir / 'summary.md'}")


if __name__ == "__main__":
    main()

