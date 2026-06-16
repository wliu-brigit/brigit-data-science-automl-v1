# neo4j_codex — Neo4j-graph fraud-control unit

A self-contained unit for the fraud **discovery → plug-the-hole → monitor**
control system. Development stays inside this folder. Importing the live repo
scenario package is fine and expected because `scenarios/register.yaml` remains
canonical. Active graph discovery is different: it is Neo4j-backed through
`control/graph/`, not the old Python/igraph graph backend. The one hard rule is
about `archived/`: borrow from it freely *while developing*, but the finished
unit must carry **no dependency on `archived/`** — copy anything useful into
`control/` and drop the import (guard tests enforce this). The goal is freedom
to try the cleanest design while keeping the forward path obvious.

## Layout

- **`docs/`** — the design and the build plan:
  - `CONTROL_SYSTEM_DESIGN.md` — the system design (discovery / plug / monitor,
    persistence, the extensible-skeleton goal).
  - `SCHEMA_DESIGN.md` — the graph schema working notes.
  - `WALKING_SKELETON_PLAN.md` — the TDD build plan for the walking skeleton.

  The durable guiding principles live one level up in
  [`../PRINCIPLES.md`](../PRINCIPLES.md) (kept project-level on purpose — the
  always-here doc; flag if you'd rather it move in here).
- **`control/`** — the walking-skeleton system code: discovery adapters,
  finding snapshots, plug derivation, two-state holdout, monitoring, and the
  orchestrator.
- **`archived/`** — the prior Neo4j-mirror POC, **reference only**. No
  obligation to use or maintain it; copy snippets out if useful. See
  `archived/README.md`.

## Control-loop workflow

The operator entry point is the full fraud-control loop report:

```bash
NEO4J_PASSWORD=fraudpocpass uv run --with neo4j --group fraud python -m \
  projects.fraud_anomaly_detection.neo4j_codex.control.control_loop_report \
  --store projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb \
  --out-dir projects/fraud_anomaly_detection/neo4j_codex/reports \
  --refresh-key fraud_control_loop_report \
  --neo4j-uri bolt://localhost:7687 \
  --neo4j-user neo4j \
  --neo4j-database neo4j
```

The report expects a local Neo4j mirror with GDS available. For the current
sample workflow, rebuild and start it first with:

```bash
bash projects/fraud_anomaly_detection/neo4j_codex/archived/scripts/setup_neo4j.sh
```

That report:

- evaluates every scenario in the canonical scenario register;
- runs graph screens in Neo4j with Cypher/GDS, then screens them by total users,
  net-new users beyond the scenario union, and marginal net-new users after
  dedupe;
- reports graph status counts, including `review_only` pockets that are real
  discovery leads but not eligible for plug derivation;
- includes only `promoted_to_plug_derivation` graph rows in the final discovery
  union used to derive plugs;
- derives plug candidates from the final State A discovery union;
- reports State A and holdout buckets for `covered_discovery`,
  `uncovered_discovery`, and `outside_discovery`.

Use `--include-status` to filter displayed graph rows without changing the
underlying evaluation:

```bash
NEO4J_PASSWORD=fraudpocpass uv run --with neo4j --group fraud python -m \
  projects.fraud_anomaly_detection.neo4j_codex.control.control_loop_report \
  --include-status review_only \
  --neo4j-uri bolt://localhost:7687
```

Valid graph statuses are:

- `promoted_to_plug_derivation` — selected into the final union and plug
  derivation.
- `review_only` — visible graph discovery/review evidence, excluded from plug
  derivation because method metadata is not promotion-safe.
- `below_min_marginal_users` — promotion-safe metadata, but too little marginal
  net-new volume.
- `below_min_marginal_dpd45_user_rate` — promotion-safe metadata, but below the
  marginal precision gate.

Generated files under `reports/` are ignored by git; rerun the command whenever
the scenario register, graph screens, thresholds, or sample store changes.

Each discovery method exposes method metadata:

- `method_type`: scenario, graph, model, or subgroup.
- `time_semantics`: snapshot_review, leakfree_asof, or production_safe.
- `promotion_tier`: evidence_only, review_queue, or plug_candidate.
- `enforcement_projection`: entity_key, scenario_rule, or none.

This keeps broad discovery safe: a method can be useful for review without
being eligible for plug derivation. `plug_candidate` is accepted only when
`time_semantics` is `leakfree_asof` or `production_safe`, and when
`enforcement_projection` is not `none`.

Plug validation buckets are intentionally named from the operator perspective:

- `covered_discovery` — discovery users the burned keys would catch.
- `uncovered_discovery` — discovery users with no deployable burned key.
- `outside_discovery` — users touched by the burned keys who were not in the
  discovery union. Its DPD45 rate tells whether the plug is overreaching or
  finding extra bad users.

### Adding a scenario discovery method

1. Add or update the scenario definition in
   `../scenarios/register.yaml`. That file remains canonical; do not fork
   scenario logic into `control/`.
2. Validate the scenario register with the existing scenario tests/validation.
3. Add or extend tests under `neo4j_codex/tests/control/` so the scenario appears
   in `control_loop_report` and its State A / holdout panels.

### Adding a graph / Neo4j discovery pattern

1. Add a reviewed screen descriptor in
   `control/discovery/graph_screen_catalog.py`.
2. Implement the Cypher/GDS query in `control/graph/methods.py`. Python should
   orchestrate and normalize results only; graph traversal, communities,
   ranking, and shared-resource expansion belong in Neo4j.
3. Give the screen explicit method metadata. Snapshot-only graph screens stay in
   `review_queue`; only leak-free/as-of or production-safe graph methods may be
   promoted to `plug_candidate`.
4. Add tests under `neo4j_codex/tests/control/` for the graph screen catalog,
   Neo4j query registry, graph status/filter behavior, and the end-to-end
   report. Active control code must not import from `neo4j_codex.archived` or
   the old Python graph backend.

## Build posture

- **Sample data only.** Build and test against
  `../data/graph/fraud_graph.duckdb` (the local sample). Full v3 / warehouse
  work is out of scope here — it needs VPN; see the design's build-staging
  section.
- **No `archived/` dependency at the end.** Borrow from `archived/` during
  development if useful, but the finished `control/` copies what it needs in —
  the plan's final guard test asserts nothing in `control/` imports
  `archived/`. Scenario definitions remain canonical in the repo scenario
  package; active graph discovery stays inside the Neo4j-backed control graph
  boundary.
- **TDD on the sample**, per the plan.
