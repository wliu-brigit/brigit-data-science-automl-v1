# codex_poc — Neo4j-graph fraud-control unit

A self-contained unit for the fraud **discovery → plug-the-hole → monitor**
control system. Development stays inside this folder. Importing the live repo
packages (`scenarios`, `graph`, `analysis`) is fine and expected. The one hard
rule is about `archived/`: borrow from it freely *while developing*, but the
finished unit must carry **no dependency on `archived/`** — copy anything
useful into `control/` and drop the import (a guard test in the plan's final
task enforces this). The goal is freedom to try the cleanest design.

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

The built skeleton is intentionally small but end-to-end. The entry point is
`run_skeleton`:

```bash
uv run --group fraud python - <<'PY'
from pathlib import Path
from pprint import pprint

from projects.fraud_anomaly_detection.codex_poc.control.config import ControlConfig
from projects.fraud_anomaly_detection.codex_poc.control.run import run_skeleton

report = run_skeleton(
    Path("projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb"),
    findings_db=Path("/tmp/fraud_control_findings.duckdb"),
    reports_db=Path("/tmp/fraud_control_reports.duckdb"),
    config=ControlConfig(min_support=2, min_coverage=1, block_tier_precision=0.5),
)
pprint(report)
PY
```

The report is the holistic view of the loop, with discovery and plug validation
kept as separate layers:

- `discovery` — which discovery methods ran, their versions, finding counts,
  method metadata, method-level DPD45 rollups, the deduped union, and
  attribution grouped by `method_type`.
- `finding_store` — refresh key, data version, snapshot id, whether a new
  snapshot was written, stored row count, distinct users, and a persisted method
  metadata snapshot.
- `state_a_backtest` — the leak-free derivation state: scenario/graph discovery
  outcomes, candidate plug validation, and plug coverage for discovered users.
- `holdout_backtest` — the held-out A→B delta: the same discovery and plug
  validation measured only on new holdout activity after the cutoff.
- `plug` — candidate count, persisted candidate-fact snapshot id, burned-key
  count, the qualified key list, and the State A plug validation panel.

Plug validation buckets are intentionally named from the operator perspective:

- `covered_discovery` — discovery users the burned keys would catch.
- `uncovered_discovery` — discovery users with no deployable burned key.
- `outside_discovery` — users touched by the burned keys who were not in the
  discovery union. This replaces the ambiguous "innocent" framing; its own
  DPD45 rate tells whether the plug is overreaching or finding extra bad users.

When `reports_db` is provided, the full report is appended to
`run_reports(refresh_key, data_version, report_json, created_at)`. Use this as
the local daily/persistent view while the production warehouse table is still
out of scope for the sample skeleton.

For the scenario-by-scenario, graph-screen, selected-union, and plug-validation
view, use the repeatable selected-discovery report runner:

```bash
uv run --group fraud python -m \
  projects.fraud_anomaly_detection.codex_poc.control.selected_discovery_report \
  --store projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb \
  --out-dir projects/fraud_anomaly_detection/codex_poc/reports \
  --refresh-key selected_discovery_plug_report
```

That report:

- evaluates every scenario in the canonical scenario register;
- screens graph methods by total users, net-new users beyond the scenario union,
  and marginal net-new users after dedupe;
- reports snapshot-review graph screens for audit/review, but excludes them
  from plug derivation until they have `leakfree_asof` or `production_safe`
  metadata;
- derives plug candidates from the final State A discovery union;
- reports State A and holdout buckets for `covered_discovery`,
  `uncovered_discovery`, and `outside_discovery`.

Generated files under `reports/` are ignored by git; rerun the command whenever
the scenario register, graph screens, thresholds, or sample store changes.

The default method list lives in `control/discovery/catalog.py`. It currently
wires `ScenarioMethod("ring_account_reuse")` and `ResidualRingMethod` as the
representative scenario + graph pair. This catalog is the reviewed extension
point for methods that are live in the skeleton.

Each discovery method exposes method metadata:

- `method_type`: scenario, graph, model, or subgroup.
- `time_semantics`: snapshot_review, leakfree_asof, or production_safe.
- `promotion_tier`: evidence_only, review_queue, or plug_candidate.
- `enforcement_projection`: entity_key, scenario_rule, or none.

This keeps broad discovery safe: a method can be useful for review without
being eligible for plug derivation. `plug_candidate` is accepted only when
`time_semantics` is `leakfree_asof` or `production_safe`, and when
`enforcement_projection` is not `none`. Disable a method by removing it from
`default_methods()` or by setting `enabled=False` in its metadata.

The selected-discovery report uses reusable selection logic. It screens graph
methods by marginal net-new contribution after the scenario baseline and records
why each graph method was selected or excluded. Snapshot-review graph screens
remain excluded with reason `promotion_tier`.

### Adding a scenario discovery method

1. Add or update the scenario definition in
   `../scenarios/register.yaml`. That file remains canonical; do not fork
   scenario logic into `control/`.
2. Validate the scenario register with the existing scenario tests/validation.
3. Add `ScenarioMethod("<scenario_name>")` to `default_methods()` in
   `control/discovery/catalog.py`.
4. Add or extend tests under `codex_poc/tests/control/` so the method emits a
   `FindingSet` and appears in the catalog/report. The scenario will then be
   measured in `discovery.methods`, included in the deduped `discovery.union`,
   and split into State A / holdout outcome panels by `run_skeleton`.

### Adding a graph / Neo4j discovery pattern

1. Implement an adapter under `control/discovery/` with:
   - `name` such as `graph:<method_name>`.
   - `run(store) -> FindingSet`.
   - evidence fields needed to explain why each user surfaced.
2. The adapter may call the live project graph package (`graph.load`,
   `graph.discover`, `analysis`, or later Neo4j/GDS wrappers). It must not
   import from `codex_poc.archived`.
3. Add the adapter to `default_methods()` in `control/discovery/catalog.py`.
4. Add tests under `codex_poc/tests/control/` for the adapter, the catalog, and
   the end-to-end report. If the method is still `snapshot_review`, keep it in
   `review_queue`; only leak-free/as-of or production-safe graph adapters may be
   promoted to `plug_candidate`.

## Build posture

- **Sample data only.** Build and test against
  `../data/graph/fraud_graph.duckdb` (the local sample). Full v3 / warehouse
  work is out of scope here — it needs VPN; see the design's build-staging
  section.
- **No `archived/` dependency at the end.** Borrow from `archived/` during
  development if useful, but the finished `control/` copies what it needs in —
  the plan's final guard test asserts nothing in `control/` imports
  `archived/`. Repo packages (`scenarios`/`graph`/`analysis`) may be imported
  freely.
- **TDD on the sample**, per the plan.
