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
  method-level DPD45 rollups, the deduped union, and scenario-vs-graph
  attribution (`scenario_only_users`, `graph_only_users`,
  `scenario_and_graph_users`).
- `finding_store` — refresh key, data version, stored row count, and distinct
  users.
- `state_a_backtest` — the leak-free derivation state: scenario/graph discovery
  outcomes, candidate plug validation, and plug coverage for discovered users.
- `holdout_backtest` — the held-out A→B delta: the same discovery and plug
  validation measured only on new holdout activity after the cutoff.
- `plug` — candidate count, burned-key count, the qualified key list, and the
  State A plug validation panel.
- `holdout` — the legacy smoke-test summary (`prevented_bad`, `leaked_bad`,
  and historical compatibility fields). New review should use
  `holdout_backtest`.

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

The default method list lives in `control/discovery/catalog.py`. It currently
wires `ScenarioMethod("ring_account_reuse")` and `ResidualRingMethod` as the
representative scenario + graph pair. This catalog is the reviewed extension
point for methods that are live in the skeleton.

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
   the end-to-end report. The graph method will then contribute to `graph_only_users` or
   `scenario_and_graph_users`, depending on overlap with scenario findings.

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
