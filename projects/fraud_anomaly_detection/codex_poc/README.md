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
- **`control/`** — the system code (built by the plan; not yet created).
- **`archived/`** — the prior Neo4j-mirror POC, **reference only**. No
  obligation to use or maintain it; copy snippets out if useful. See
  `archived/README.md`.

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
