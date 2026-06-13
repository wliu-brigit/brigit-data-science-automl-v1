# codex_poc — Neo4j-graph fraud-control unit

A self-contained unit for the fraud **discovery → plug-the-hole → monitor**
control system. Treat it as standalone: development stays inside this folder,
and it is deliberately **not obligated** to import the rest of the repo's
evolving code. If something in `scenarios/`, `graph/`, or `analysis/` is
useful, copy what you need in here rather than coupling to it — the goal is
freedom to try the cleanest design, not reuse for its own sake.

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
- **Freedom over reuse.** Reuse of existing repo code is opt-in (copy in),
  never an obligation. Start fresh where that gives a cleaner system.
- **TDD on the sample**, per the plan.
