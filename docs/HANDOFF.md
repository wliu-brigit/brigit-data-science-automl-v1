# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up. **Not a changelog** — keep only what's current and
relevant; detail lives in the project docs. Rewritten at each wrap, not appended
to.

**Last updated:** 2026-06-13 (designed the fraud control system, wrote the
build plan, reorganized `codex_poc/` into a standalone unit; build not started)

**Status:** Design phase complete and committed. The next session **builds the
walking skeleton** of the fraud control system in `codex_poc/control/`, on the
sample data, by executing a written TDD plan. Nothing is pushed.

## How to pick up

1. Read this file.
2. Read `projects/fraud_anomaly_detection/codex_poc/README.md` — the standalone
   unit, its layout, and the build posture.
3. Read `projects/fraud_anomaly_detection/codex_poc/docs/CONTROL_SYSTEM_DESIGN.md`
   (the design) and `projects/fraud_anomaly_detection/PRINCIPLES.md` (P1–P9, the
   durable principles the build answers to). `docs/SCHEMA_DESIGN.md` is the graph
   schema notes.
4. Execute `projects/fraud_anomaly_detection/codex_poc/docs/WALKING_SKELETON_PLAN.md`
   — 9 bite-sized TDD tasks. Use the `superpowers:subagent-driven-development`
   skill (recommended) or `superpowers:executing-plans`.

## The task: build the walking skeleton

An end-to-end slice of the **discovery → plug-the-hole → monitor** control loop,
all in `codex_poc/control/`, on the sample store
(`projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb`): a discovery
contract, a versioned finding store, one scenario adapter + one graph adapter
(the "representative few" — not exhaustive), the extract→validate→qualify plug
derivation, a two-state holdout, monitoring stats, and an orchestrator. Task 9
severs any `archived/` dependency with a guard test. The plan has the complete
code and tests per task; follow it task-by-task.

## What this session produced (all committed, not pushed)

- Committed the previously-uncommitted DuckDB graph-stack work (link-grain edges
  mechanical side + new discovery methods); added advance-outcome counts + rates
  to the Neo4j mirror export.
- Designed the control system from first principles: **PRINCIPLES.md P1–P9**,
  `codex_poc/docs/CONTROL_SYSTEM_DESIGN.md`, and the walking-skeleton plan.
- Reorganized `codex_poc/` into a standalone unit: `docs/` (design + plan),
  `archived/` (the prior Neo4j-mirror POC, reference only), fresh README.

## Non-negotiables for the build (from PRINCIPLES + the design)

- **Sample data only.** v3 / warehouse work is out of scope here (needs VPN).
- **Standalone unit.** Importing the repo packages (`scenarios`/`graph`/
  `analysis`) is fine; borrowing from `archived/` during dev is fine, but the
  finished `control/` must carry **no dependency on `archived/`** (Task 9 guard).
- **Discovery and enforcement are separate, validated differently** (P9): a plug
  is validated against DPD45 **and** discovery-coverage.
- **Leak-free at promotion** (P7): discovery uses full as-of state, but any
  deployable plug is validated via the two-state holdout.
- **Thresholds are parameters** (P6), in one config; **no MLflow** (P8) — state
  is operational (DuckDB/parquet here; warehouse/GCS later).
- **Do not push.**

## Parked / out of scope (not the skeleton)

- **v3-gated** (needs VPN/prod): threshold tuning, which discovery methods/keys
  actually work, real `SHARES_RESOURCES` size, the link-grain + IP-key SQL.
- **Post-skeleton** (plug into the built skeleton, no rebuild): plug
  lifecycle/expiry, finding-snapshot diff tooling, the warehouse-facing burned-key
  table, live daily monitoring. See `CONTROL_SYSTEM_DESIGN.md` §8/§9.
- **`data/queries/link_table.sql`** — written, not warehouse-validated; lower
  priority, VPN session (tracked in `TODO.md`).
- **Open decision:** whether `PRINCIPLES.md` moves into `codex_poc/docs/` for
  full self-containment (kept at project level for now).

## Git

Branch `feature/fraud-anomaly-detection`; working tree clean; **nothing pushed**
(fork-PR workflow — push to the SoulEvill fork remote when ready, per memory).
This session's commits, oldest→newest: `8f04dd1` (graph-stack work) → `2998262`
(mirror outcome counts) → `69f7f63` (principles + schema notes) → `6ac82fe`
(control-system design) → `f41134a` (walking-skeleton scope) → `39b7dbf`
(codex_poc reorg) → `c2d945f` (archived-reuse policy).
