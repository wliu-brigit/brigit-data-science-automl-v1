# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up. **Not a changelog** — keep only what's current and
relevant; detail lives in the project docs. Rewritten at each wrap, not appended
to.

**Last updated:** 2026-06-09 (persisted graph stack built + proven on the local
sample; old network scripts pruned → start here)

**Status: APPROVED on the sample (wendao, 2026-06-09)** — capability validated
end-to-end on `data/sample/graph_sample.parquet` (20k advances, fraud-enriched;
all rates are workflow evidence only). **Next milestone: the same stack on the
full v3 data** — follow the runbook below.

## How to pick up (per wendao)

Don't dive into code. (1) Read this plus the project docs below; (2) summarize
where things stand; (3) **recommend 2–3 options and let wendao pick.** The next
move is wendao's call, not a queue to drain.

Project docs (`projects/fraud_anomaly_detection/`): **`TODO.md`** (top section);
**`LEARNINGS.md`** (newest first — the two 2026-06-09 entries: graph store +
graph/entity-ring verdict); `analysis/README.md` (the script map);
`SCENARIOS.md`; `scenarios/register.yaml` (untouched this session — wendao:
"that part is solid").

Design + plan for the graph stack:
`docs/superpowers/specs/2026-06-09-fraud-entity-graph-store-design.md` and
`docs/superpowers/plans/2026-06-09-fraud-entity-graph-store.md` (executed).

## What this session built (all on the LOCAL SAMPLE — no prod access)

The 2026-06-08/09 throwaway in-memory graph is replaced by a **persisted,
tested graph stack** under `projects/fraud_anomaly_detection/graph/` with thin
runners in `analysis/` (see `analysis/README.md` for the map):

- **store** (`build.py`) — lossless, self-contained DuckDB file: uncapped
  edges, full timestamps, ALL 7 entity types (ip/email stored, default-off),
  full base-table snapshot inside. Rebuild-only refresh. Principle: **lossless
  store, opinionated views** — every judgment call (degree cap, layers, as-of,
  scenario flags, weights) is an analysis-time parameter.
- **views** (`load.py`) — layers / per-view degree cap / as-of / window;
  dynamic scenario overlay (register evaluated at load time, never persisted).
- **questions** (`queries.py`) — near_flagged, components, ring,
  project_users, hub_report.
- **leak-free** (`asof.py`) — event-ordered strictly-prior replay with
  maturity-activated seeds + self-exclusion (the pruned v1 sweep's core,
  rebuilt as a tested store consumer). Use this for ANY precision claim.
- **queues** (`discover.py`) — five snapshot review queues: residual ring
  members, bad neighbours, emerging farms, multi-witness pairs, fresh rings.
  Snapshot semantics on purpose (wendao: current use case is discovery).

87 project tests green. Deps in the **`fraud` dependency group**
(`uv sync --group fraud`, or `uv run --group fraud ...` per command — every
graph command needs it).

**Sample caveats (memorize):** `data/sample/graph_sample.parquet` (20k rows,
gitignored) is fraud-ENRICHED (~3.5% fraud users vs ~0.13% natural) and
graph-thinned, and appears sampled around known rings (scenario coverage of
fraud ≈ total: only ONE unflagged confirmed-fraud user exists in it). All
sample rates are workflow evidence, NOT findings. Notable sample readouts:
bad-neighbour queue 48.6% DPD45 vs 10.7% base; multi-witness PAIRS ≈ base
(couples share everything — ring SIZE ≥3 matters, the v1 lesson re-confirmed);
hub users are ~100% scenario-covered (hubs = confirmation, not discovery).

**Pruned this session** (findings in LEARNINGS, recoverable from git): the v1
graph trio (discovery_sweep / validate_winner / seed_coverage), five completed
v1-pinned screens (ceiling_probe, edge_precision_screen, institution_screen,
ip_screen, residual_next_layer), and the DuckPGQ probe (no osx_arm64 build for
DuckDB 1.5.3 — verdict SKIPPED; igraph is the traversal engine, plain SQL the
rest).

## Environment (this worktree, deliberate)

- **No `.env` here** — this was a sample-only session; the harness session
  cannot bind. `graph_store_build.py` preflights and reports exactly what's
  missing. Local test MLflow (127.0.0.1:54321) is up but auth'd; the v3
  dataset `v2_2ac98b52` lives in the PROD registry — not reachable from here.
- Loading a registered dataset needs MLflow + GCS only; Snowflake/VPN is only
  for materializing new datasets.

## Runbook — first run on the full v3 data (step → validate → proceed)

Session-scope notes (wendao, 2026-06-09): **snapshot analysis only — no as-of
work this pass** (asof.leakfree_features stays the later gate for rule
candidates, step 4; don't run it as part of the v3 validation). And **record
wall-clock + peak memory at every step** — this is the first run at real
scale (~120× the sample; ~14M edges expected). Where the time goes: store
build = GCS download + one SQL pass; battery = scenario assign on 2.4M rows +
two graph loads; queues = two more loads + the projection SQL. If any step
blows past ~15 min or memory past a few GB, stop and apply the spec's
fallbacks (collapse parallel edges at load / scipy for global passes /
rustworkx) before grinding on.

**Step 0 — prereqs.** Drop a `.env` at the repo root (prod MLflow + GCS
values; Snowflake fields NOT needed — registered-dataset loads read MLflow +
GCS only). `uv sync --group fraud`. Verify: running
`uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_build`
gets PAST preflight (it reports exactly what's missing otherwise).

**Step 1 — build the v3 store** (same command; defaults to dataset
`v2_2ac98b52` → `data/graph/fraud_graph_v3.duckdb`). Expect: minutes (GCS
download + one SQL pass); store file ~1.5–3GB. **Validate against the v3
build facts (2026-06-09 materialization)** before trusting anything:
- `meta`: `n_advances` = **2,412,045**; `advances` has **115** columns;
  span 2025-01-01 → 2026-06-08.
- All 7 entity types present in `edges`; email stays near-noise
  (`SELECT max(n_users) FROM entities WHERE entity_type='email'` → **6**).
- Sharing-edge sanity (entities with ≥2 users, vs the warehouse build):
  bank ≈ **4,897**, device ≈ **5,466**, persistent ≈ **478**, phone ≈
  **2,291**, address ≈ **2,649**, email ≈ **593**
  (`SELECT entity_type, count(*) FROM entities WHERE n_users >= 2 GROUP BY 1`).
- File reopens read-only from a fresh process.

**Step 2 — battery** (`graph_question_battery.py --store
data/graph/fraud_graph_v3.duckdb`). Expect: igraph load ~1 min / 2–3GB RAM
(if heavier: collapse parallel edges at load / scipy global pass / rustworkx
— spec's named fallbacks). Validate:
- Pooled base rates land near natural prevalence (~0.13% fraud at advance
  grain; pool defaults = mature-only + Aug-2025+, the settled decision).
- The four locked scenarios fire (the runner preflights `TRIGGER_COLUMNS`).
- Hubs: big hubs should show high scen_coverage (v1 consistency); LOW-coverage
  velocity hubs are the new-farm leads.
- **The decision readout (TODO #2):** multi-type census + residual cut with
  concentration and early/late stability — compare against the v1 ceilings
  (LEARNINGS 2026-06-09: structural rule ~55–65% @ ~0.006% coverage;
  bad-neighbour ~40%/7× @ ~0.013%). The question: do the new phone/address
  node types + deeper history move coverage and/or precision?

**Step 3 — queues** (`graph_discovery_queues.py --store ...`). Validate queue
SIZES are reviewable before sharing (tune `min_users`/`min_types`/`--days`;
pair queue only in conjunction — pairs alone ≈ households). Spot-check top
members by hand (`ring(g, user_id)` for the ego view).

**Step 4 — any rule candidate from the queues** gets measured leak-free
(`graph.asof.leakfree_features` — strictly-prior, maturity-activated seeds)
BEFORE any precision number is quoted; then the register's normal path
(draft → monthly backtest → sign-off), wendao's call.

**Step 5 — if the graph still caps at review-tier/~0.01%** → Tier-3 pivot
(TODO), and bad-neighbour features into the supervised lens (TODO #3) remain
the fallback value path.

## What's open — wendao to pick

1. **★ Run the v3 runbook above (TODO #2, the value test).** Needs only the
   `.env`. Decides whether deeper history + phone/address node types move the
   graph off review-tier/~0.01% coverage.
2. **Queue tuning for the review team** — thresholds (`min_users`, `min_types`,
   `--days`, caps) against what review can absorb; conjunctions for the pair
   queue (pair + freshness/velocity — pairs alone ≈ households).
3. **Bad-neighbour-count as a model feature (TODO #3)** —
   `asof.leakfree_features` already produces the per-advance feature columns
   (nb_comp / nb_d1 / comp_users / comp_types); join to the training pool and
   test in the supervised lens.
4. Tier-3 data pivot for the neobank fast-churn cohort — unchanged, parked in
   TODO.

## Gotchas (still live)

- **`uv run --group fraud ...`** for anything importing duckdb/igraph; bare
  `pytest` skips graph tests cleanly (importorskip).
- Use `dry_run=False` and dataset id **`v2_2ac98b52`** when prod access exists
  (the `v2_` prefix is lineage; schema_version 1 — known naming quirk).
- `config.exclude_cols` edits only take effect on the next materialize.
- `experiment run` / `data materialize` preflight needs Snowflake/VPN (~30s).
- `--instruction` is shell-evaluated — keep it plain prose.
- A killed `experiment run` holds the session lock (~6h); `trial lock release`.
- `base_table.sql` comments must avoid `;` and `'` until the `_scrub_sql`
  harness bug is fixed (library to-do).
- Scenario register: requires its trigger columns; battery/queues runners
  preflight `TRIGGER_COLUMNS` against the base before evaluating.

## State / loose ends

- Branch `feature/fraud-anomaly-detection`, all committed, suite green (87).
  Commit trail this session: spec/plan docs → store/views/queries (TDD +
  two-stage review each) → demo → battery → asof (leak-free) → discover
  (queues) → prunes + docs. `git log --oneline` tells the story.
- Register NOT edited (wendao instruction). No new scenario candidates locked;
  queue-derived rules would go through `asof.leakfree_features` measurement
  first, then the register's normal draft → backtest → sign-off path.
- The scaffold template + root README now document the per-project dependency
  group convention (contract test pins it).
