# Fraud entity graph — persisted store + analysis layers (design)

**Date:** 2026-06-09
**Project:** `projects/fraud_anomaly_detection`
**Status:** approved direction (wendao, 2026-06-09); spec for implementation planning

## Context

The 2026-06-08/09 graph effort proved that entity-ring structure carries real
fraud signal (multi-type density, bad-neighbour proximity — see
`LEARNINGS.md` 2026-06-09), but the graph itself was a **throwaway in-memory
UnionFind rebuilt inside each analysis script**: connected components only, no
persistence, no path/hop queries, every new question required new plumbing.

This effort replaces that with a **persisted, queryable entity graph** built
once per data snapshot and analyzed many times. Scope for this session: prove
the capability end-to-end on the local sample
(`data/sample/graph_sample.parquet`, 20k advances, fraud-enriched, all labels
mature). Precision/lift numbers measured on the sample do NOT transfer to
production data — this session is about capability, not metrics.

## Decisions (settled with wendao, 2026-06-09)

1. **Two decoupled layers.** A persisted **store** (plain node/edge tables in
   a single DuckDB file) and an **analysis engine** (igraph in Python) on top.
   Either side can be swapped later without redesigning the other.
2. **Store = DuckDB file.** Free, no server, zero maintenance, SQL-inspectable
   with a tool already known. The build step is a pure SQL transformation —
   portable to Snowflake if/when this graduates beyond local analysis.
3. **Engine = igraph** (`igraph`). Battle-tested (C core, ~20 years,
   active), pip/uv-installable, handles the full-scale graph (~14M edges at
   v3 size) in a few GB of RAM. NetworkX rejected for full-scale memory/speed;
   rustworkx is the named fallback if igraph ever disappoints (the store
   doesn't change).
4. **DuckPGQ is a scoped probe, not a foundation.** The community extension
   gets one experiment against the same tables (multi-hop `MATCH` in SQL);
   its verdict is recorded and nothing depends on it.
5. **Graph databases (Neo4j/Memgraph/Kùzu forks) deferred.** No real-time
   requirement, no BI consumers, offline batch analytics only — a server is
   all cost here. Revisit trigger: a visual ring-investigation UI for ops, or
   non-Python consumers.

## Core principle: lossless store, opinionated views

**The persisted layer is the lossless record of who-touched-what-when.
Every judgment call is a parameter of an analysis-time view.**

| Concern | Layer | How |
|---|---|---|
| Degree cap (junk-hub control) | analysis | `load(...)` parameter; off by default, recommended ~20 for ring traversal (v1 finding: caps 10–50 gave identical net-new results) |
| Edge-type selection (device/bank/…) | analysis | `layers=` parameter over the `entity_type` column |
| IP edges (NAT/household junk) | analysis | stored, excluded from default layers |
| Email edges (near-noise, max 6 users) | analysis | stored, excluded from default layers |
| As-of / time windows | analysis | every edge carries full timestamp; views filter |
| Scenario fraud flags | analysis | scenario engine runs at load time against the current register — never persisted, so the ever-growing register is always fresh |
| Edge weights | analysis | computed in the user↔user projection (counts by type / distinct types / recency); schemes are tweakable per analysis |
| Sentinel screening (`MAIL RETURN`, zip 00000, `''`/`none`/`0-0`…) | build | data cleaning of non-values, not judgment; screened→absent edge, counts logged in `meta` |

Rationale (from this session's discussion): high-degree nodes are **both** the
fraud pattern (136 identities programmatically cycling one device — exactly
what the locked scenarios catch) **and** a traversal hazard (infrastructure
hubs gluing strangers into giant components). Baking a cap into storage would
destroy the first to serve the second. Instead the store keeps everything,
traversal views cap, and a dedicated **hub report** treats high-degree
entities as a first-class finding (the bigger, the more interesting).

## Persisted layer (the contract)

One DuckDB file: `projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb`
(gitignored; add `*.duckdb` to `.gitignore`). Rebuildable at any time from the
source table — the file is derived state, never a source of truth (consistent
with the repo's two-stores rule; pushing the file to GCS/MLflow as an artifact
is a documented later step, not this session).

Tables:

```
edges    advance_id, user_id, entity_type, entity_value, ts
         -- one row per advance × non-null entity key; UNCAPPED, lossless.
         -- entity_type ∈ {device, bank, persistent, phone, address, email, ip}
         -- ts = feature_as_of_ts at full precision (NEVER day-bucketed:
         --      fraud rings burst intra-day — the prior effort's worst bug)

advances advance_id, user_id, ts + ALL base-table columns
         -- full base-table snapshot at build time. Makes the file
         -- SELF-CONTAINED (wendao requirement): node metadata, labels, and
         -- the scenario overlay all run off the store alone — no live base
         -- table needed at analysis time. Labels are a build-time snapshot;
         -- maturing labels refresh on rebuild (see Refresh model).

users    user_id, n_advances, first_seen_ts, last_seen_ts
entities entity_type, entity_value, n_users, n_advances, first_seen_ts, last_seen_ts
         -- convenience materializations, derivable from edges; entities is
         -- the hub report's base table

meta     key, value
         -- build stamp, source (sample path or dataset id), row counts,
         -- edge counts per entity_type, screened-sentinel counts per type,
         -- build options
```

Node identity convention: users are `user:<user_id>`; entities are
`<entity_type>:<entity_value>`. Constructed in the analysis layer — the store
keeps the columns raw.

What the store does NOT contain: anything *derived by judgment* — scenario
flags, weights, caps, layer choices. Metadata and labels ARE in the file (the
`advances` snapshot), but flags are always computed from it at load time
against the current register.

### Refresh model: rebuild, not incremental

New data is incorporated by **full rebuild only** — never incremental append.
Incremental graph maintenance silently drifts (aggregate counts, late rows,
half-failed appends), and append would leave matured labels stale on old rows;
a rebuild refreshes both for free and matches the repo's rebuild-don't-migrate
posture. The build is one SQL pass (minutes at full v3 scale; offline
workflow). Revisit trigger: build time reaching hours.

### Production path: Snowflake runs the same contract

The persistence contract is the **table schemas, not DuckDB**. The majority of
data lives in Snowflake; in production the same derivation SQL runs there and
`edges`/`users`/`entities`(/`advances`) become Snowflake tables. Analyses then
either query Snowflake directly or materialize those tables into the local
`.duckdb` for laptop-speed work — same file shape, same analysis code, only
the source differs. Snowflake has no graph traversal (recursive CTEs only), so
the igraph layer remains the analysis engine in both worlds.

## Build module

`projects/fraud_anomaly_detection/graph/build.py` — reads the base table
(this session: the sample parquet; later: the registered dataset / Snowflake
table), derives the tables above via DuckDB SQL (UNPIVOT of the entity-key
columns + sentinel screening + aggregates), writes the `.duckdb` file.
Idempotent: rebuild replaces the file.

Source selection is an explicit argument (default: the sample parquet) so
re-pointing at `v2_2ac98b52`/Snowflake later is a parameter change, not a
rewrite. The SQL stays warehouse-portable (no DuckDB-only syntax in the
core derivation beyond UNPIVOT mechanics).

## Analysis layer

`projects/fraud_anomaly_detection/graph/load.py` — parameterized views over
the store, returning an igraph bipartite multigraph (users ↔ entities, edge
attrs `entity_type`/`ts`, parallel edges kept):

```python
load_graph(
    store=GRAPH_DB_PATH,  # the .duckdb file
    base=None,            # node-attr/scenario source; default = the store's
                          # own `advances` snapshot (file is self-contained);
                          # pass a path/DataFrame to override
    layers=("device", "bank", "persistent", "phone", "address"),  # default; email/ip opt-in
    degree_cap=None,      # drop entity nodes with > cap distinct users
                          # (None = lossless; pass ~20 for ring traversal)
    as_of=None,           # edges with ts <= as_of only
    window=None,          # (start, end) edge-time slice
    node_attrs=("is_fraud", "label_gross_dpd45", "label_mature_d45",
                "is_neobank_high_risk_institution"),  # joined onto user nodes
    scenarios=True,       # run the scenario engine on `base` NOW (current
)                         # register), aggregate advance→user, attach
                          # scenario_<name> vertex attributes
```

`projects/fraud_anomaly_detection/graph/queries.py` — question-level helpers
on a loaded graph (each a small function; this is the query surface, since the
engine has no query language):

- `near_flagged(g, flag, max_hops)` — multi-hop proximity: users within k hops
  of any flagged user, with hop distance and the connecting path. Powers
  "who's close to a fraudster?" for any flag (scenario, `is_fraud`, custom).
- `components(g)` — connected components annotated with user count, distinct
  entity-type count (the multi-type density discriminator: `comp≥5 & types≥2`
  was the durable v1 finding), and flag/label mix.
- `ring(g, user_id, hops)` — ego subgraph around a user for deep-dives
  (communities, centrality, plotting live here, per component — small after
  capping).
- `project_users(store, ...)` — user↔user projection (SQL on the store — set
  math is the database's home turf); weight schemes: shared-entity count +
  distinct-type count (recency-decay deferred — add as a scheme when an
  analysis needs it). Always projects from a degree-capped view (an uncapped
  136-user device alone emits 9,180 pairs).
- `hub_report(store, top_n)` — SQL on `entities`/`edges`, NO cap: distinct
  users, activity span, users-per-day density, attached-user fraud rate.
  Separates fraud farms (many users, days, high fraud rate) from shared
  infrastructure (many users, years, base-rate) — the time axis is the
  disambiguator.

1-hop and aggregate questions stay plain SQL against the store; the helpers
exist for what SQL can't do (traversal, algorithms).

## DuckPGQ probe (scoped experiment)

Against the built store: `INSTALL duckpgq FROM community; LOAD duckpgq;`,
define a property graph over `users`/`entities`/`edges`, run the same
"within 3 hops of a flagged user" question as SQL/PGQ `MATCH`, compare answers
with `near_flagged` and note ergonomics/performance. Outcome = a short verdict
in LEARNINGS.md. Requires network for the extension install; if unavailable,
the probe is skipped and noted. Nothing downstream depends on it.

## Session demo (the acceptance test for the capability)

`projects/fraud_anomaly_detection/analysis/graph_store_demo.py` (runnable
module, prints a structured report), in order:

1. Build the store from the sample → table counts, screened-sentinel counts.
2. Reopen the file fresh; inspect with plain SQL (prove persist → reopen).
3. Load with defaults; overlay current register scenarios dynamically.
4. "Who is within 1/2/3 hops of a flagged user?" — union and per-layer.
5. Component census: multi-type vs single-type, sizes, flag concentration.
6. Hub report top-20: farms vs infrastructure via time density.
7. One ring deep-dive: extract, per-layer view, communities/centrality.
8. DuckPGQ probe (or skip note).

## Scale path (why this survives v3 full size)

| | sample (now) | v3 full (~2.4M advances) |
|---|---|---|
| edges table | ~100k rows | ~12–14M rows — trivial for DuckDB |
| advances snapshot | 20k × 110 | 2.4M × 115 ≈ 1–2GB compressed in-file — fine |
| igraph build | <1s | ~1 min load, ~2–3GB RAM — fine on a laptop |
| global components | igraph | igraph, or scipy.sparse.csgraph (already a dep) if faster needed |
| per-ring analysis | igraph | unchanged (rings are small after capping) |
| store build | local parquet → DuckDB | same SQL shape against Snowflake; store file artifact-able to GCS/MLflow |

If full-scale igraph memory runs heavier than estimated: collapse parallel
edges into weighted edges at load (large cut), push global passes to scipy,
or swap rustworkx — all behind the same store, no schema change.

## Error handling & edge cases

- Missing/sentinel entity values → no edge (screened at build; counts in `meta`).
- `persistent_account_id` is ~18.5% filled in the sample — absent keys simply
  emit no edge; per-type edge counts land in `meta` for sanity checks.
- Duplicate (advance, entity) pairs deduped at build; parallel edges across
  advances kept (multigraph semantics).
- `load_graph` raises on unknown layer names or empty edge selections.
- Scenario overlay: a register column missing from the provided base df
  surfaces the engine's own error (same behavior as `validation.py`) rather
  than silently dropping a scenario.
- Strict per-advance leak-freedom (score advance i using only edges with
  ts < its ts) is NOT what snapshot views give; if precision measurement
  returns later, the prior event-ordered pattern is reimplemented on top of
  the store. Out of scope this session (sample metrics don't transfer).

## Testing

`projects/fraud_anomaly_detection/tests/` (existing pattern, pytest, tiny
synthetic DataFrames — no live services):

- `test_graph_build.py` — losslessness invariants (edge rows == non-null,
  non-sentinel key cells per type; no cap applied), sentinel screening, dedup,
  aggregate tables consistent with edges, meta populated.
- `test_graph_load.py` — layer selection, degree-cap semantics (entity nodes
  dropped, users retained), as-of/window filtering, node-attr join, scenario
  overlay matches `engine.matches` aggregation.
- `test_graph_queries.py` — `near_flagged` hop counts on a hand-built toy
  ring; component type-counts; projection weights; hub-report columns.

## Dependencies

`uv add --group fraud duckdb igraph` (both free/OSS). DuckPGQ installed at runtime
from the community extension repo (probe only).

## Risks & mitigations (called out for wendao, 2026-06-09)

1. **Product-value expectation (the big one).** Infrastructure doesn't change
   the data: the v1 verdict was review-tier signal at ~0.01% coverage. The
   value test remains graph-on-v3-with-new-node-types (project TODO #2). This
   session proves capability only.
2. **Sample thinning.** 20k rows sampled from 2.4M fragments rings and shrinks
   hub degrees — structural demo results don't transfer to full scale either
   (extends the metrics non-goal to structure).
3. **igraph memory at full v3** — fallbacks above; low risk, mechanical.
4. **DuckPGQ flake** — scoped probe; failure costs a LEARNINGS note.
5. **Label-snapshot staleness** — the `advances` snapshot freezes label state
   at build time while labels mature ~45 days; mitigated by rebuild-refresh +
   build stamp in `meta`. Non-issue on the all-mature sample.

## Non-goals (this session)

- No real-time serving, no BI/ops query surface, no server database.
- No precision/lift conclusions from the sample (capability only).
- No scenario-register edits (the graph verdict on v1 was review-tier; any
  `ring_multitype_structural` registration remains wendao's separate call).
- No GCS/MLflow artifact push (documented as the later step), no Snowflake
  build run.
