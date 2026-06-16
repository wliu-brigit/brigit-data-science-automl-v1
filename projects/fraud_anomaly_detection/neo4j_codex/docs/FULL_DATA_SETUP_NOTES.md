# neo4j_codex — improvement backlog

Open friction and things worth improving that aren't done yet. Not an audit
trail — when something's done, delete it from here.

## Setup / ops

- **Docker preflight is minimal.** Setup assumes Docker is installed and running.
  Could detect a missing/stopped Docker and print actionable guidance (or offer to
  launch/install it).
- **Document Docker VM memory.** The full mirror + GDS wants headroom; the Docker
  Desktop VM defaults low (~7.7 GB seen on a 24 GB host). Recommend bumping it to
  >=12 GB for full-data runs, and note where in Docker Desktop settings.

## Graph discovery — robustness

- **Memory-heavy queries lean on a fat heap.** `high_risk_entity_members`
  (`count(DISTINCT member)` across all entities) and `residual_ring_members`
  (`collect(gds.util.asNode(...))` over the graph) only fit because setup gives
  Neo4j a 5 GB heap / 4 GB txn pool. Rework them to aggregate/stream so they scale
  past v4 and run on smaller machines.

## Graph discovery — make it better

- **Source + dependency labels (done).** Every discovery row declares `source`
  (DuckDB / Neo4j Cypher / Neo4j GDS, GDS flagged non-deterministic) and `depends_on`
  (`column:is_fraud`, `scenario:<name>`, or `structural`), declared in the catalogs
  (`graph_screen_catalog.py`, `native_scenarios.py`).
- **Graph methods re-seeded from native scenarios (done).** The scenario-seeded graph
  methods no longer read the DuckDB-baked `scenario_*` / `MATCHED_SCENARIO` flags:
  `scenario_neighborhood:*` and `high_risk`/`multi_witness` take the active scenario
  source's user sets as Cypher params (`$scenario_users` / `$seed_users`); `is_fraud`
  seeds come from the `FraudUser` label; `residual_ring_members` now seeds from
  `FraudUser` too (its dep is `column:is_fraud`). Needs a `user_id` index (created in
  setup and on the running mirror). The only remaining external dependency is
  `column:is_fraud`. Graph methods are still `review_only`, so this changed no headline
  numbers — it's a coherence fix.
- **Still open — retire the DuckDB scenario path.** With native default + re-seeded
  graph methods, the DuckDB scenario computation is only reached via
  `--duckdb-scenarios` and the parked `ring_account_reuse`. Once `ring_account_reuse`
  is handled (advance-node model) or accepted as a permanent DuckDB holdout, the
  DuckDB scenario path can be removed. Also: add the `user_id` index to
  `setup_neo4j.sh` so fresh mirrors have it.

- **Methods are review-only.** All current graph methods are `snapshot_review`, so
  none can be promoted into plug derivation. Give them leak-free as-of semantics so
  graph discovery can actually contribute plugs, not just surface leads.
- **Unify discovery on Neo4j (native is now the default).** Three burst scenarios
  (`ring_device_burst`, `ring_identity_burst`, `ring_shared_persistent_account`) are
  derived natively in Neo4j (72h burst over entity linkage, from `first_ts` — no
  feature column) and are the **default** scenario source in the report
  (`control/graph/native_scenarios.py`). `--duckdb-scenarios` forces the old DuckDB
  aggregate path. Native union runs at slightly higher precision than the DuckDB
  union (95.0% vs 93.3% DPD45) for ~96% of the volume; the residual is the
  timestamp-grain caveat.
  - **Parked: `ring_account_reuse`** still falls through to DuckDB inside the native
    source. Its trigger needs advance-grain fields (loan_amount, is_joint,
    prior-advance counts) the user-level mirror does not carry; going native there
    would mean modeling the Advance as a node — deferred to keep the mirror at user
    level.
  - **To fully retire the DuckDB scenario path:** handle `ring_account_reuse` (or
    accept it as a DuckDB-only holdout). Standalone confusion-matrix comparison lives
    in `scratch/native_scenarios_compare.py`.
