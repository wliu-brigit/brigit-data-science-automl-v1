# neo4j_codex — session handoff (2026-06-16)

Wrap-up of the session that built the **ad-hoc discovery eval** loop on top of the
Neo4j-native control system. Read [`DISCOVERY_GUIDE.md`](DISCOVERY_GUIDE.md) first for
how the loop is meant to be used; this doc is the "what's done / what's next" map.

## What this session delivered (committed `e0dd50b`, branch `fraud/neo4j_codex`)

- **Ad-hoc discovery evaluator** — `control/discovery/adhoc_eval.py`. Run one
  candidate Cypher that returns `user_id` rows and get it scored in the control-loop
  report's format (DPD45 user/advance rates, net-new beyond the discovery union,
  per-method overlap), **without editing the method catalog**.
- **Discovery cache** — `control_loop_report` now writes a sidecar
  `reports/<refresh_key>.cache.json` (per-method + `scenario_union` +
  `final_discovery` user-id sets). The evaluator reads it so net-new is a cheap set
  subtraction, not a full Neo4j re-run.
- **One outcome metric** — `control/discovery/metrics.py` `outcome` is the single
  scorer over a per-user `user_truth` frame; windowing is an optional
  `user_truth(start=, end=)` filter. Discovery (no window) and plug/coverage
  validation (windowed) differ only in the frame they build. `summarize_users` was
  deleted; `plug_report` + `discovery_report` now use the shared `outcome`. Verified
  behavior-identical on full data (scenario rows + State A/Holdout plug tables
  byte-for-byte unchanged).
- **The guide** — `docs/DISCOVERY_GUIDE.md` orients an agent on the loop.

Status: **76 tests pass**; full-data report runs clean; branch is **ahead of
`origin/fraud/neo4j_codex` (unpushed)**.

## How to run it (prerequisites for a fresh session)

- Neo4j mirror up with GDS (Docker container `fraud-neo4j-codex-poc`,
  `bolt://localhost:7687`, `NEO4J_PASSWORD=fraudpocpass`). Stand it up via
  `neo4j_mirror/scripts/setup_neo4j.sh` if down.
- Full store at `projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb`.
- Run **off-VPN** (GCS/Neo4j flap on VPN). Use **absolute paths** when launching in a
  background shell (relative paths resolved to the wrong CWD once this session).

1. Refresh the baseline + cache (run the control-loop report — see the README command).
2. Score a candidate:
   ```bash
   NEO4J_PASSWORD=fraudpocpass uv run --with neo4j --group fraud python -m \
     projects.fraud_anomaly_detection.neo4j_codex.control.discovery.adhoc_eval \
     --cypher-file my_idea.cql \
     --store projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb
   ```
   (`--cache` defaults to the standard `fraud_control_loop_report` cache. The CLI is
   awkward for long inline Cypher — prefer `--cypher-file`, or call
   `adhoc_eval.evaluate_candidate` from a small driver, as done this session.)

## Key finding to build on

Every fraud-**anchored** discovery method converges on the same ~3,466 users (the
final union). A fraud-anchored phone pocket scored 99% DPD45 but only **5 net-new** —
precise, but a re-find, not a new instrument. **Net-new lives in structural /
non-fraud-anchored candidates.**

## Next steps (priority order)

1. **End-to-end test — cold-start guide check.** Hand a fresh subagent *only*
   `DISCOVERY_GUIDE.md` + a candidate idea in words, and see if it reaches a correct
   panel unaided. This validates the guide-as-interface premise (never exercised).
2. **Explore more fraud cases (net-new probe).** Run structural candidates, starting
   with relaxed-window device/bank pockets: users on an entity shared by ≥3 distinct
   users *ever* (no 72h window, no fraud seed), then read net-new + net-new DPD45
   rate vs the ~50% bar. Iterate on other structural angles (phone/address pockets
   not anchored to fraud, multi-type corroboration without a seed).
3. **Later — extend the full-data setup / native scenarios** per
   [`FULL_DATA_SETUP_NOTES.md`](FULL_DATA_SETUP_NOTES.md): retire `ring_account_reuse`
   from the DuckDB path, give graph methods leak-free as-of semantics so they can
   promote, rework memory-heavy queries to scale.

## Deferred by design (don't re-litigate without intent)

- **Plug validation on a candidate** — the evaluator stops at discovery metrics; it
  does not run plug derivation for an ad-hoc query.
- **Promotion path** — "adding to the definition" (a proven query → catalog with
  leak-free/as-of semantics + tests) is a separate, later, reviewed step.
