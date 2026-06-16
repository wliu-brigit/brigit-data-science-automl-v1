# Discovery guide — talk out an idea, get it measured

POC (2026-06-16). Orientation for an agent (or a person) to run the
**talk → run → measure** loop over the Neo4j fraud graph: describe a discovery
idea, express it as one Cypher query, and get back the same analysis the
control-loop report uses — *without* editing the method catalog.

Promotion of a proven query into the catalog (`graph_screen_catalog.py` +
`methods.py`, with leak-free/as-of semantics) is a **separate, later, reviewed
step** — see the README section "Adding a graph / Neo4j discovery pattern". This
loop is exploration only.

## The candidate contract

A candidate idea is **one Cypher query that `RETURN`s a `user_id` column** (extra
columns are ignored). That is the same contract a cataloged graph method satisfies
(`control/graph/methods.py`). If an idea can't be written as a single
user-returning query yet, it's out of scope for this POC — talk it through first.

## The graph it runs over

User + Entity nodes; entities are the enforcement unit. Entity types: device, bank
account, persistent account, phone, address (+ email/IP in the schema). Edges are
`USED_DEVICE | USED_BANK_ACCOUNT | USED_PERSISTENT_ACCOUNT | USED_PHONE |
USED_ADDRESS`, each carrying timestamps (`first_ts`, `last_ts`). Fraud-labelled
users also carry a `:FraudUser` label. Full schema notes: `SCHEMA_DESIGN.md`.
As-of filtering convention: `r.first_ts <= localdatetime($as_of)`.

## The metric vocabulary

- **DPD45 user rate** — of the candidate's users that have a mature advance, the
  share with a DPD45-bad advance. The headline precision signal.
- **DPD45 advance rate** — same at advance grain (DPD45 advances / mature advances).
- **Net-new beyond final discovery** — candidate users *minus* the current deduped
  discovery union (scenarios + promoted graph methods). "Is this finding anything
  the loop doesn't already catch?" Its own DPD45 rates say whether the net-new is
  good.
- **Overlap per method** — how much the candidate intersects each cataloged method,
  and how many users it adds beyond that one method.

A promising candidate: meaningful **net-new** users at a **net-new DPD45 user rate**
in the same league as the promoted graph methods (the report's bar is ≥ 50%).

## The current method inventory

The live list is the source of truth — don't trust a frozen copy here:

- Scenario methods: `projects/fraud_anomaly_detection/scenarios` (register.yaml).
- Graph screens: `control/discovery/graph_screen_catalog.py` (metadata) +
  `control/graph/methods.py` (Cypher/GDS bodies).

## Baseline facts (read the cache)

The control-loop report writes a sidecar cache of discovery user-id sets when it
runs: `reports/<refresh_key>.cache.json` — per-method sets plus the `scenario_union`
and `final_discovery` unions. The evaluator reads it to compute net-new/overlap.
No cache yet → candidate DPD45 facts still print; net-new shows as unavailable.

Refresh the baseline (and the cache) by running the report (README has the full
command). _Last observed: populate on first full run._

## Run the evaluator

```bash
NEO4J_PASSWORD=fraudpocpass uv run --with neo4j --group fraud python -m \
  projects.fraud_anomaly_detection.neo4j_codex.control.discovery.adhoc_eval \
  --cypher "MATCH (u:User)-[:USED_DEVICE]->(d:Device)<-[:USED_DEVICE]-(v:User)
            WHERE u <> v RETURN DISTINCT u.user_id AS user_id" \
  --store projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb \
  --cache projects/fraud_anomaly_detection/neo4j_codex/reports/fraud_control_loop_report.cache.json
```

Use `--cypher-file path.cql` for longer queries, `--params '{"thr": 3}'` for
parameters, `--json` for machine-readable output. The `--cache` default already
points at the standard `fraud_control_loop_report` cache.

## How to read the result

1. **Candidate DPD45 user/advance rates** — is the idea precise at all?
2. **Net-new users + net-new DPD45 rate** — does it add good users the loop misses?
3. **Per-method overlap** — is it a genuinely new angle or a re-find of an existing
   method?

Before anything here could become a real plug, it needs a leak-free as-of or
production-safe implementation in the catalog — this loop only measures discovery
precision, not deployability.
