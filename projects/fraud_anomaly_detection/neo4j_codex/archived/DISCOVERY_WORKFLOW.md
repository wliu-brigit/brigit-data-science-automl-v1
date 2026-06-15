# Neo4j Discovery Workflow

This POC uses Neo4j as a disposable investigation mirror for the sample. The
purpose is to test whether graph-native discovery gives us a repeatable process
before the full-data run.

Current conclusion: Neo4j + Cypher + GDS is useful enough to continue. It found
high-concentration residual users near known scenario rings and made the
evidence easier to explain through typed user-entity links. The production
shape should be a daily "find and plug the hole" process, not a whole-graph
visualization.

## Run The Experiment

Start the local Neo4j mirror first:

```bash
bash projects/fraud_anomaly_detection/neo4j_codex/archived/scripts/setup_neo4j.sh
```

Run the executable Python report:

```bash
uv run --with neo4j --group fraud python \
  -m projects.fraud_anomaly_detection.neo4j_codex.archived.neo4j_discovery_experiments
```

Optional slow similarity queue:

```bash
uv run --with neo4j --group fraud python \
  -m projects.fraud_anomaly_detection.neo4j_codex.archived.neo4j_discovery_experiments \
  --include-slow
```

Outputs land in:

```text
projects/fraud_anomaly_detection/neo4j_codex/out/discovery/
```

Start with `summary.md`, then open the CSV for the queue or scenario you want
to inspect.

## Validation Gates

The report starts with validation because usability depends on trust.

1. Scenario mirror validation:
   - Recomputes scenario assignment from DuckDB using the Python scenario engine.
   - Compares counts to Neo4j `MATCHED_SCENARIO` relationships.
   - Compares counts to Neo4j user properties such as `scenario_ring_device_burst`.
   - The current sample passes all scenario count checks.

2. Semantic caveat:
   - Scenario flags are not recomputed from raw graph traversal.
   - The 72h scenario definitions come from existing feature columns.
   - That is intentional for this mirror: Neo4j should explain and discover
     around the current register, not silently redefine it.

3. Queue usability:
   - Every queue emits an executable CSV.
   - Queue rows include candidate user, score, DPD45/fraud/scenario flags, and
     evidence columns where possible.
   - The report marks whether a queue is directly usable for review.

## Current Sample Baseline

Current sample inventory:

- Users: 19,301
- Scenario users: 1,024
- Fraud-proxy users: 673
- DPD45 users: 2,056
- Non-scenario users: 18,277
- Non-scenario, non-fraud users: 18,276
- Non-scenario DPD45 users: 1,130

The useful benchmark is the non-scenario, non-fraud residual pool:

```text
18,276 users, 1,129 DPD45 users, 6.18% DPD45 baseline
```

A discovery queue needs to beat that residual baseline.

## Plug-The-Hole Workflow

The best operating model from the sample is:

```text
warehouse/parquet or DuckDB graph build
  -> rebuild Neo4j graph
  -> run scenario validation
  -> run Cypher + GDS discovery queues
  -> emit review queues and risklists
  -> send risky users/entities/clusters to operations or production controls
```

Cypher should explain evidence:

- which device, bank account, phone, address, or persistent account connects the
  candidate to known risky users;
- how many connected users are scenario, fraud-proxy, or DPD45;
- whether multiple independent entity types support the same candidate.

GDS should discover and rank:

- connected islands;
- dense communities inside those islands;
- PageRank/PPR-style proximity to current scenario/fraud seeds;
- high-risk entity/community membership.

Do not ship a rule named "LabelProp community." Use Label Propagation to find a
group, then express the operational reason in typed evidence such as:

```text
candidate shares device + phone + bank with a community where most prior users
are scenario users and DPD45.
```

## Signal Families

The runner currently tests these families:

1. Local neighbor propagation:
   - `shared_risky_neighbors`
   - `multi_witness_neighbors`
   - `rarity_weighted_neighbors`

2. Entity-centric risk:
   - `high_risk_entity_members`

3. Component/community risk:
   - `gds_wcc_high_risk_components`
   - `gds_labelprop_high_risk_communities`

4. Personalized ranking:
   - `gds_ppr_scenario_fraud_seed`
   - `gds_ppr_dpd45_seed_upper_bound`

5. Negative/control signal:
   - `gds_global_pagerank`

6. Optional slow similarity:
   - `gds_node_similarity_to_risky`

The broader checklist lives in `out/discovery/signal_catalog.md`.

## First Sample Readout

Useful, interpretable residual queues:

- `high_risk_entity_members`: 48 users, 66.7% DPD45.
- `gds_wcc_high_risk_components`: 48 users, 68.8% DPD45.
- `gds_labelprop_high_risk_communities`: 15 users, 86.7% DPD45.
- `multi_witness_neighbors`: top 10 users, 100% DPD45; all 61 users, 24.6% DPD45.
- `shared_risky_neighbors`: all 174 users, 19.0% DPD45.

Broader but less directly scenario-ready queue:

- `gds_ppr_scenario_fraud_seed`: top 10 users, 90% DPD45; top 100 users, 23%
  DPD45; all 1,000 emitted users, 8.1% DPD45.

Controls:

- `gds_ppr_dpd45_seed_upper_bound`: 1,000 emitted users, 100% DPD45. This is
  expected because it seeds from DPD45; use it for case review, not validation.
- `gds_global_pagerank`: all 1,000 emitted users, 3.4% DPD45. This underperforms
  the residual baseline and should not be promoted.

Scenario-neighborhood add-on, using interpretable graph proximity around the
current register:

```text
35 additional users on top of 1,024 scenario users
17 DPD45
48.6% DPD45 rate
```

This is modest volume but strong concentration versus the residual baseline.
The main value may be risky entity/ring risklists, not only extra user IDs.

For `ring_device_burst`, current stable community readout:

```text
Base scenario: 988 users, 899 DPD45, 91.0% DPD45

WCC islands containing those users:
  11 islands
  1,053 total users
  34 non-scenario users
  18 non-scenario DPD45
  52.9% non-scenario DPD45

High-risk LabelProp communities containing those users:
  53 communities with deterministic concurrency=1
  691 total users
  16 non-scenario users
  14 non-scenario DPD45
  87.5% non-scenario DPD45
```

Label Propagation can vary slightly when streamed with parallel concurrency.
For day-over-day reporting, persist community assignments from a specific daily
run instead of recomputing them ad hoc in every notebook/query.

## Scenario-First Review

The runner also emits one residual queue per registered scenario:

- `scenario_ring_account_reuse_residual_candidates.csv`
- `scenario_ring_identity_burst_residual_candidates.csv`
- `scenario_ring_shared_persistent_account_residual_candidates.csv`
- `scenario_ring_device_burst_residual_candidates.csv`

Current sample results:

- `ring_account_reuse` neighborhood: 13 residual candidates, 84.6% DPD45.
- `ring_identity_burst` neighborhood: 21 residual candidates, 71.4% DPD45.
- `ring_device_burst` neighborhood: 33 residual candidates, 51.5% DPD45.
- `ring_shared_persistent_account` neighborhood: 7 residual candidates, 42.9% DPD45.

This is the most usable analyst loop:

```text
scenario -> matched users -> shared typed entities -> residual neighbors -> case review
```

## What Pattern Is Showing Up

On the sample, the strongest new candidates look like near-miss members of
existing ring patterns rather than a brand-new fraud family.

The common shape:

- A residual user is not scenario-flagged.
- The user shares one or more scarce typed resources with scenario/fraud/DPD45
  users.
- Stronger cases share multiple independent entity types, such as device plus
  bank, device plus phone, or device plus bank plus address.
- Top entities often have very high DPD45 concentration among attached users.

This suggests a candidate review category:

```text
Graph-near ring member: not caught by current scenario predicates, but attached
to high-risk ring entities or communities already dominated by DPD45/scenario users.
```

That is not yet a shippable scenario. To become one, it needs an as-of/leakage
safe rewrite into features such as:

- count of prior scenario users sharing the same device/bank/phone/address;
- count of distinct shared entity types with prior risky users;
- entity risk concentration using only prior-known users;
- freshness/span constraints to separate fraud farms from old shared infrastructure.

## Community Terms

- **WCC component:** the true isolated island. If two users can be reached
  through any chain of included user-entity links, they are in the same WCC.
- **LabelProp community:** a smaller dense group inside an island, produced by
  GDS community detection.
- **High-risk LabelProp community:** this POC's filter over LabelProp groups:
  at least 3 users, at least 2 DPD45 users, and at least 50% DPD45 rate.
- **ReviewCluster:** this POC's pre-ranked case-review artifact. It is designed
  for analyst workflow and includes DPD45/fraud/scenario composition.

## Visual Use

Use Neo4j Browser for bounded drilldown, not whole-graph browsing.

Recommended visual path:

1. Open a scenario residual candidate CSV.
2. Pick a candidate user.
3. In Neo4j Browser, run:

```cypher
:param user_id => '<candidate user id>';
MATCH path = (u:User {user_id: $user_id})
  -[:USED_DEVICE|USED_BANK_ACCOUNT|USED_PERSISTENT_ACCOUNT|USED_PHONE|USED_ADDRESS*1..4]-(n)
RETURN path
LIMIT 500;
```

Then read edge labels:

- `USED_DEVICE`
- `USED_BANK_ACCOUNT`
- `USED_PHONE`
- `USED_ADDRESS`
- `USED_PERSISTENT_ACCOUNT`

Those labels explain how the user is connected. `MATCHED_SCENARIO` only says a
scenario matched; it does not explain the shared resource.

## Full-Data Standardization

For the full dataset, keep this process:

1. Rebuild DuckDB graph store.
2. Rebuild Neo4j mirror.
3. Run `neo4j_discovery_experiments.py`.
4. Require scenario mirror validation to pass.
5. Compare methods by:
   - top-N DPD45 rate;
   - cumulative net-new users;
   - cumulative net-new DPD45 users;
   - runtime;
   - evidence interpretability.
6. Persist daily component/community assignments and risky entity/user outputs.
7. Promote only interpretable queue heads into candidate scenario definitions or
   daily risklist rules.
8. Re-test candidate scenarios outside Neo4j with as-of-safe features.

The Neo4j mirror is useful if it produces better review queues and clearer case
packets. It can become the graph analysis store if the full-data rebuild and
GDS runtime are acceptable, but raw warehouse/parquet should remain the source
of truth.
