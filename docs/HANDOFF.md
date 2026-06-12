# Handoff — continue here

Temporary hand-over note: where the last session left off and what's open, so a
new session can pick up. **Not a changelog** — keep only what's current and
relevant; detail lives in the project docs. Rewritten at each wrap, not appended
to.

**Last updated:** 2026-06-11 (Neo4j/Cypher/GDS POC paused after sample proof;
work tree still has unrelated uncommitted fraud-graph changes from parallel
sessions)

**Status:** Neo4j earned continuation on the sample. The `codex_poc/` track
proved that a disposable Neo4j mirror plus Cypher/GDS can explain known
scenario rings and find concentrated residual candidates around them. The next
session should preserve this as a reproducible POC, then decide how much of the
flow to standardize for the full-data run.

## How to pick up

1. Read this file.
2. Read `projects/fraud_anomaly_detection/codex_poc/README.md`.
3. Read `projects/fraud_anomaly_detection/codex_poc/DISCOVERY_WORKFLOW.md`.
4. If touching the non-Neo4j graph stack, also read
   `projects/fraud_anomaly_detection/TODO.md` and
   `projects/fraud_anomaly_detection/analysis/README.md`; those files may
   include changes from another active session.

Do not modify `projects/fraud_anomaly_detection/poc/`; that is the separate
notebook/export visualization track. The Neo4j work lives in:

```text
projects/fraud_anomaly_detection/codex_poc/
```

Generated files live under `projects/fraud_anomaly_detection/codex_poc/out/`
and are disposable. Rebuild them from scripts instead of committing them.

## Current Neo4j POC

Durable files:

- `codex_poc/README.md` — entry point, setup commands, evaluation result.
- `codex_poc/HOW_TO_USE_NEO4J.md` — how to use Neo4j Browser without getting
  lost in generic dot clusters.
- `codex_poc/DISCOVERY_WORKFLOW.md` — discovery process, validation gates,
  sample readout, plug-the-hole workflow.
- `codex_poc/export_neo4j_mirror.py` — DuckDB-to-Neo4j CSV/export bundle.
- `codex_poc/neo4j_discovery_experiments.py` — executable discovery report.
- `codex_poc/scripts/setup_neo4j.sh` — rebuilds CSVs, imports Neo4j, starts
  local Docker Neo4j with GDS.
- `codex_poc/scripts/stop_neo4j.sh` — stops the POC container.
- `codex_poc/tests/test_neo4j_mirror_export.py` — regression coverage for the
  export shape and usage docs.

Commands:

```bash
bash projects/fraud_anomaly_detection/codex_poc/scripts/setup_neo4j.sh
```

Then open:

```text
http://localhost:7474
username: neo4j
password: fraudpocpass
```

Run discovery:

```bash
uv run --with neo4j --group fraud python \
  -m projects.fraud_anomaly_detection.codex_poc.neo4j_discovery_experiments
```

Optional slow similarity queue:

```bash
uv run --with neo4j --group fraud python \
  -m projects.fraud_anomaly_detection.codex_poc.neo4j_discovery_experiments \
  --include-slow
```

## What was proven

The useful mental model:

```text
DuckDB today
  -> local rebuild/export bridge and validation layer

Neo4j
  -> graph-native mirror/review store
  -> User -[:USED_DEVICE/BANK/PHONE/ADDRESS/PERSISTENT_ACCOUNT]-> Entity

Cypher
  -> graph query language for explainable evidence and bounded drilldown

GDS
  -> graph algorithm plugin for components, communities, PageRank/PPR,
     and similarity-style discovery
```

The POC should not be used by clicking random relationship types in the Neo4j
Browser sidebar. Use the supplied Cypher files or the discovery runner. The
analyst workflow is:

```text
ranked scenario/cluster/candidate queue
  -> selected user/entity/community
  -> typed user-entity evidence
  -> review packet or risklist output
```

## Sample findings to preserve

Existing registered scenario coverage:

```text
scenario_any users: 1,024
scenario_any DPD45 users: 926
scenario_any DPD45 rate: 90.4%
```

Interpretable scenario-neighborhood add-on:

```text
35 additional users on top of 1,024
17 DPD45
48.6% DPD45 rate
```

This is about 3.4% more users over the scenario baseline and about 1.8% more
DPD45 users over the scenario DPD45 count. The value is not just user lift; it
also identifies risky shared entities that can help plug the next linked user.

Strong GDS community example:

```text
gds_labelprop_high_risk_communities:
  15 residual users
  13 DPD45
  86.7% DPD45 rate
```

For `ring_device_burst`, stable single-thread Label Propagation readout:

```text
Base scenario:
  988 users
  899 DPD45
  91.0% DPD45

WCC islands containing those users:
  11 islands
  1,053 total users
  34 non-scenario users
  18 non-scenario DPD45
  52.9% non-scenario DPD45

High-risk LabelProp communities containing those users:
  53 communities
  691 total users
  16 non-scenario users
  14 non-scenario DPD45
  87.5% non-scenario DPD45
```

Community definitions:

- **WCC component:** true isolated island under the included typed entity
  links.
- **LabelProp community:** smaller dense group inside an island.
- **High-risk LabelProp community:** POC filter over LabelProp groups:
  at least 3 users, at least 2 DPD45 users, and at least 50% DPD45 rate.
- **ReviewCluster:** POC case-review artifact with DPD45/fraud/scenario
  composition and an explicit review score.

Label Propagation can vary under parallel execution. For reproducible
day-over-day reporting, persist community assignments from one daily run rather
than recomputing them ad hoc in each query.

## Architecture decision so far

Short term:

```text
warehouse/sample data
  -> DuckDB graph store
  -> Neo4j mirror
  -> Cypher/GDS discovery
  -> CSV/report/risklist outputs
```

Longer term if Neo4j continues to earn it on full data:

```text
warehouse/parquet = raw source of truth
Neo4j = daily rebuilt graph analysis/review store
DuckDB = optional dev/audit/export tool, not required as a permanent layer
```

Open design choices:

- graph schema: which entity types become nodes, which raw events collapse into
  relationships, and which metadata belongs on users/entities/relationships;
- daily persistence: how to store component/community IDs, risklists, and
  evidence snapshots day over day;
- plug-the-hole output: whether the main product is risky users, risky
  entities, risky communities, or all three;
- as-of safety: any promoted scenario/rule must be rewritten outside Neo4j
  with leakage-safe features;
- visualization: lowest priority. Neo4j Browser is a developer console; Bloom
  is optional evaluation; a custom frontend is likely needed for serious case
  review.

## Licensing decision so far

Working assumption: Neo4j Community + GDS Community is acceptable for internal
evaluation and likely acceptable for internal commercial use if legal approves
GPLv3 server usage. GDS Community enforces `concurrency <= 4` at runtime; the
local POC confirmed requests above 4 fail with an unlicensed-GDS error.

Enterprise/Bloom procurement only matters if full-data runtime, security, or a
shared analyst UI requires it.

## Cleanup state

The POC's generated `out/` directory was removed from the workspace during this
handoff cleanup. Recreate it with `scripts/setup_neo4j.sh` or by running
`export_neo4j_mirror.py`.

The broader worktree still contains modified/new files outside `codex_poc/`
from other fraud-graph sessions. Do not revert them unless the owner asks.

## Next useful work

1. Run the full-data benchmark with the Neo4j mirror:
   rebuild graph, import Neo4j, run discovery, record import/GDS runtimes and
   memory.
2. Add/persist a daily community assignment artifact if LabelProp/WCC remain
   useful.
3. Standardize the plug-the-hole output:
   risky users, risky entities, risky communities, evidence columns, expiry, and
   review status.
4. Decide whether DuckDB remains in the production path or whether warehouse /
   parquet can feed Neo4j directly.
5. Only after the above, revisit visualization beyond Neo4j Browser.
