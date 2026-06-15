# neo4j_codex/ — Neo4j mirror POC

This folder is intentionally separate from `projects/fraud_anomaly_detection/poc/`.
The other POC is the notebook/export visualization track. This one tested a
different question:

> If we rebuild a disposable Neo4j mirror from the DuckDB graph store, does the
> graph-native workflow make fraud discovery and scenario explanation clearer
> enough to justify the next full-data experiment?

No shared project files are edited by this POC. Generated CSVs and reports live
under `out/`, which is gitignored.

Current read: **yes, the sample POC earned continuation.** Neo4j + Cypher + GDS
helped explain existing scenario rings and found high-concentration nearby
misses. The remaining work is tuning the graph schema, standardizing the daily
"plug the hole" workflow, and deciding whether DuckDB remains a bridge or Neo4j
is rebuilt directly from warehouse/parquet.

## User Story

As a fraud analyst, I want to:

1. Start from a named fraud scenario.
2. See the scenario's matched users and fraud/outcome flags.
3. Pick one user and inspect the local ring around them.
4. See connected entity types, fraud composition, and obvious shared resources.
5. Try graph-native component/PageRank-style discovery without loading the full
   graph into a browser.

The usability question is not "can Neo4j store the graph?" It can. The question
is whether the scenario -> users -> ring workflow is clearer than the existing
DuckDB/Python artifacts.

If Neo4j Browser feels like unlabeled dot clusters, start with
[`HOW_TO_USE_NEO4J.md`](HOW_TO_USE_NEO4J.md). The short version:
`MATCHED_SCENARIO` means "this user matched this scenario"; typed `USED_*`
relationships are the bank/device/phone/address/etc. linkage that explains how
users are connected. Start from suspicious clusters, then inspect the ring.

For graph-native discovery experiments, start with
[`DISCOVERY_WORKFLOW.md`](DISCOVERY_WORKFLOW.md). It documents the executable
Python runner, validation gates, queue outputs, and the current sample readout.

## Build The Mirror Bundle

From the repo root:

```bash
uv run --group fraud python -m projects.fraud_anomaly_detection.neo4j_codex.archived.export_neo4j_mirror
```

For a smaller smoke bundle:

```bash
uv run --group fraud python -m projects.fraud_anomaly_detection.neo4j_codex.archived.export_neo4j_mirror --max-edges 10000
```

Outputs:

- `out/neo4j/users.csv`
- `out/neo4j/entities.csv`
- `out/neo4j/scenarios.csv`
- `out/neo4j/used_device_rels.csv`
- `out/neo4j/used_bank_account_rels.csv`
- `out/neo4j/used_persistent_account_rels.csv`
- `out/neo4j/used_phone_rels.csv`
- `out/neo4j/used_address_rels.csv`
- `out/neo4j/scenario_match_rels.csv`
- `out/neo4j/cypher/*.cypher`
- `out/neo4j/summary.md`
- `out/neo4j/neo4j_admin_import.sh`

## One-Command Local Setup

If Docker is running locally:

```bash
bash projects/fraud_anomaly_detection/neo4j_codex/archived/scripts/setup_neo4j.sh
```

This does four things:

1. Rebuilds the Neo4j CSV bundle from the current DuckDB graph store.
2. Deletes only this POC's prior Neo4j container/data under `neo4j_codex/out/`.
3. Runs `neo4j-admin database import full` into a disposable local database.
4. Starts Neo4j Browser at `http://localhost:7474`.

Default login:

```text
username: neo4j
password: fraudpocpass
```

Stop it with:

```bash
bash projects/fraud_anomaly_detection/neo4j_codex/archived/scripts/stop_neo4j.sh
```

The setup uses `neo4j:5.26-community` by default. Override it if needed:

```bash
NEO4J_IMAGE=neo4j:latest bash projects/fraud_anomaly_detection/neo4j_codex/archived/scripts/setup_neo4j.sh
```

## Import Into Neo4j

The generated `neo4j_admin_import.sh` is a starting point for a local Neo4j
install or container. It assumes a full rebuild into an empty/disposable
database, matching this project's rebuild-not-incremental posture.

```bash
cd projects/fraud_anomaly_detection/neo4j_codex/out/neo4j
bash neo4j_admin_import.sh fraud_mirror
```

After importing, open Neo4j Browser and run the queries in `out/neo4j/cypher/`.
Bloom is optional evaluation only; the current POC does not require it.

## Run Discovery Experiments

With the local mirror running:

```bash
uv run --with neo4j --group fraud python \
  -m projects.fraud_anomaly_detection.neo4j_codex.archived.neo4j_discovery_experiments
```

Outputs:

- `out/discovery/summary.md`
- `out/discovery/signal_catalog.md`
- `out/discovery/queue_summary.csv`
- `out/discovery/coverage_top100_by_method.csv`
- `out/discovery/scenario_mirror_validation.csv`
- `out/discovery/scenario_*_residual_candidates.csv`
- `out/discovery/scenario_entity_evidence.csv`

The runner validates that Neo4j scenario counts match DuckDB/Python scenario
assignment before reporting discovery queues.

## Evaluation Result

The sample POC answered the first-pass gates:

- **Usability:** Neo4j Browser is usable only with supplied Cypher. Left-sidebar
  clicking is confusing; bounded scenario/cluster/entity drilldowns work.
- **Discovery:** Cypher/GDS exposed suspicious components, neighbors, and graph
  rankings that were awkward in the current flow; they found residual users
  near scenario/fraud rings.
- **Performance:** Sample rebuild/export/import is mechanical. Full-data runtime
  is still an open benchmark, especially for GDS under the 4-concurrency
  Community limit.
- **Clarity:** Are fraud users, scenario users, DPD45 users, and entity types
  visually obvious? Yes in query outputs; visual styling still needs a custom
  frontend or optional Bloom evaluation.
- **Scope:** Neo4j should be treated as the graph analysis/review store, not as
  the raw warehouse source of truth.

Working licensing assumption: Neo4j Community + GDS Community is acceptable for
internal evaluation and likely acceptable for internal production use if legal
approves GPLv3 server usage; GDS Community enforces `concurrency <= 4` at
runtime. Enterprise/Bloom only need procurement follow-up if full-data runtime,
security, or shared analyst UI requirements force them.
