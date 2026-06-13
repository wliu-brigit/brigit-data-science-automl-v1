# archived/ — prior Neo4j-mirror POC (reference only)

The original `codex_poc` proof-of-concept: a DuckDB→Neo4j mirror export plus
Cypher/GDS discovery experiments. Kept as **reference**, not as a dependency.
The new control-system build (see [`../docs/`](../docs/)) is free to ignore,
copy from, or replace any of it — nothing here is obligated to stay green or be
maintained.

Contents:
- `MIRROR_POC_README.md` — the original POC entry point (build/run instructions).
- `export_neo4j_mirror.py` — DuckDB → Neo4j CSV-bundle export.
- `neo4j_discovery_experiments.py` — executable Cypher/GDS discovery report.
- `scripts/setup_neo4j.sh`, `scripts/stop_neo4j.sh` — local Docker Neo4j with GDS.
- `DISCOVERY_WORKFLOW.md`, `HOW_TO_USE_NEO4J.md` — the POC's workflow + Browser usage.
- `tests/test_neo4j_mirror_export.py` — the export regression test (imports +
  paths repointed to this archived location; still runs with
  `uv run --group fraud pytest projects/fraud_anomaly_detection/codex_poc/archived/tests/`).

What it proved (preserved for reference): a disposable Neo4j mirror + Cypher/GDS
explained known scenario rings and surfaced concentrated residual candidates on
the sample. Full findings are in `DISCOVERY_WORKFLOW.md`.
