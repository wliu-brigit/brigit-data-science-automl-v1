# Neo4j Mirror

This package rebuilds the local Neo4j sample mirror used by the
`neo4j_codex.control.control_loop_report` runner.

The boundary is intentionally narrow:

- `export.py` pours DuckDB sample graph facts into Neo4j import CSVs.
- `scripts/setup_neo4j.sh` rebuilds a disposable local Neo4j database with GDS.
- Discovery logic lives in `neo4j_codex.control.graph` and runs through
  Cypher/GDS at report time.

The exporter does not build Python graph clusters and does not import the old
Python graph backend.

