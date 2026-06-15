#!/usr/bin/env bash
set -euo pipefail

# Rebuild a disposable local Neo4j mirror from the DuckDB graph store and start
# Neo4j Browser. Everything persistent lives under neo4j_codex/out/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$UNIT_DIR/../../.." && pwd)"

NEO4J_IMAGE="${NEO4J_IMAGE:-neo4j:5.26-community}"
CONTAINER_NAME="${CONTAINER_NAME:-fraud-neo4j-codex-poc}"
DB_NAME="${DB_NAME:-neo4j}"
NEO4J_AUTH="${NEO4J_AUTH:-neo4j/fraudpocpass}"
HOST_UID_GID="$(id -u):$(id -g)"

OUT_DIR="$UNIT_DIR/out/neo4j"
DATA_DIR="$UNIT_DIR/out/neo4j-data"
LOGS_DIR="$UNIT_DIR/out/neo4j-logs"

echo "==> Rebuilding Neo4j mirror CSV bundle from DuckDB"
cd "$REPO_ROOT"
uv run --group fraud python -m projects.fraud_anomaly_detection.neo4j_codex.archived.export_neo4j_mirror \
  --out "$OUT_DIR"

echo "==> Removing prior POC container/data, if any"
docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
rm -rf "$DATA_DIR" "$LOGS_DIR"
mkdir -p "$DATA_DIR" "$LOGS_DIR"

echo "==> Importing CSVs into disposable Neo4j database ($DB_NAME)"
docker run --rm \
  --user="$HOST_UID_GID" \
  --volume="$DATA_DIR:/data" \
  --volume="$OUT_DIR:/import:ro" \
  "$NEO4J_IMAGE" \
  neo4j-admin database import full "$DB_NAME" \
    --overwrite-destination=true \
    --nodes=User=/import/users.csv \
    --nodes=Entity=/import/entities.csv \
    --nodes=ReviewCluster=/import/clusters.csv \
    --nodes=Scenario=/import/scenarios.csv \
    --relationships=USED_DEVICE=/import/used_device_rels.csv \
    --relationships=USED_BANK_ACCOUNT=/import/used_bank_account_rels.csv \
    --relationships=USED_PERSISTENT_ACCOUNT=/import/used_persistent_account_rels.csv \
    --relationships=USED_PHONE=/import/used_phone_rels.csv \
    --relationships=USED_ADDRESS=/import/used_address_rels.csv \
    --relationships=IN_REVIEW_CLUSTER=/import/cluster_member_rels.csv \
    --relationships=MATCHED_SCENARIO=/import/scenario_match_rels.csv

echo "==> Starting Neo4j Browser on http://localhost:7474"
docker run -d \
  --name "$CONTAINER_NAME" \
  --user="$HOST_UID_GID" \
  --publish=7474:7474 \
  --publish=7687:7687 \
  --volume="$DATA_DIR:/data" \
  --volume="$LOGS_DIR:/logs" \
  --env "NEO4J_AUTH=$NEO4J_AUTH" \
  --env 'NEO4J_PLUGINS=["graph-data-science"]' \
  --env 'NEO4J_dbms_security_procedures_unrestricted=gds.*' \
  --env 'NEO4J_dbms_security_procedures_allowlist=gds.*' \
  "$NEO4J_IMAGE" >/dev/null

cat <<EOF

Neo4j POC is starting.

URL:      http://localhost:7474
Login:    neo4j
Password: ${NEO4J_AUTH#*/}

Start with:
  $OUT_DIR/summary.md
  $OUT_DIR/cypher/00_top_suspicious_clusters.cypher

Check logs:
  docker logs -f $CONTAINER_NAME

Stop:
  bash $UNIT_DIR/archived/scripts/stop_neo4j.sh
EOF
