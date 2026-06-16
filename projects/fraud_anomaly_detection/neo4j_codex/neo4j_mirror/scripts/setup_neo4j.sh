#!/usr/bin/env bash
set -euo pipefail

# Stand up a disposable local Neo4j mirror from the DuckDB graph store, then pour
# the store in (export CSVs -> bulk import -> start). Pins the Neo4j image and the
# GDS plugin version, and downloads the GDS jar on the HOST (the container cannot
# reach the plugin host) so every machine runs the same, repeatable versions.
#
# The graph store itself is a separate, run-once build (registered dataset -> GCS
# -> DuckDB; needs .env + GCS ADC, no VPN):
#
#   uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_build \
#     --dataset-id v4_086fbc5a \
#     --out projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb
#
# Everything persistent for the mirror lives under neo4j_codex/out/.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
REPO_ROOT="$(cd "$UNIT_DIR/../../.." && pwd)"

# --- pinned versions (Neo4j 5.26.x <-> GDS 2.13.x are compatible) ---------------
NEO4J_IMAGE="${NEO4J_IMAGE:-neo4j:5.26-community}"
GDS_VERSION="${GDS_VERSION:-2.13.10}"
GDS_JAR_URL="${GDS_JAR_URL:-https://graphdatascience.ninja/neo4j-graph-data-science-${GDS_VERSION}.jar}"

# --- memory (sized for the full v4 mirror inside a ~8GB Docker VM) --------------
NEO4J_HEAP="${NEO4J_HEAP:-5G}"
NEO4J_PAGECACHE="${NEO4J_PAGECACHE:-1G}"
NEO4J_TXN_MAX="${NEO4J_TXN_MAX:-4G}"

CONTAINER_NAME="${CONTAINER_NAME:-fraud-neo4j-codex-poc}"
DB_NAME="${DB_NAME:-neo4j}"
NEO4J_AUTH="${NEO4J_AUTH:-neo4j/fraudpocpass}"
NEO4J_USER="${NEO4J_AUTH%/*}"
NEO4J_PASS="${NEO4J_AUTH#*/}"
STORE="${STORE:-projects/fraud_anomaly_detection/data/graph/fraud_graph.duckdb}"
HOST_UID_GID="$(id -u):$(id -g)"

OUT_DIR="$UNIT_DIR/out/neo4j"
DATA_DIR="$UNIT_DIR/out/neo4j-data"
LOGS_DIR="$UNIT_DIR/out/neo4j-logs"
PLUGINS_DIR="$UNIT_DIR/out/neo4j-plugins"

cd "$REPO_ROOT"

# --- preflight: Docker + store --------------------------------------------------
if ! command -v docker >/dev/null 2>&1 || ! docker info >/dev/null 2>&1; then
  echo "ERROR: Docker is not available/running. Start Docker Desktop and retry." >&2
  exit 1
fi
if [ ! -e "$STORE" ]; then
  cat >&2 <<EOF
ERROR: graph store not found: $STORE

Build it first (registered dataset -> GCS -> DuckDB; needs .env + GCS ADC, no VPN):

  uv run --group fraud python -m projects.fraud_anomaly_detection.analysis.graph_store_build \\
    --dataset-id v4_086fbc5a --out $STORE

Or point STORE at an existing store:
  STORE=/path/to/store.duckdb bash $0
EOF
  exit 1
fi
echo "==> Store: $STORE"

# --- GDS plugin: download once on the host, pin the version ---------------------
mkdir -p "$PLUGINS_DIR"
GDS_JAR="$PLUGINS_DIR/neo4j-graph-data-science-${GDS_VERSION}.jar"
# drop any other GDS jars so only the pinned version is mounted
find "$PLUGINS_DIR" -maxdepth 1 -name 'neo4j-graph-data-science-*.jar' \
  ! -name "$(basename "$GDS_JAR")" -delete 2>/dev/null || true
if [ ! -s "$GDS_JAR" ]; then
  echo "==> Downloading GDS $GDS_VERSION plugin (host)"
  curl -fsSL -m 180 "$GDS_JAR_URL" -o "$GDS_JAR"
fi
if ! unzip -l "$GDS_JAR" >/dev/null 2>&1; then
  echo "ERROR: GDS jar is not a valid archive (download failed?): $GDS_JAR" >&2
  rm -f "$GDS_JAR"
  exit 1
fi

echo "==> Rebuilding Neo4j mirror CSV bundle from DuckDB"
uv run --group fraud python -m projects.fraud_anomaly_detection.neo4j_codex.neo4j_mirror.export \
  --store "$STORE" \
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
    --nodes=Scenario=/import/scenarios.csv \
    --relationships=USED_DEVICE=/import/used_device_rels.csv \
    --relationships=USED_BANK_ACCOUNT=/import/used_bank_account_rels.csv \
    --relationships=USED_PERSISTENT_ACCOUNT=/import/used_persistent_account_rels.csv \
    --relationships=USED_PHONE=/import/used_phone_rels.csv \
    --relationships=USED_ADDRESS=/import/used_address_rels.csv \
    --relationships=MATCHED_SCENARIO=/import/scenario_match_rels.csv

echo "==> Starting Neo4j (pinned image + mounted GDS $GDS_VERSION) on http://localhost:7474"
docker run -d \
  --name "$CONTAINER_NAME" \
  --user="$HOST_UID_GID" \
  --publish=7474:7474 \
  --publish=7687:7687 \
  --volume="$DATA_DIR:/data" \
  --volume="$LOGS_DIR:/logs" \
  --volume="$PLUGINS_DIR:/plugins" \
  --env "NEO4J_AUTH=$NEO4J_AUTH" \
  --env 'NEO4J_dbms_security_procedures_unrestricted=gds.*' \
  --env 'NEO4J_dbms_security_procedures_allowlist=gds.*' \
  --env "NEO4J_server_memory_heap_initial__size=$NEO4J_HEAP" \
  --env "NEO4J_server_memory_heap_max__size=$NEO4J_HEAP" \
  --env "NEO4J_server_memory_pagecache_size=$NEO4J_PAGECACHE" \
  --env "NEO4J_dbms_memory_transaction_total_max=$NEO4J_TXN_MAX" \
  "$NEO4J_IMAGE" >/dev/null

echo "==> Waiting for Neo4j + GDS to be ready"
ready=""
for _ in $(seq 1 60); do
  if docker exec "$CONTAINER_NAME" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASS" \
       "RETURN gds.version();" >/dev/null 2>&1; then
    ready="yes"; break
  fi
  sleep 5
done
if [ -z "$ready" ]; then
  echo "WARNING: Neo4j/GDS not ready in time. Check: docker logs -f $CONTAINER_NAME" >&2
else
  echo "==> Ensuring user_id index (fast native scenario seeding)"
  docker exec "$CONTAINER_NAME" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASS" \
    "CREATE INDEX user_id_idx IF NOT EXISTS FOR (u:User) ON (u.user_id);" >/dev/null 2>&1 || true
  echo "==> Healthcheck:"
  docker exec "$CONTAINER_NAME" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASS" --format plain \
    "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY n DESC;" || true
  docker exec "$CONTAINER_NAME" cypher-shell -u "$NEO4J_USER" -p "$NEO4J_PASS" --format plain \
    "RETURN gds.version() AS gds_version;" || true
fi

cat <<EOF

Neo4j mirror is up.

URL:      http://localhost:7474
Login:    $NEO4J_USER
Password: $NEO4J_PASS
GDS:      $GDS_VERSION (mounted from $PLUGINS_DIR)
Memory:   heap=$NEO4J_HEAP pagecache=$NEO4J_PAGECACHE txn-max=$NEO4J_TXN_MAX

Run the control report with:
  NEO4J_PASSWORD=$NEO4J_PASS uv run --with neo4j --group fraud python -m projects.fraud_anomaly_detection.neo4j_codex.control.control_loop_report --store $STORE --neo4j-uri bolt://localhost:7687

Check logs:
  docker logs -f $CONTAINER_NAME

Stop:
  bash $UNIT_DIR/neo4j_mirror/scripts/stop_neo4j.sh
EOF
