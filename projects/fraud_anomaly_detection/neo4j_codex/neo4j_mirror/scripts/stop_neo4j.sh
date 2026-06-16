#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-fraud-neo4j-codex-poc}"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
echo "Stopped $CONTAINER_NAME if it was running."

