#!/usr/bin/env bash
# Trigger dataset generation against a running API container.
set -euo pipefail

LOG_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/generate.log"
URL="${API_URL:-http://127.0.0.1:8000}"

{
  echo "=========================================="
  echo "Date: $(date)"
  echo "Triggering dataset generation at $URL ..."
} | tee -a "$LOG_FILE"

curl -sS -X POST "$URL/v1/admin/generate-dataset" | tee -a "$LOG_FILE"
echo

echo "Check container logs for progress: docker compose logs -f api" | tee -a "$LOG_FILE"
