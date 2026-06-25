#!/usr/bin/env bash
# Send a couple of sample requests to a running API and print the results.
set -euo pipefail

URL="${API_URL:-http://127.0.0.1:8000}"

send() {
  local prompt="$1"
  echo
  echo "Prompt: $prompt"
  curl -sS -X POST "$URL/v1/chat/completions" \
    -H 'Content-Type: application/json' \
    -d "{\"user_id\": \"user_123\", \"prompt\": \"$prompt\"}" | python3 -m json.tool
}

echo "Smoke test against $URL"
send "What is the capital of Australia?"
send "Explain the technical difference between v-if and v-show in Vue.js."
send "Summarize the benefits of NVMe SSDs compared to traditional HDDs."
