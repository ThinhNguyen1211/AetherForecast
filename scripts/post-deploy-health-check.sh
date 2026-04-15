#!/usr/bin/env bash
set -euo pipefail

BACKEND_BASE_URL="${BACKEND_BASE_URL:-}"
WS_SYMBOL="${WS_SYMBOL:-BTCUSDT}"

if [[ -z "${BACKEND_BASE_URL}" ]]; then
  echo "Set BACKEND_BASE_URL before running this script, for example:"
  echo "  BACKEND_BASE_URL=https://api.example.com ./scripts/post-deploy-health-check.sh"
  exit 1
fi

echo "[check] API health"
curl -fsS "${BACKEND_BASE_URL}/health" | jq .

echo "[check] API docs reachable"
curl -fsSI "${BACKEND_BASE_URL}/docs" >/dev/null

echo "[check] websocket endpoint path"
WS_URL="${BACKEND_BASE_URL/https:/wss:}"
WS_URL="${WS_URL/http:/ws:}"
WS_URL="${WS_URL}/ws/${WS_SYMBOL}"
echo "  ${WS_URL}"

echo "Health checks passed."
