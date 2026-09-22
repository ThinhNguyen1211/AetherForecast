#!/usr/bin/env sh
set -eu

# ---------------------------------------------------------------------------
# AetherForecast Entrypoint
#
# Supports 3 modes:
#   TRAIN_MODE=true  → exec training only (no API)
#   CRON_MODE=true   → exec ingestion cron only (no API)
#   (default)        → API mode: start ingestion cron in background,
#                       then run Uvicorn + Caddy in foreground.
#
# In default API mode, the ingestion cronjob runs as a background process
# alongside the API server inside a single container. This avoids needing
# a separate container or ECS task for the 15-minute data fetch loop.
# ---------------------------------------------------------------------------

api_pid=""
caddy_pid=""
cron_pid=""

shutdown() {
  echo "[entrypoint] Shutting down..."
  if [ -n "$cron_pid" ]; then
    kill "$cron_pid" 2>/dev/null || true
  fi
  if [ -n "$api_pid" ]; then
    kill "$api_pid" 2>/dev/null || true
  fi
  if [ -n "$caddy_pid" ]; then
    kill "$caddy_pid" 2>/dev/null || true
  fi
}

trap shutdown INT TERM

# --- Dedicated training mode ---
if [ "${TRAIN_MODE:-false}" = "true" ]; then
  echo "[entrypoint] Starting training mode"
  exec python -m ml.training.train
fi

# --- Dedicated cron-only mode (for separate container/task) ---
if [ "${CRON_MODE:-false}" = "true" ]; then
  echo "[entrypoint] Starting data fetch cron mode (standalone)"
  # cronjob.sh is single-shot by default (the EC2 host crontab owns scheduling).
  # A dedicated cron-only container has no such scheduler, so it must self-loop.
  INGESTION_LOOP="${INGESTION_LOOP:-1}"
  export INGESTION_LOOP
  exec /app/cronjob.sh
fi

# --- Default: API mode with embedded ingestion cron ---
echo "[entrypoint] Starting API mode (uvicorn + caddy + ingestion cron)"

# 1. Warm the dataset with a single ingestion pass in the background.
#    Recurring scheduling is owned by the EC2 host crontab
#    (/etc/cron.d/aetherforecast-fetch, */30). This used to start a second
#    perpetual 15-minute loop here, which ran concurrently with the host cron
#    and doubled S3 write volume. Set INGESTION_LOOP=1 to self-schedule instead
#    (for hosts with no crontab).
if [ "${INGESTION_ENABLED:-true}" = "true" ]; then
  echo "[entrypoint] Starting background ingestion cronjob..."
  /app/cronjob.sh &
  cron_pid="$!"
  echo "[entrypoint] Ingestion cron PID: ${cron_pid}"
else
  echo "[entrypoint] Ingestion cron DISABLED (INGESTION_ENABLED=false)"
fi

# 2. Start Uvicorn (FastAPI) in the background
uvicorn src.main:app --host 127.0.0.1 --port "${PORT:-8000}" --workers "${UVICORN_WORKERS:-1}" &
api_pid="$!"

# 3. Start Caddy reverse proxy in the background
caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &
caddy_pid="$!"

echo "[entrypoint] All services started:"
echo "[entrypoint]   Uvicorn  PID=${api_pid}"
echo "[entrypoint]   Caddy    PID=${caddy_pid}"
echo "[entrypoint]   Cron     PID=${cron_pid:-disabled}"

# 4. Monitor critical processes — exit if either Uvicorn or Caddy dies
while true; do
  if ! kill -0 "$api_pid" 2>/dev/null; then
    set +e
    wait "$api_pid"
    exit_code=$?
    set -e
    echo "[entrypoint] Uvicorn exited with code ${exit_code}"
    kill "$caddy_pid" 2>/dev/null || true
    [ -n "$cron_pid" ] && kill "$cron_pid" 2>/dev/null || true
    wait "$caddy_pid" 2>/dev/null || true
    exit "$exit_code"
  fi

  if ! kill -0 "$caddy_pid" 2>/dev/null; then
    set +e
    wait "$caddy_pid"
    exit_code=$?
    set -e
    echo "[entrypoint] Caddy exited with code ${exit_code}"
    kill "$api_pid" 2>/dev/null || true
    [ -n "$cron_pid" ] && kill "$cron_pid" 2>/dev/null || true
    wait "$api_pid" 2>/dev/null || true
    exit "$exit_code"
  fi

  sleep 2
done
