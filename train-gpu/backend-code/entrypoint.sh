#!/usr/bin/env sh
set -eu

api_pid=""
caddy_pid=""

shutdown() {
  if [ -n "$api_pid" ]; then
    kill "$api_pid" 2>/dev/null || true
  fi
  if [ -n "$caddy_pid" ]; then
    kill "$caddy_pid" 2>/dev/null || true
  fi
}

trap shutdown INT TERM

if [ "${TRAIN_MODE:-false}" = "true" ]; then
  echo "[entrypoint] Starting training mode"
  exec python -m ml.training.train
fi

if [ "${CRON_MODE:-false}" = "true" ]; then
  echo "[entrypoint] Starting data fetch cron mode"
  exec /app/cronjob.sh
fi

echo "[entrypoint] Starting API mode (uvicorn + caddy)"
uvicorn src.main:app --host 127.0.0.1 --port "${PORT:-8000}" --workers "${UVICORN_WORKERS:-1}" &
api_pid="$!"

caddy run --config /etc/caddy/Caddyfile --adapter caddyfile &
caddy_pid="$!"

while true; do
  if ! kill -0 "$api_pid" 2>/dev/null; then
    set +e
    wait "$api_pid"
    exit_code=$?
    set -e
    echo "[entrypoint] Uvicorn exited"
    kill "$caddy_pid" 2>/dev/null || true
    wait "$caddy_pid" 2>/dev/null || true
    exit "$exit_code"
  fi

  if ! kill -0 "$caddy_pid" 2>/dev/null; then
    set +e
    wait "$caddy_pid"
    exit_code=$?
    set -e
    echo "[entrypoint] Caddy exited"
    kill "$api_pid" 2>/dev/null || true
    wait "$api_pid" 2>/dev/null || true
    exit "$exit_code"
  fi

  sleep 2
done
