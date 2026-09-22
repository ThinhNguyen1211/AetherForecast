#!/usr/bin/env sh
set -eu

# ---------------------------------------------------------------------------
# AetherForecast Data Ingestion
#
# Runs ONE data_ingestion.py pass and exits. Scheduling is owned by the host
# crontab (/etc/cron.d/aetherforecast-fetch, */30), which invokes this under
# `flock -n`.
#
# This script used to `while true; sleep 900`. Because it never returned, the
# very first cron invocation held the flock forever: every later cron run hit
# the lock and exited 1, so /var/lib/aetherforecast/cron-last-success was never
# written and the CronHealthy metric sat at 0 permanently. Meanwhile the loop
# ingested every 15 min rather than the intended 30, doubling S3 write volume.
#
# Set INGESTION_LOOP=1 to restore the old self-scheduling behaviour (for
# running outside cron, e.g. a bare container).
# ---------------------------------------------------------------------------

LOG_FILE="${AETHER_INGESTION_LOG:-/var/log/aether_ingestion.log}"
INTERVAL="${INGESTION_INTERVAL_SECONDS:-1800}"
LOOP="${INGESTION_LOOP:-0}"

mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

log() {
  msg="[cron] $(date -u '+%Y-%m-%d %H:%M:%S UTC') $1"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE" 2>/dev/null || true
}

cd /app 2>/dev/null || cd "$(dirname "$0")" || true

run_once() {
  log "Starting data ingestion run..."
  if python3 -m src.ml.data_ingestion >> "$LOG_FILE" 2>&1; then
    log "Data ingestion completed successfully."
    return 0
  fi
  exit_code=$?
  log "! Data ingestion exited with code ${exit_code}"
  if command -v logger >/dev/null 2>&1; then
    logger -t aether-ingestion -p user.warning "Data ingestion failed with exit code ${exit_code}"
  fi
  return "$exit_code"
}

if command -v logger >/dev/null 2>&1; then
  logger -t aether-ingestion "Aether Data Ingestion invoked (loop=${LOOP})"
fi

if [ "$LOOP" = "1" ]; then
  log "Loop mode enabled — interval ${INTERVAL}s ($(( INTERVAL / 60 )) minutes)"
  while true; do
    run_once || true
    log "Sleeping ${INTERVAL} seconds until next run..."
    sleep "$INTERVAL"
  done
fi

run_once
