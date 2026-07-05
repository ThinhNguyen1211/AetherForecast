#!/usr/bin/env sh
set -eu

# ---------------------------------------------------------------------------
# AetherForecast Data Ingestion Cron
#
# Runs data_ingestion.py every 15 minutes in a loop.
# Logs to /var/log/aether_ingestion.log AND stdout (Docker logs).
# Sends syslog marker for journalctl tracing.
# ---------------------------------------------------------------------------

LOG_FILE="${AETHER_INGESTION_LOG:-/var/log/aether_ingestion.log}"
INTERVAL="${INGESTION_INTERVAL_SECONDS:-900}"

# Ensure log directory exists (writable by app user in Docker)
mkdir -p "$(dirname "$LOG_FILE")" 2>/dev/null || true

# Helper: log to both stdout and log file with timestamp
log() {
  local msg="[cron] $(date -u '+%Y-%m-%d %H:%M:%S UTC') $1"
  echo "$msg"
  echo "$msg" >> "$LOG_FILE" 2>/dev/null || true
}

# Syslog notification (if logger is available — works on EC2/systemd)
if command -v logger >/dev/null 2>&1; then
  logger -t aether-ingestion "Aether Data Ingestion Started (interval=${INTERVAL}s)"
fi

log "AetherForecast Data Ingestion Cron — starting loop"
log "Interval: ${INTERVAL} seconds ($(( INTERVAL / 60 )) minutes)"
log "Log file: ${LOG_FILE}"

# Ensure we run from the correct working directory
cd /app 2>/dev/null || cd "$(dirname "$0")" || true

while true; do
  log "Starting data ingestion run..."

  # Run ingestion, capturing output to both stdout and log file.
  # The `|| true` prevents set -e from killing the loop on failure.
  if python3 -m src.ml.data_ingestion >> "$LOG_FILE" 2>&1; then
    log "Data ingestion completed successfully."
  else
    exit_code=$?
    log "⚠ Data ingestion exited with code ${exit_code}"
    # Syslog the failure for alerting
    if command -v logger >/dev/null 2>&1; then
      logger -t aether-ingestion -p user.warning "Data ingestion failed with exit code ${exit_code}"
    fi
  fi

  log "Sleeping ${INTERVAL} seconds until next run..."
  sleep "$INTERVAL"
done
