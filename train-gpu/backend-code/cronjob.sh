#!/usr/bin/env sh
set -eu

echo "[cron] AetherForecast Data Ingestion Cron — starting loop"
echo "[cron] Interval: 900 seconds (15 minutes)"

while true; do
    echo "[cron] $(date -u '+%Y-%m-%d %H:%M:%S UTC') Starting data ingestion..."
    python -m src.ml.data_ingestion || echo "[cron] data_ingestion exited with code $?"
    echo "[cron] $(date -u '+%Y-%m-%d %H:%M:%S UTC') Sleeping 900 seconds..."
    sleep 900
done
