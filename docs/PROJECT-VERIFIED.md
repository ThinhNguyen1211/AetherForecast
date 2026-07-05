# AetherForecast - Verified Project Notes (generated 2026-06-02)

## Scope
These notes are verified from repository code, infra (CDK), and docs as of 2026-06-02.

## Verified Architecture (code + infra)
- Region: ap-southeast-1 (default in config and CDK).
- Frontend: React 18 + Vite + Tailwind, hosted on S3 + CloudFront.
- Backend: FastAPI + Caddy in a single container on EC2 (default t4g.small, Arm64).
- Realtime: Binance WebSocket -> backend hub -> frontend.
- Storage: S3 parquet data lake; S3 model bucket with manifest/latest.json.
- Auth: Cognito JWT.
- Monitoring: CloudWatch logs/metrics/alarms + SNS.
- Training: AWS Batch + optional training EC2; LoRA fine-tuning for Chronos-2.

## Backend APIs (current)
- GET /health
- GET /symbols (Binance exchangeInfo, fallback static list)
- GET /chart/{symbol} (live Binance REST; not S3)
- POST /predict (S3 parquet history + Chronos-2 inference + sentiment)
- WS /ws/{symbol}?timeframe=...
- POST /api/ai/analyze (SSE via CrewAI + Gemini; requires GEMINI_API_KEY)

## Data Ingestion
Primary cron entrypoint: /app/cronjob.sh -> python -m src.ml.data_ingestion
- Interval: 900s loop (15 min) once running.
- Fetches spot OHLCV + futures funding/OI/LS + macro (DXY, US10Y via yfinance).
- Writes parquet to s3://<data-bucket>/market/klines/symbol=SYM/ partitioned by year/month.
Secondary pipeline (not used by cron by default): src.data.fetcher + parquet_writer
- Incremental fetch, watermark, partitions year/month/day, includes timeframe.

## Known discrepancies vs docs
1) EC2 size: docs say t3.micro; infra default is t4g.small (Arm64).
2) Chart data source: docs say S3 parquet; actual chart endpoint hits Binance REST.
3) Cron cadence: docs say every 30 min; actual cron job is scheduled every 30 min but runs a 15-min loop once started.
4) Parquet partitions: docs mention day partitions; primary ingestion writes year/month only (day used only in secondary fetcher).
5) Batch Spot: docs say spot GPU; CDK keeps spot compute env disabled and on-demand enabled.
6) Origin secret: infra sets ORIGIN_VERIFY_SECRET but Caddyfile does not enforce header check.
7) Extra AI feature: /api/ai/analyze (CrewAI + Gemini) not documented; requires GEMINI_API_KEY/GEMINI_MODEL env.
8) External sentiment and covariate env vars not listed in docs/ENVIRONMENT-VARIABLES.

## Suggested doc updates
- Update README.md, docs/HANDOVER.md, docs/RUNBOOK.md for EC2 type, chart source, cron cadence.
- Update ENVIRONMENT-VARIABLES.md to include GEMINI_API_KEY/GEMINI_MODEL and external sentiment/covariate fields.
