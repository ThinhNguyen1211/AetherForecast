# AetherForecast

Production-grade ML time-series forecasting platform for gold SJC and ~2000 Binance symbols.

## Architecture Overview
- Frontend: React + Vite + Tailwind on S3 + CloudFront.
- API and realtime: FastAPI + WebSocket on single EC2 t3.micro behind Caddy.
- Data ingestion: EC2 host cron (every 15 minutes) fetches Binance klines and writes partitioned parquet to S3.
- Inference: Hugging Face-based forecasting model loaded from S3 manifest.
- Training: AWS Batch Spot GPU jobs with LoRA fine-tuning and checkpoint recovery.
- Monitoring: CloudWatch dashboard, alarms, SNS notifications, and structured JSON logging.

## Monorepo Layout

```text
AetherForecast/
├── packages/
│   ├── infra/      # AWS CDK v2 infrastructure
│   ├── backend/    # FastAPI API, realtime, cron, training code
│   └── frontend/   # React trading dashboard
├── docs/           # Runbook, handover, verification, env vars
├── .github/workflows/
└── deploy.sh
```

## One-Command Deploy (Final)

```bash
cd packages/infra && npx cdk deploy --all
```

## Build And Validate

```bash
npm install
npm run infra:build
npm run infra:synth
```

## Monitoring Links
- Dashboard name: `AetherForecast-Operations`
- CloudWatch Dashboard URL template:
  - `https://<region>.console.aws.amazon.com/cloudwatch/home?region=<region>#dashboards:name=AetherForecast-Operations`
- CloudWatch Alarms URL template:
  - `https://<region>.console.aws.amazon.com/cloudwatch/home?region=<region>#alarmsV2:`

## Cost Overview (Low Traffic Estimate)
- Estimated monthly range: **$65-$315** depending on traffic and training frequency.
- Major cost drivers:
  - EC2 backend host (always-on)
  - AWS Batch Spot GPU training usage
  - CloudWatch logs/metrics and S3 storage growth

## Operational Documentation
- Runbook: [docs/RUNBOOK.md](docs/RUNBOOK.md)
- Handover summary: [docs/HANDOVER.md](docs/HANDOVER.md)
- End-to-end acceptance checklist: [docs/VERIFICATION-CHECKLIST.md](docs/VERIFICATION-CHECKLIST.md)
- Environment variable reference: [docs/ENVIRONMENT-VARIABLES.md](docs/ENVIRONMENT-VARIABLES.md)

## Data and Model Path Conventions
- Parquet data: `s3://<data-bucket>/market/klines/symbol=<SYMBOL>/year=<YYYY>/month=<MM>/day=<DD>/`
- Training checkpoints: `s3://<model-bucket>/checkpoints/`
- Model versions: `s3://<model-bucket>/chronos-v1/model/versions/<timestamp>/`
- Active model manifest: `s3://<model-bucket>/chronos-v1/model/manifest/latest.json`
