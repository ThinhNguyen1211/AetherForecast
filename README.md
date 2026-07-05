# AetherForecast

Production-grade ML time-series forecasting platform for gold SJC and ~2000 Binance symbols.

## Architecture Overview
- Frontend: React + Vite + Tailwind on S3 + CloudFront.
- API and realtime: FastAPI + WebSocket on single EC2 t4g.small (Arm64) behind Caddy.
- Data ingestion: EC2 host cron (*/30 min) invokes a container loop (15 min cadence) that writes partitioned parquet to S3.
- Inference: Hugging Face-based forecasting model loaded from S3 manifest.
- Training: AWS Batch GPU jobs with LoRA fine-tuning and checkpoint recovery (on-demand default, spot optional).
- Optional: AI Council SSE (/api/ai/analyze) via CrewAI + Gemini.
- Monitoring: CloudWatch dashboard, alarms, SNS notifications, and structured JSON logging.
