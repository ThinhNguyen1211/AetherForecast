# AetherForecast

Production-grade ML time-series forecasting platform for gold SJC and ~2000 Binance symbols.

## Architecture Overview
- Frontend: React + Vite + Tailwind on S3 + CloudFront.
- API and realtime: FastAPI + WebSocket on single EC2 t3.micro behind Caddy.
- Data ingestion: EC2 host cron (every 30 minutes) fetches Binance klines and writes partitioned parquet to S3.
- Inference: Hugging Face-based forecasting model loaded from S3 manifest.
- Training: AWS Batch Spot GPU jobs with LoRA fine-tuning and checkpoint recovery.
- Monitoring: CloudWatch dashboard, alarms, SNS notifications, and structured JSON logging.
