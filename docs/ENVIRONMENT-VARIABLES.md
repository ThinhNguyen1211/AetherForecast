# Environment Variables

This file documents runtime configuration for AetherForecast services.

## Backend Core
- APP_NAME: Service display name.
- APP_ENV: dev or prod.
- LOG_LEVEL: DEBUG/INFO/WARN/ERROR.
- PORT: FastAPI upstream port behind Caddy (default 8000).
- UVICORN_WORKERS: API worker count.
- APP_DOMAIN: Caddy site address. Use :80 for HTTP-only bootstrap, or a DNS hostname for automatic HTTPS.
- LETSENCRYPT_EMAIL: Email for Caddy ACME registration.

## AWS + Storage
- AWS_REGION: AWS region for S3, CloudWatch, etc.
- AWS_ENDPOINT_URL: Optional localstack/custom endpoint.
- DATA_BUCKET: Primary parquet bucket.
- DATA_S3_BUCKET: Optional alias for training/fetcher.
- MODEL_BUCKET: Primary model bucket.
- MODEL_S3_URI: Root S3 model URI.
- CHECKPOINT_S3_URI: S3 prefix for training checkpoints.

## Auth
- COGNITO_USER_POOL_ID
- COGNITO_CLIENT_ID
- COGNITO_REGION

## Inference
- HF_MODEL_FALLBACK_ID
- MODEL_CACHE_DIR
- INFERENCE_NUM_SAMPLES

## Realtime
- BINANCE_WS_URL (default wss://stream.binance.com:9443)
- REALTIME_KLINE_INTERVAL (default 1m)

## Data Fetch Cron
- CRON_MODE=true to run fetcher mode in container (invoked by host cron in production).
- BINANCE_BASE_URL
- KLINE_INTERVAL
- FETCH_CONCURRENCY
- FETCH_LOOP_SECONDS
- BOOTSTRAP_LOOKBACK_MINUTES
- MAX_KLINE_PAGES
- SYMBOLS
- SYMBOL_LIMIT
- QUOTE_ASSETS
- PARQUET_PREFIX
- WATERMARK_PREFIX
- SENTIMENT_MODE (simple or hf)
- HF_SENTIMENT_MODEL_ID

## Training
- TRAIN_MODE=true to run training mode in container.
- TIMEFRAME
- TRAINING_HORIZON
- CONTEXT_LENGTH
- MAX_ROWS_PER_SYMBOL
- EPOCHS
- LEARNING_RATE
- BATCH_SIZE
- GRAD_ACCUM_STEPS
- WARMUP_RATIO
- WEIGHT_DECAY
- SAVE_STEPS
- EVAL_STEPS
- LOGGING_STEPS
- LORA_R
- LORA_ALPHA
- LORA_DROPOUT
- MAX_SEQ_LENGTH
- BASE_MODEL_ID
- TRAIN_OUTPUT_DIR
- HF_CACHE_DIR

## Frontend
- VITE_API_BASE_URL (set to EC2/Caddy public API URL, e.g. https://api.example.com)

## GitHub Actions (Repository Variables)
- AWS_GITHUB_OIDC_ROLE_ARN
- BACKEND_EC2_INSTANCE_ID
- FRONTEND_S3_BUCKET
- CLOUDFRONT_DISTRIBUTION_ID
- PROD_APPROVERS

## GitHub Actions (Repository Secrets)
- SLACK_WEBHOOK_URL (optional)
- TEAMS_WEBHOOK_URL (optional)

Never commit real credentials, .env files, or long-term access keys.
