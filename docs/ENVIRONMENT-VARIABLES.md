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
- ORIGIN_VERIFY_SECRET: Shared CloudFront origin secret (only enforced if Caddy header check is enabled).

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
- HF_TOKENIZER_FALLBACK_ID
- MODEL_CACHE_DIR
- MODEL_TORCH_DTYPE
- INFERENCE_NUM_SAMPLES
- REQUIRE_S3_MODEL

## Sentiment + Covariates (Inference)
- SENTIMENT_MODE (simple or hf)
- SENTIMENT_MODEL_ID
- SENTIMENT_CACHE_DIR
- EXTERNAL_SENTIMENT_ENABLED
- EXTERNAL_SENTIMENT_REFRESH_SECONDS
- EXTERNAL_SENTIMENT_FORCE_REFRESH_PER_REQUEST
- EXTERNAL_SENTIMENT_REQUIRE_LIVE_SOURCES
- EXTERNAL_SENTIMENT_NEWS_RSS_URLS (comma-separated)
- EXTERNAL_NEWS_API_ENDPOINT
- EXTERNAL_NEWS_API_KEY
- EXTERNAL_NEWS_API_QUERY
- EXTERNAL_NEWS_API_LIMIT
- EXTERNAL_X_SENTIMENT_ENDPOINT
- EXTERNAL_X_SEARCH_ENDPOINT
- EXTERNAL_X_SEARCH_BEARER_TOKEN
- EXTERNAL_X_SEARCH_QUERY
- EXTERNAL_X_SEARCH_LIMIT
- EXTERNAL_GEOPOLITICAL_SENTIMENT_ENDPOINT
- EXTERNAL_EVENT_KEYWORDS (comma-separated)
- EXTERNAL_COVARIATES_ENABLED
- EXTERNAL_COVARIATES_REFRESH_SECONDS
- EXTERNAL_COVARIATE_SCALE

## Realtime
- BINANCE_WS_URL (default wss://stream.binance.com:9443)
- REALTIME_KLINE_INTERVAL (default 1m)

## AI Council (CrewAI)
- GEMINI_API_KEY (required for /api/ai/analyze)
- GEMINI_MODEL (default gemini-2.0-flash)

## Data Fetch Cron
Note: cronjob.sh runs src.ml.data_ingestion by default. Incremental fetcher vars below apply when using src.data.fetcher.
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
- TRAIN_SPLIT_RATIO
- WALK_FORWARD_WINDOWS
- WALK_FORWARD_EVAL_SIZE
- EXTERNAL_COVARIATE_SCALE
- ENABLE_EXTERNAL_FETCH
- STRICT_EXTERNAL_DATA
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
- BASE_MODEL_FALLBACK_ID
- TRAIN_OUTPUT_DIR
- HF_CACHE_DIR
- HF_TRUST_REMOTE_CODE
- HF_LOCAL_FILES_ONLY
- HF_FORCE_DOWNLOAD
- CHRONOS2_TRAIN_STEPS
- PREDICT_VARIANCE_SCALE
- PREDICT_DIFFUSION_STEPS

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
