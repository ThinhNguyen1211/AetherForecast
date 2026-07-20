# AetherForecast - Tài liệu dự án đầy đủ (quét code + hạ tầng)

> Tài liệu này mô tả trạng thái hiện tại của dự án dựa trên code, cấu hình CDK, scripts và docs trong repo.
> Ngày cập nhật: 2026-06-03.

---

## 1) Tổng quan nhanh

**Mục tiêu**: Xây dựng dashboard dự báo giá tài chính thời gian thực cho crypto (spot Binance) và vàng, với chart chất lượng cao, dự báo AI kèm độ tin cậy và overlay dự báo trực quan.

**Giá trị cốt lõi**:
- Dữ liệu real-time (Binance REST + WebSocket), data lake S3 parquet.
- Dự báo bằng Chronos-2 + LoRA + sentiment + external covariates.
- UX mượt, chart nhẹ, realtime ổn định.
- Vận hành đơn giản, cost-optimized, CI/CD rõ ràng.

**Người dùng mục tiêu**:
- Trader/nhà đầu tư crypto & vàng.
- Ops/ML engineer vận hành hệ thống và train model.

---

## 2) Kiến trúc tổng thể (End-to-End)

### 2.1 Luồng dữ liệu chính
1. **Realtime/Chart**:
   - Binance REST → backend `/chart/{symbol}` → frontend chart.
   - Binance WebSocket → backend realtime hub → frontend cập nhật realtime.

2. **Data Lake**:
   - Cron trên EC2 gọi container `cronjob.sh` (loop 15 phút) → `src.ml.data_ingestion` → S3 parquet.

3. **Inference**:
   - Frontend gọi `/predict` → backend đọc S3 parquet + sentiment + covariates → Chronos-2 → forecast + confidence bands.

4. **AI Council (tùy chọn)**:
   - Frontend gọi `/api/ai/analyze` (SSE) → CrewAI + Gemini → quyết định LONG/SHORT/HOLD.

5. **Training**:
   - Local/GPU/EC2/BATCH → fine-tune Chronos-2 + LoRA → upload model → cập nhật `manifest/latest.json`.

### 2.2 Thành phần chính
- **Frontend**: React 18 + Vite + Tailwind, S3 + CloudFront.
- **Backend**: FastAPI + Caddy, chạy trên EC2 t4g.small (Arm64).
- **Auth**: Cognito JWT.
- **Storage**: S3 parquet data lake + S3 model bucket.
- **Training**: AWS Batch GPU + optional Training EC2.
- **Monitoring**: CloudWatch logs/metrics/alarms + SNS.

---

## 3) Cấu trúc repo chính

```
AetherForecast/
  packages/
    backend/              # FastAPI + ML + Caddy
    frontend/             # React + Vite + Tailwind
    infra/                # CDK stacks
  scripts/                # Ops & deploy scripts
  docs/                   # Runbook + handover + verification
  train-gpu/              # Local GPU env (venv)
  artifacts/              # HF cache + training outputs
```

---

## 4) Backend (FastAPI)

### 4.1 Tech stack
- Python 3.11, FastAPI, Uvicorn, Pydantic v2.
- Caddy reverse proxy (HTTPS, gzip/zstd).
- Auth: Cognito JWT validation.
- Metrics: CloudWatch custom metrics.
- Rate limit: SlowAPI.
- ML: transformers + chronos-forecasting + torch + peft.

### 4.2 API endpoints
- `GET /health`: health check.
- `GET /symbols`: danh sách symbols (Binance exchangeInfo, fallback list).
- `GET /chart/{symbol}`: lấy candles từ Binance REST (theo timeframe, limit, from_timestamp).
- `POST /predict`: dự báo giá từ S3 parquet + inference.
- `WS /ws/{symbol}`: realtime kline stream (timeframe qua query string).
- `POST /api/ai/analyze` (SSE): AI Council (CrewAI + Gemini), rate limit 5 req/hour.

### 4.3 Auth & bảo mật
- JWT từ Cognito, middleware xác thực Bearer token.
- CORS configurable.
- Optional `ORIGIN_VERIFY_SECRET` (CloudFront → Caddy) nhưng hiện tại Caddy chưa enforce check.

### 4.4 Logging & metrics
- JSON structured logs (structlog).
- Custom metrics: ApiRequests, ApiLatencyMs, Api5xx, WebSocketConnections, FetchRuns, FetchErrors, etc.
- CloudWatch log groups: `/aetherforecast/ec2-backend`, `/aetherforecast/training-ec2-manual`, batch log group.

### 4.5 Realtime engine
- Backend giữ WebSocket connections cho từng symbol/timeframe.
- Mỗi stream là một kết nối Binance WS riêng (multiplex theo symbol/timeframe).
- Tự reconnect khi timeout/error.

---

## 5) Inference pipeline (/predict)

### 5.1 Nguồn dữ liệu
- Lấy candles từ S3 parquet (primary).
- Fallback thêm candles từ Binance REST nếu thiếu.
- Có caching ngắn hạn theo symbol/timeframe.

### 5.2 Feature engineering trong inference
- Candlestick patterns (TA-Lib hoặc fallback tự tính).
- Multi-window realized volatility.
- ATR, momentum, trend strength, wick imbalance.
- External covariates: funding rate, open interest, long/short ratio, btc dominance, DXY, US10Y, sentiment events.
- Sentiment blending: market + external (RSS + X + geopolitics + FNG).

### 5.3 Model inference
- Chronos-2 (HF) + LoRA.
- Hỗ trợ predict quantiles nếu model hỗ trợ, fallback to sampling.
- Post-process: enforce monotonic quantiles + dynamic minimum spread dựa trên realized vol.

### 5.4 Output chính
- `prediction_array`, `confidence_bands`, `volatility_bands`.
- `pattern_markers` (bullish/bearish signals).
- `confidence_interval`, `confidence`, `trend_direction`.
- `sentiment_score`, `external_sentiment_score` + sources.
- `model_name`, `model_version`, `explanation`.

---

## 6) Data ingestion (parquet)

### 6.1 Pipeline chính (cronjob.sh)
- Entrypoint: `cronjob.sh` → `python -m src.ml.data_ingestion`.
- Loop interval: 900s (15 phút) bên trong container.
- Cron host EC2 kích hoạt mỗi 30 phút.

**Nguồn dữ liệu**:
- Binance Spot OHLCV.
- Binance Futures: funding rate, open interest, long/short ratio.
- Macro: DXY, US10Y (yfinance).
- TA indicators: RSI, MACD, Bollinger, ATR.

**Output**:
- S3 parquet: `s3://<data-bucket>/market/klines/symbol=<SYM>/year=<YYYY>/month=<MM>/`.

### 6.2 Pipeline incremental (tuỳ chọn)
- Module: `src.data.fetcher` + `src.data.parquet_writer`.
- Uses watermarking (`_metadata/watermarks`).
- Partition theo `symbol/year/month/day`, có timeframe.
- Thêm sentiment series (từ SentimentScorer).

---

## 7) Sentiment & External covariates

### 7.1 Sentiment engine
- RSS: CoinDesk, CoinTelegraph, Reuters...
- News API (tuỳ config), X/Twitter search (tuỳ config).
- Fear & Greed Index.
- Geopolitical feeds (Reuters/BBC).
- HF sentiment model (FinBERT) nếu bật.

### 7.2 Covariates
- Funding rate, open interest, long/short ratio, top trader ratio.
- Taker buy/sell ratio.
- BTC dominance (CoinGecko).
- Macro DXY + US10Y (FRED).
- Event impact score từ headline keywords.

---

## 8) Training pipeline

### 8.1 Training entrypoints
- `packages/backend/ml/training/train.py`: pipeline chính.
- `packages/backend/train-local.py`: wrapper training local + promote manifest.
- Training trên Training EC2 duoc trigger qua AWS Batch job definition hoac SSM truc tiep.

### 8.2 Training flow
1. Load parquet từ S3 (multi-timeframe resample nếu cần).
2. Build dataset (walk-forward validation windows).
3. Fine-tune Chronos-2 với LoRA.
4. Checkpoint định kỳ lên S3.
5. Upload final model lên S3 `versions/<timestamp>/`.
6. Update `manifest/latest.json` để promote model.

### 8.3 Training environment
- AWS Batch GPU (g4dn.2xlarge) on-demand default, spot optional.
- Training EC2 GPU (g4dn.xlarge) + SSM.
- Local GPU via `train-gpu` venv.

---

## 9) Frontend (React + Vite)

### 9.1 Pages
- **LandingPage**: giới thiệu, sign in/up, highlight features.
- **Dashboard**: chart + prediction + AI council + sidebar symbols.

### 9.2 State management
- Zustand `useMarketStore`: token, symbol, timeframe, candles, prediction.
- LocalStorage cho JWT token.

### 9.3 Chart engine
- TradingView Lightweight Charts.
- Candlestick series + overlays:
  - Forecast line (median).
  - Confidence bands.
  - RSI, MACD, Bollinger (frontend-calculated).
- Cơ chế lazy loading & caching candle data.

### 9.4 Prediction UI
- Horizon selection (6h / 24h / 3d / 7d).
- Panel hiển thị confidence, delta %, sentiment.
- Prediction pipeline progress steps.

### 9.5 AI Council UI
- Button chạy `/api/ai/analyze`.
- Hiển thị SSE log dạng terminal.
- Render final decision LONG/SHORT/HOLD.

### 9.6 Auth UI
- Custom modal sign in / sign up / confirm code.
- Cognito API trực tiếp (InitiateAuth, SignUp, ConfirmSignUp).

---

## 10) Infrastructure (CDK)

### 10.1 Stacks chính (đang sử dụng)
- **NetworkStack**: VPC public + private isolated, no NAT.
- **StorageStack**: S3 parquet bucket + model bucket.
- **AuthStack**: Cognito User Pool + Client.
- **Ec2Stack**: backend EC2 t4g.small (Arm64) + EIP + cron + logs.
- **TrainingEc2Stack**: EC2 GPU training (g4dn.xlarge).
- **MlBatchStack**: AWS Batch GPU jobs.
- **FrontendHostingStack**: S3 + CloudFront.
- **MonitoringStack**: CloudWatch dashboards + alarms + SNS.

### 10.2 Stacks khác (chưa wiring vào main stack)
- **ComputeStack**: ECS Fargate + ALB (legacy/optional).
- **SchedulerStack**: Scheduled Fargate task (legacy/optional).

---

## 11) CI/CD (GitHub Actions)

### Backend CI/CD
- Lint + mypy + bandit + pytest.
- Build Docker runtime image.
- Push ECR (arm64) + Trivy scan.
- Rollout EC2 qua SSM (canary + health check).

### Frontend CI/CD
- TypeScript check, ESLint/Prettier warning-only.
- Build artifact + upload.
- Sync lên S3 + invalidate CloudFront.

### Infra CI/CD
- CDK build + synth.

### Security pipeline
- git-secrets scan.
- Trivy config + secret scan.

---

## 12) Ops scripts

- `scripts/post-deploy-health-check.sh`: health + docs + ws check.
- Cac wrapper PowerShell cu (`deploy-e2e.ps1`, `af-ops.ps1`) va script lifecycle EC2 training (`start-training.sh`, `stop-training.sh`) da duoc loai bo; dung CDK CLI/AWS CLI truc tiep.

---

## 13) Môi trường & cấu hình

- `.env.example` trong backend cho dev.
- ENV vars cho inference, sentiment, covariates, training, auth.
- Secrets qua GitHub Actions + SSM (no hardcoded keys).

---

## 14) Những điểm đáng chú ý

- **/chart** hiện lấy Binance REST trực tiếp (không phải S3 parquet).
- **Cron**: host chạy 30 phút/lần, bên trong container loop 15 phút.
- **Batch Spot**: có cấu hình nhưng default = on-demand.
- **AI Council**: cần GEMINI_API_KEY.
- **Origin secret** có env nhưng chưa enforce ở Caddyfile.

---

## 15) Gợi ý cập nhật/hoàn thiện tiếp

- Nếu cần S3 chart, thay /chart để dùng S3ParquetClient.
- Bổ sung Caddy origin verify header để bảo vệ origin.
- Đồng bộ thêm docs/ops VN nếu mở rộng luồng ECS/Fargate.
- Mở rộng test coverage cho predict/sentiment.

---

## 16) Tóm tắt nhanh để đưa vào luận văn

AetherForecast là hệ thống dự báo time-series sản xuất sẵn, kết hợp data pipeline (Binance + macro + sentiment), ML model Chronos-2 fine-tune với LoRA, và dashboard realtime trên web. Dữ liệu được lưu ở S3 parquet, inference sử dụng external covariates + sentiment để tăng ổn định dự báo. Hệ thống triển khai trên AWS với EC2 + S3 + CloudFront + Cognito + Batch GPU, có CI/CD và monitoring đầy đủ.

---

Nếu cần, tôi có thể tiếp tục mở rộng tài liệu này với sơ đồ Mermaid, sequence diagrams, hoặc mapping từng module sang mục trong luận văn.