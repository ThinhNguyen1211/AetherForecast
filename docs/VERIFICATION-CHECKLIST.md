# End-to-End Verification Checklist

Use this checklist after each major deploy or before handover acceptance.

## Frontend & Auth
- [ ] Frontend loads successfully with cosmic eye branding.
- [ ] User can provide Cognito JWT in UI.
- [ ] Protected routes work with valid token and fail with invalid token.

## Chart & Realtime
- [ ] Symbol list loads.
- [ ] Historical chart candles load from backend parquet source.
- [ ] WebSocket stream works at wss://<api-domain>/ws/BTCUSDT.
- [ ] Realtime candle updates are rendered on chart.

## Inference
- [ ] /predict endpoint returns valid forecast with confidence interval.
- [ ] Response includes model_name/model_version and explanation.
- [ ] Inference latency remains within expected SLO.

## Data Cron + Parquet
- [ ] Manual host cron trigger (`/usr/local/bin/aetherforecast-fetch-cron.sh`) runs successfully.
- [ ] New parquet objects appear in symbol/year/month/day partitions.
- [ ] Watermark files update in metadata prefix.
- [ ] FetchErrors metric remains low/zero.

## Training + Promotion
- [ ] Manual Batch training job runs to completion.
- [ ] Checkpoints are written to model bucket checkpoints prefix.
- [ ] New model version uploaded to versions path.
- [ ] Manifest latest.json updates active_model_s3_uri.
- [ ] Inference uses promoted model after cache window.

## Monitoring + Alarms
- [ ] CloudWatch dashboard AetherForecast-Operations is visible.
- [ ] EC2 CPU/status, API latency/5xx, Batch, cron widgets update.
- [ ] Alarm SNS topic exists and has at least one subscription.
- [ ] Synthetic alarm test triggers notification successfully.

## CI/CD
- [ ] Backend workflow passes lint/type/test/security stages.
- [ ] Frontend workflow builds artifact and deploys on main.
- [ ] Infra workflow synth/diff/deploy with approval gate works.
