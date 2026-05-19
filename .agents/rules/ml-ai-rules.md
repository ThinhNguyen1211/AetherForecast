---
trigger: always_on
---

# ML & AI Rules for Time-Series Forecasting

## General ML Principles
- Dữ liệu phải thật 100% từ S3 + realtime + external sources.
- Input model: candles OHLCV + TA-Lib patterns + volatility + external covariates (fear-greed, funding rate, news, X, macro).
- Model: Chronos-2 base + LoRA (rank 16), multi-timeframe training, walk-forward validation.
- Không dùng statistical fallback trong production prediction.

## Training Rules
- Batch size nhỏ (2), context 1024, epochs 5-8 cho production.
- External data fetch realtime khi predict.
- Promote model qua manifest/latest.json trên S3.

## Prediction Rules
- Mỗi predict: fetch fresh external sentiment + covariates → append vào input.
- Post-processing: variance scaling để prediction có biến động thực tế.
- Output: clear confidence bands, sentiment source, trend.

## Best Practices
- Luôn đo lường chất lượng (walk-forward, backtest).
- Monitor drift (model performance theo thời gian).
- Experiment tracking (nếu có MLflow hoặc simple logging).

Áp dụng cho mọi ML-related code.