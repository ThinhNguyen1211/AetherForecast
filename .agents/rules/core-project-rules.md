---
trigger: always_on
---

# AetherForecast Core Project Rules

## Project Overview & Goals
- Đây là dashboard ML dự đoán giá crypto (Binance spot) và vàng (XAUUSD, PAXG).
- Scope: Chart tương tác Binance-like + ML prediction overlay (Chronos-2 + LoRA) + realtime WebSocket.
- Không có giao dịch, chỉ dashboard + prediction.
- Ưu tiên: UX mượt mà, prediction chất lượng cao, chi phí AWS thấp, code sạch, production-ready.

## General Principles (áp dụng mọi lúc)
- Luôn nghĩ đến **production-grade**: security, cost optimization, scalability, observability (CloudWatch, logs structured).
- Code phải **readable, maintainable, testable**. Ưu tiên explicit > implicit.
- Sử dụng least-privilege, infrastructure as code (AWS CDK TypeScript), immutable infrastructure khi có thể.
- Error handling: graceful degradation, clear user messages, không crash toàn bộ service.
- Performance: lazy loading, caching (Zustand + S3 partition), minimal re-renders.

## Tech Stack & Conventions
- **Backend**: FastAPI (Python), Docker, Caddy reverse proxy.
- **Frontend**: React + Vite + TypeScript + Tailwind + Zustand + Lightweight Charts (hoặc TradingView nếu cần).
- **Infrastructure**: AWS CDK v2 (TypeScript), EC2 (t4g.small), S3 Parquet partitioned, Batch (nếu train), Cognito.
- **ML**: Hugging Face Chronos-2 + LoRA, awswrangler/polars cho data, TA-Lib cho features.
- Naming: snake_case Python, camelCase TS/React, kebab-case file/folder.
- TypeScript strict mode, Python type hints mạnh (mypy).
- Logging: structlog (structured JSON).

## AWS & DevOps Best Practices
- Luôn dùng CDK để định nghĩa hạ tầng (không console click).
- VPC, subnets, security groups least-privilege.
- Cost awareness: Spot instances cho train, public subnet khi cần, giảm S3 requests (cron 30 phút).
- CI/CD: GitHub Actions + OIDC, lint/type/test/security scan trước deploy.
- Monitoring: CloudWatch alarms + dashboards, structured logs.
- Secret management: không commit .env, dùng AWS Secrets Manager hoặc GitHub Secrets.

## ML/AI Specific Rules
- Dữ liệu phải thật 100% (từ S3 Parquet + realtime Binance + external sources).
- Input cho model: candles + volatility + candlestick patterns + external sentiment/covariates (fear-greed, funding rate, news, X, macro).
- Fine-tune: LoRA rank 16, multi-timeframe (1h/4h/1d), walk-forward validation.
- Prediction: không fallback statistical, luôn dùng promoted model từ manifest/latest.json.
- External sentiment phải fetch realtime từ nhiều nguồn khi predict.

## UI/UX Principles (áp dụng cho mọi component)
- Dark theme cosmic (màu tím, xanh neon, glow nhẹ như logo eye).
- Binance-like layout: top bar, sidebar symbols searchable, main chart lớn, prediction panel bên phải.
- Minimal icons, clean typography, high contrast.
- Loading: skeleton/shimmer effect, không block UI (silent lazy load khi scroll quá khứ).
- Chart: realtime mượt, anti-snap khi user kéo lịch sử, prediction overlay tím mờ nhưng có variance thực tế.
- Responsive, accessible (ARIA), mobile-friendly khi có thể.