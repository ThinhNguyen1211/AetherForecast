# HƯỚNG DẪN CHI TIẾT — AetherForecast

Tài liệu này dành cho việc rà soát (review) đồ án tốt nghiệp AetherForecast. Nội dung được biên soạn dựa trên trạng thái thực tế của mã nguồn tại thời điểm hoàn thiện, không dựa trên tài liệu cũ.

---

## Phần 1: Hướng dẫn Cài đặt & Chạy Local

### 1.1. Clone repository

Đây là repository công khai, **chỉ đọc** (read-only) đối với người rà soát — không cần quyền push:

```bash
git clone https://github.com/ThinhNguyen1211/AetherForecast.git
cd AetherForecast
```

### 1.2. Cài đặt & chạy Frontend

```bash
cd packages/frontend
npm install
npm run dev
```

Frontend sẽ chạy tại `http://localhost:5173` (cổng mặc định của Vite).

### 1.3. Lưu ý quan trọng: vì sao nên dùng website production để test đầy đủ tính năng

Đây là điểm **cần đọc kỹ** trước khi rà soát API/AI Council: chạy Frontend ở local **không thể** kiểm thử đầy đủ tính năng API/AI Council một cách "cắm là chạy" (out-of-the-box), vì các lý do sau — đã được xác minh trực tiếp trong mã nguồn, không phải suy đoán:

1. **Frontend local mặc định KHÔNG gọi vào backend production.** Hàm `resolveApiBaseUrl()` (`packages/frontend/src/services/api.ts`) khi chạy trên `localhost` và không có biến môi trường `VITE_API_BASE_URL`, sẽ tự động trỏ về `http://localhost:8000` — tức là một backend local (chưa chạy), **không phải** `api.aetherforcast.io.vn`. Muốn trỏ về production, phải tự tạo file `.env` với `VITE_API_BASE_URL=https://api.aetherforcast.io.vn`.
2. **Ngay cả khi đã trỏ đúng URL, CORS ở backend production KHÔNG chặn `localhost:5173`/`localhost:3000`** — danh sách origin được phép trong `packages/backend/src/main.py` (biến `_origins`) đã chủ động thêm sẵn 2 cổng dev mặc định này. Vì vậy đây **không phải** là rào cản chính (khác với giả định ban đầu rằng "localhost sẽ luôn bị CORS chặn").
3. **Rào cản thực sự** nằm ở xác thực: mọi endpoint quan trọng (`/predict`, `/api/ai/analyze`) đều yêu cầu `require_authenticated_user` (JWT hợp lệ từ AWS Cognito User Pool thật của hệ thống production) — một bản clone local không có sẵn tài khoản/luồng đăng nhập Cognito hợp lệ để tự tạo token này. Ngoài ra `/api/ai/analyze` còn bị giới hạn `5 requests/hour` theo IP.
4. **Chạy backend đầy đủ ở local** cũng đòi hỏi: mô hình Chronos-2 (tải từ S3, ~vài trăm MB), khóa API cho DeepSeek (AI Council), và các khóa sentiment bên ngoài — không thực tế cho mục đích rà soát nhanh.

**Khuyến nghị**: để xem hệ thống hoạt động đầy đủ (dự đoán giá, AI Council, biểu đồ real-time), vui lòng dùng trực tiếp website production: **https://aetherforcast.io.vn**. Chạy Frontend ở local chỉ phù hợp để đọc/kiểm tra mã nguồn giao diện, không phải để kiểm thử luồng dữ liệu thật.

---

## Phần 2: Các tinh chỉnh Mock API (AI-Assisted Changes)

Để phục vụ demo bảo vệ đồ án mà không cần khóa API sàn giao dịch thật và không rủi ro tiền thật, một đoạn log mô phỏng (mock) lệnh Binance Futures Testnet đã được thêm vào đúng nơi hệ thống ra quyết định cuối cùng.

**File**: `packages/backend/src/ml/agents/graph_council.py`

**Hàm mới `_log_mock_binance_execution()`** (thêm ngay trước `execution_judge_node`):

```python
def _log_mock_binance_execution(decision: AiCouncilDecision, market: MarketContext) -> None:
    """Log a simulated Binance Futures Testnet order — no real HTTP call is made.

    Every value below is a hardcoded, obviously-fake placeholder, not a real
    credential. This exists to demonstrate where a genuine exchange execution
    call would be wired in, without requiring real exchange API keys or
    risking real funds in this repo.
    """
    side = "BUY" if decision.action == TradeAction.LONG else "SELL"
    mock_payload = {
        "symbol": market.symbol,
        "side": side,
        "type": "MARKET",
        "quantity": decision.position_size_pct,
        "TESTNET_API_KEY": "MOCK-TESTNET-API-KEY-0000000000000000000000000000000000000000",
        "TESTNET_API_SECRET": "MOCK-TESTNET-API-SECRET-0000000000000000000000000000000000000000",
        "HMAC_SHA256_SIGNATURE": "MOCK-SIGNATURE-deadbeefcafebabe0000000000000000000000000000000000000000000000",
    }
    logger.info("[MOCK BINANCE TESTNET EXECUTION] Payload: %s", mock_payload)
```

**Điểm gọi**: trong `execution_judge_node()`, ngay sau dòng `decision = _parse_trade_decision(raw)`:

```python
decision = _parse_trade_decision(raw)

if decision.action in (TradeAction.LONG, TradeAction.SHORT):
    _log_mock_binance_execution(decision, market)
```

**Giải thích an toàn**:
- Đây **chỉ là một dòng log**, không có bất kỳ lời gọi HTTP thật nào đến Binance — không dùng `requests`/`httpx`, không có URL sàn giao dịch nào được gọi.
- `TESTNET_API_KEY`, `TESTNET_API_SECRET`, `HMAC_SHA256_SIGNATURE` đều là chuỗi tĩnh, giả, được viết cứng (hardcoded) trong mã nguồn — không đọc từ biến môi trường hay secret thật nào, nên không có rủi ro rò rỉ khóa thật.
- Log chỉ kích hoạt khi quyết định cuối cùng của Execution Judge là `LONG` hoặc `SHORT` (bỏ qua khi `HOLD`), mô phỏng đúng thời điểm một hệ thống thật sẽ gửi lệnh.
- Mục đích: minh họa rõ ràng điểm tích hợp (integration point) nơi một lệnh gọi Binance Futures Testnet thật sẽ được viết vào, mà không cần cấu hình khóa API sàn giao dịch thật trong đồ án.

---

## Phần 3: Giải phẫu Kiến trúc Hệ thống (Code Comprehension)

### 3.1. AI Council Workflow (Multi-Agent System)

**Điểm vào**: `POST /api/ai/analyze` — `packages/backend/src/routers/ai_council.py`, hàm `ai_analyze()`. Giới hạn `5 requests/giờ` theo IP (`@limiter.limit("5/hour")`), yêu cầu xác thực Cognito.

Luồng xử lý trong `ai_analyze()`:
1. **Lấy dữ liệu nến**: `s3_client.fetch_chart_points(...)` — đọc dữ liệu nến (candle) mới nhất từ Parquet trên S3.
2. **Chạy dự báo Chronos-2**: `inference_service.predict(predict_request)` (xem chi tiết ở mục 3.2).
3. **Lấy dữ liệu thị trường thật** (không mock): `asyncio.gather(fetch_fear_greed(), fetch_funding_rate(symbol))` — từ `src/services/external_data.py`.
4. **Xây dựng `MarketContext`** (Pydantic model, định nghĩa tại `src/ml/agents/crew.py`): gộp forecast + nến + dữ liệu thật ở bước 3.
5. **Chạy debate LangGraph theo dạng streaming**: `run_graph_council_streaming(market_context)` — trả về `StreamingResponse` (Server-Sent Events).

**Đồ thị 4 tác nhân** — `packages/backend/src/ml/agents/graph_council.py`, hàm `build_council_graph()`:

```
quant_analyst → devil_advocate --(route_after_devil)--> [quant_analyst | risk_manager] → execution_judge → END
```

- **Quant Analyst** (`quant_analyst_node`): nhận `market_json` (bao gồm forecast Chronos-2), sinh đề xuất giao dịch (LONG/SHORT/NEUTRAL, entry, stop-loss, take-profit) qua LLM DeepSeek (`_build_chat_llm()` trong `crew.py`).
- **Devil's Advocate** (`devil_advocate_node`, agent định nghĩa qua `_devils_advocate_agent` ngay trong `graph_council.py`): phản biện đề xuất, trả về verdict `strong_contradiction | weak_signal | acceptable`. Đây là debate hai chiều — verdict tốt (risk/reward > 1:2, kỹ thuật mạnh) được APPROVE (kèm khuyến nghị giảm size/thắt stop), không tự động HOLD.
- **Định tuyến** (`route_after_devil`): chỉ verdict `strong_contradiction`/`weak_signal` (và chưa vượt `_MAX_RETRIES = 2`) mới quay lại Quant Analyst để phản biện lại; ngược lại đi tiếp sang Risk Manager.
- **Risk Manager** (`risk_manager_node`): đánh giá sentiment, funding rate, Fear/Greed Index, biến động thực tế; áp giới hạn đòn bẩy nghiêm ngặt theo risk profile (CONSERVATIVE 2-10x, BALANCED 11-40x, DEGEN 41-125x).
- **Execution Judge** (`execution_judge_node`): dùng model reasoner riêng (`_build_reasoner_llm()`), nhận **toàn bộ transcript debate** (`debate_transcript`, gộp mọi vòng Quant Analyst ↔ Devil's Advocate) + đánh giá của Risk Manager, buộc phải trả về đúng cấu trúc JSON `AiCouncilDecision`. Đây cũng là nơi log mock Binance được kích hoạt (xem Phần 2).
- **`_parse_trade_decision()`**: bóc tách JSON từ output thô của LLM (xử lý cả trường hợp bị bọc trong markdown code fence), có fallback về quyết định HOLD an toàn nếu parse thất bại.

### 3.2. Machine Learning Pipeline (Chronos-2)

**Chuẩn bị dữ liệu chuỗi thời gian** — `packages/backend/src/ml/inference.py`, class `ForecastInferenceService`:
- `_build_context_series()`: dựng chuỗi "context" tổng hợp từ nến gốc, kết hợp tín hiệu mẫu hình nến (`_pattern_signal_series` — dùng TA-Lib nếu có, hoặc fallback tự viết bằng NumPy cho doji/engulfing/harami/hammer/...), biến động thực hiện đa khung (`_realized_volatility_series`, cửa sổ 3–160 nến), momentum/EMA, z-score khối lượng, sentiment, và covariate ngoài — giới hạn 512 bước cuối để tối ưu tốc độ inference trên CPU.
- `_build_multivariate_context()`: dựng tensor PyTorch đa biến `(1, 5, T)` cho Chronos-2 — biến 0 = giá đóng cửa (biến duy nhất được dự báo), 1 = z-score log-khối lượng, 2 = tín hiệu mẫu hình nến, 3 = covariate ngoài (funding rate, open interest, Fear/Greed...), 4 = sentiment tổng hợp.

**Nơi inference thực sự chạy**: hàm `predict()` trong cùng file — gọi `_run_hf_quantile_forecast()` (ưu tiên phương thức `predict_quantiles` gốc của model) hoặc fallback `_run_hf_inference()` (thử nhiều chữ ký gọi hàm khác nhau để tương thích nhiều phiên bản API Chronos/HuggingFace).

**Nạp model**: `packages/backend/src/ml/model_loader.py`, hàm `get_loaded_forecasting_model()` — phân giải model đang active từ manifest trên S3 (`_resolve_active_model_uri_from_manifest`, cache 90 giây), tải file model, nạp qua `BaseChronosPipeline` (base `amazon/chronos-2` + LoRA adapter đã fine-tune).

**Hậu xử lý**: `_postprocess_quantile_variance()` — đảm bảo thứ tự quantile hợp lệ (monotonic), áp sàn biến động tối thiểu dựa trên volatility thực tế (Dynamic Minimum Spread) để tránh dải dự báo quá hẹp.

**Kết quả trả về**: `PredictResponse` (`packages/backend/src/ml/schemas.py`) — gồm `predicted_price`, `prediction_array`, `confidence_bands`, `volatility_bands`, `pattern_markers`, các trường sentiment, `trend_direction`, `model_name`/`model_version`, `explanation`.

**Điểm gọi**: `POST /predict` (`packages/backend/src/routers/predict.py`, hàm `predict_price`) cho dự báo độc lập, và được gọi lại bên trong `ai_council.py` (bước 2 ở mục 3.1) để cấp dữ liệu cho AI Council.

### 3.3. Sentiment Analysis

**Vị trí**: `packages/backend/src/data/sentiment.py`, class `SentimentScorer`.

**Hàm chính**: `score_latest()` — được gọi từ `inference.py` (`_estimate_sentiment_score`). Kết hợp:
- **Điểm dựa trên thị trường** (`score_dataframe` → `_headline_keyword_score`): chấm điểm heuristic theo từ khóa (`_POSITIVE_KEYWORDS`/`_NEGATIVE_KEYWORDS`, ví dụ "bullish", "surge", "etf" vs. "bearish", "crash", "hack"), hoặc dùng pipeline FinBERT (`transformers.pipeline`) nếu có sẵn.
- **Điểm bên ngoài** (external): lấy đồng thời qua `httpx.AsyncClient` + `asyncio.gather` — Fear & Greed Index (alternative.me), tin tức crypto (RSS CoinTelegraph/CoinDesk + News API dự phòng), sentiment X/Twitter (endpoint tìm kiếm cấu hình sẵn hoặc RSS/nitter), sentiment địa chính trị (RSS Reuters/BBC World, chấm theo từ khóa sự kiện như "war", "sanction", "recession").

**`get_external_feature_snapshot()`**: trả về `ExternalSentimentSnapshot` (score, source, fear_greed_index, crypto_news_sentiment, x_sentiment_score, geopolitical_sentiment, event_impact_score) — dùng cả cho điểm sentiment tổng hợp lẫn làm covariate đầu vào cho `_build_multivariate_context` ở mục 3.2.

**Xử lý đồng bộ/bất đồng bộ**: `_compute_external_snapshot()` chạy tác vụ async bên trong `concurrent.futures.ThreadPoolExecutor` riêng, tránh xung đột với event loop đang chạy sẵn của FastAPI khi được gọi từ route bất đồng bộ.

**Cache**: kết quả sentiment ngoài được cache theo TTL (cấu hình qua `external_refresh_seconds`) để tránh gọi lại toàn bộ nguồn dữ liệu ngoài ở mỗi request dự báo.
