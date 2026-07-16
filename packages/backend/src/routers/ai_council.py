"""POST /api/ai/analyze — CrewAI multi-agent trading analysis (SSE streaming).

Pulls latest candles + Chronos-2 forecast + sentiment for the given symbol,
then runs the 3-agent CrewAI pipeline (Quant → Risk → Judge).
Streams agent thoughts in real-time via Server-Sent Events.
Final event contains [FINAL_RESULT]:<TradeDecision JSON>.
Rate limited to 5 requests per hour per IP.
"""

from __future__ import annotations

import logging
import traceback
from collections.abc import Generator

import numpy as np
from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.dependencies.cognito import require_authenticated_user
from src.dependencies.s3_client import S3ParquetClient, get_s3_parquet_client
from src.ml.agents.crew import (
    AiAnalyzeRequest,
    MarketContext,
    run_trading_crew_streaming,
)
from src.ml.inference import ForecastInferenceService, get_inference_service
from src.ml.schemas import Candle, PredictRequest

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai-council"])
limiter = Limiter(key_func=get_remote_address)

_HISTORY_LIMIT = 1500
_FORECAST_HORIZON = 24


def _sse_error(event: str) -> str:
    """Format an SSE error line (single data: line, no embedded newlines)."""
    return f"data: {event.replace(chr(10), ' | ')}\n\n"


def _error_stream(message: str, exc: Exception | None = None) -> Generator[str, None, None]:
    """Yield a CORS-safe SSE error stream for pre-flight failures."""
    yield _sse_error(f"[ERROR]:{message}")
    if exc is not None:
        error_tb = traceback.format_exc().replace("\n", " | ")
        yield _sse_error(f"[TRACE]:{error_tb}")


@router.post("/analyze")
@limiter.limit("5/hour")
async def ai_analyze(
    request: Request,
    payload: AiAnalyzeRequest = Body(...),
    _claims: dict = Depends(require_authenticated_user),
    inference_service: ForecastInferenceService = Depends(get_inference_service),
    s3_client: S3ParquetClient = Depends(get_s3_parquet_client),
) -> StreamingResponse:
    """Run CrewAI 3-agent council on the current market state for a symbol.

    Returns Server-Sent Events stream with agent reasoning in real-time.
    Any exception before or during streaming is converted into an SSE error
    event so the browser receives CORS headers and the real error message
    instead of a bare 500 response.
    """
    symbol = payload.symbol.upper().strip()
    timeframe = payload.timeframe or "1h"

    try:
        # --- Step 1: Pull latest candles from backend data sources ---
        try:
            raw_candles = s3_client.fetch_chart_points(
                symbol=symbol,
                timeframe=timeframe,
                limit=_HISTORY_LIMIT,
                use_cache=False,
            )
        except Exception as exc:
            logger.warning("Failed to fetch candles for AI analysis: %s", exc)
            return StreamingResponse(
                _error_stream(f"Cannot fetch market data for {symbol}"),
                media_type="text/event-stream",
            )

        if len(raw_candles) < 30:
            return StreamingResponse(
                _error_stream(
                    f"Insufficient candle data for {symbol} ({len(raw_candles)} candles)"
                ),
                media_type="text/event-stream",
            )

        # --- Step 2: Run Chronos-2 forecast ---
        try:
            predict_request = PredictRequest(
                symbol=symbol,
                timeframe=timeframe,
                latest_candles=[Candle(**item) for item in raw_candles],
                horizon=_FORECAST_HORIZON,
                quantiles=[0.05, 0.5, 0.95],
                sentiment_score=None,
            )
            forecast = inference_service.predict(predict_request)
        except Exception as exc:
            logger.warning("Chronos-2 forecast failed for AI analysis: %s", exc)
            return StreamingResponse(
                _error_stream("Forecast model unavailable", exc),
                media_type="text/event-stream",
            )

        # --- Step 3: Build MarketContext from forecast + candles ---
        closes = np.array([c["close"] for c in raw_candles[-50:]], dtype=np.float64)
        log_returns = np.diff(np.log(np.maximum(closes, 1e-8)))
        realized_vol = float(np.std(log_returns)) if len(log_returns) > 2 else 0.01

        current_price = float(raw_candles[-1]["close"])
        forecast_median = forecast.predicted_price
        forecast_lower = forecast.confidence_interval.lower
        forecast_upper = forecast.confidence_interval.upper

        market_context = MarketContext(
            symbol=symbol,
            current_price=current_price,
            forecast_median=forecast_median,
            forecast_lower=forecast_lower,
            forecast_upper=forecast_upper,
            realized_volatility=realized_vol,
            sentiment_score=forecast.sentiment_score,
            fear_greed_index=50.0,  # will be enriched by sentiment scorer if available
            timeframe=timeframe,
        )

        # --- Step 4: Stream CrewAI pipeline via SSE ---
        logger.info("Starting SSE streaming AI analysis for %s @ %s", symbol, timeframe)
        return StreamingResponse(
            run_trading_crew_streaming(market_context),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    except Exception as exc:
        logger.exception("Unexpected error in /api/ai/analyze")
        return StreamingResponse(
            _error_stream(
                f"SSE Stream Error - {type(exc).__name__}: {exc}", exc
            ),
            media_type="text/event-stream",
        )
