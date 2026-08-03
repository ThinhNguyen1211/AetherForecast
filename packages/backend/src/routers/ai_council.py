"""POST /api/ai/analyze — LangGraph cyclic AI Council debate (SSE streaming).

Reuses the ML forecast + sentiment the client already computed via a prior
POST /predict call (the UI requires a prediction before AI Council is
reachable, so re-running Chronos-2 and re-fetching S3 candle history here
would be pure duplicate work). Only genuinely time-sensitive data is fetched
fresh: the current market price (real-time Binance klines) and funding
rate/Fear-Greed Index (not present on PredictResponse). Runs the 4-agent
LangGraph debate pipeline:
    Quant Analyst → Devil's Advocate → Risk Manager → Execution Judge.
The Devil's Advocate can force the Quant Analyst to re-evaluate if severe
contradictions are found (capped to prevent infinite loops).
Streams agent thoughts and debate logs in real-time via Server-Sent Events.
Final event contains [FINAL_RESULT]:<AiCouncilDecision JSON>.
Rate limited to 20 requests per hour per IP.
"""

import asyncio
import logging
import traceback
from collections.abc import Generator
from typing import Literal

import numpy as np
from fastapi import APIRouter, Body, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from slowapi import Limiter
from slowapi.util import get_remote_address

from src.dependencies.cognito import require_authenticated_user
from src.ml.agents.crew import (
    MarketContext,
    RiskProfile,
)
from src.ml.agents.graph_council import run_graph_council_streaming
from src.ml.schemas import PredictResponse
from src.services.external_data import fetch_fear_greed, fetch_funding_rate, fetch_latest_klines

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ai", tags=["ai-council"])
limiter = Limiter(key_func=get_remote_address)

_REALTIME_KLINES_LIMIT = 50


def _sse_error(event: str) -> str:
    """Format an SSE error line (single data: line, no embedded newlines)."""
    return f"data: {event.replace(chr(10), ' | ')}\n\n"


def _error_stream(message: str, exc: Exception | None = None) -> Generator[str, None, None]:
    """Yield a CORS-safe SSE error stream for pre-flight failures."""
    yield _sse_error(f"[ERROR]:{message}")
    if exc is not None:
        error_tb = traceback.format_exc().replace("\n", " | ")
        yield _sse_error(f"[TRACE]:{error_tb}")


class AiAnalyzeRequest(BaseModel):
    """POST /api/ai/analyze request body.

    `prediction` is the PredictResponse the client already received from a
    prior POST /predict call — the AI Council reuses its forecast and
    sentiment instead of recomputing them.
    """

    symbol: str = Field(description="Trading pair e.g. BTCUSDT")
    timeframe: str = Field(default="1h")
    risk_profile: RiskProfile = Field(
        default=RiskProfile.BALANCED,
        description="Risk profile: CONSERVATIVE, BALANCED, or DEGEN",
    )
    language: Literal["en", "vi"] = Field(
        default="vi",
        description="Language for the final reasoning field: en (English) or vi (Vietnamese)",
    )
    prediction: PredictResponse = Field(
        description="The PredictResponse from a prior POST /predict call for this symbol/timeframe"
    )


# Force Pydantic V2 to fully resolve this model before FastAPI builds the
# route TypeAdapter. Prevents the ForwardRef crash on startup.
AiAnalyzeRequest.model_rebuild()


@router.post("/analyze")
@limiter.limit("20/hour")
async def ai_analyze(
    request: Request,
    payload: AiAnalyzeRequest = Body(...),
    _claims: dict = Depends(require_authenticated_user),
) -> StreamingResponse:
    """Run the LangGraph 4-agent cyclic council on the current market state.

    Reuses payload.prediction instead of re-running Chronos-2 inference or
    re-fetching S3 candle history — only the current price (real-time
    Binance klines) and funding rate/Fear-Greed Index are fetched fresh,
    since those aren't part of a PredictResponse. Returns a Server-Sent
    Events stream with agent reasoning, Devil's Advocate debate logs, and the
    final decision in real-time. Any exception before or during streaming is
    converted into an SSE error event so the browser receives CORS headers
    and the real error message instead of a bare 500.
    """
    symbol = payload.symbol.upper().strip()
    timeframe = payload.timeframe or "1h"
    prediction = payload.prediction

    if prediction.symbol.upper().strip() != symbol or prediction.timeframe != timeframe:
        return StreamingResponse(
            _error_stream(
                f"Prediction context ({prediction.symbol}/{prediction.timeframe}) does not match "
                f"the requested analysis ({symbol}/{timeframe}) — regenerate the prediction and retry."
            ),
            media_type="text/event-stream",
        )

    try:
        # --- Step 1: Fetch the current market price directly from Binance (real-time, never mocked) ---
        try:
            recent_klines = await fetch_latest_klines(symbol, timeframe, limit=_REALTIME_KLINES_LIMIT)
        except Exception as exc:
            logger.warning("Failed to fetch real-time klines for AI analysis: %s", exc)
            recent_klines = []

        if len(recent_klines) < 5:
            return StreamingResponse(
                _error_stream(f"Cannot fetch real-time market data for {symbol}"),
                media_type="text/event-stream",
            )

        # --- Step 2: Fetch real external market data not covered by the prediction (never mocked) ---
        fear_greed_index, funding_rate = await asyncio.gather(
            fetch_fear_greed(),
            fetch_funding_rate(symbol),
        )

        # --- Step 3: Build MarketContext from real-time price + the client-supplied ML forecast/sentiment ---
        closes = np.array([c["close"] for c in recent_klines], dtype=np.float64)
        log_returns = np.diff(np.log(np.maximum(closes, 1e-8)))
        realized_vol = float(np.std(log_returns)) if len(log_returns) > 2 else 0.01

        current_price = float(recent_klines[-1]["close"])

        market_context = MarketContext(
            symbol=symbol,
            current_price=current_price,
            forecast_median=prediction.predicted_price,
            forecast_lower=prediction.confidence_interval.lower,
            forecast_upper=prediction.confidence_interval.upper,
            realized_volatility=realized_vol,
            sentiment_score=prediction.sentiment_score,
            funding_rate=funding_rate,
            fear_greed_index=fear_greed_index,
            timeframe=timeframe,
            risk_profile=payload.risk_profile,
            language=payload.language,
        )

        # --- Step 4: Stream LangGraph debate council via SSE ---
        logger.info(
            "Starting LangGraph AI council debate for %s @ %s (reusing client-supplied prediction)",
            symbol,
            timeframe,
        )
        return StreamingResponse(
            run_graph_council_streaming(market_context),
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
