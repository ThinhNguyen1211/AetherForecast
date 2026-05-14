"""POST /api/ai/analyze — CrewAI multi-agent trading analysis endpoint.

Pulls latest candles + Chronos-2 forecast + sentiment for the given symbol,
then runs the 3-agent CrewAI pipeline (Quant → Risk → Judge).
Returns a TradeDecision with Action, Entry, Leverage, SL, TP, Reasoning.
"""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, status

from src.dependencies.cognito import require_authenticated_user
from src.dependencies.s3_client import S3ParquetClient, get_s3_parquet_client
from src.ml.agents.crew import (
    AiAnalyzeRequest,
    MarketContext,
    TradeDecision,
    run_trading_crew,
)
from src.ml.inference import ForecastInferenceService, get_inference_service
from src.ml.schemas import Candle, PredictRequest

router = APIRouter(prefix="/api/ai", tags=["ai-council"])
logger = logging.getLogger(__name__)

_HISTORY_LIMIT = 1500
_FORECAST_HORIZON = 24


@router.post("/analyze", response_model=TradeDecision)
def ai_analyze(
    body: AiAnalyzeRequest,
    _claims: dict = Depends(require_authenticated_user),
    inference_service: ForecastInferenceService = Depends(get_inference_service),
    s3_client: S3ParquetClient = Depends(get_s3_parquet_client),
) -> TradeDecision:
    """Run CrewAI 3-agent council on the current market state for a symbol."""
    symbol = body.symbol.upper().strip()
    timeframe = body.timeframe or "1h"

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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Cannot fetch market data for {symbol}",
        ) from exc

    if len(raw_candles) < 30:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Insufficient candle data for {symbol} ({len(raw_candles)} candles)",
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
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Forecast model unavailable",
        ) from exc

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

    # --- Step 4: Run CrewAI pipeline ---
    try:
        decision = run_trading_crew(market_context)
        logger.info(
            "AI Council decision for %s: action=%s confidence=%.2f",
            symbol,
            decision.action,
            decision.confidence,
        )
        return decision
    except RuntimeError as exc:
        # CrewAI not installed or GEMINI_API_KEY missing
        logger.warning("CrewAI runtime error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error in AI analysis pipeline")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI analysis pipeline failed unexpectedly",
        ) from exc
