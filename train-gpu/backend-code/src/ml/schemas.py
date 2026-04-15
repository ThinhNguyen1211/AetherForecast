from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


SupportedTimeframe = Literal["1m", "5m", "15m", "1h", "4h", "1d", "1w"]


class Candle(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class PredictRequest(BaseModel):
    symbol: str = Field(min_length=2, max_length=20)
    timeframe: SupportedTimeframe = "1h"
    latest_candles: list[Candle] = Field(min_length=10, max_length=5000)
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    horizon: int = Field(default=24, ge=1, le=336)
    quantiles: list[float] = Field(default_factory=lambda: [0.1, 0.5, 0.9], min_length=3)


class ConfidenceInterval(BaseModel):
    lower: float
    upper: float


class ConfidenceBand(BaseModel):
    quantile: float = Field(ge=0.0, le=1.0)
    values: list[float] = Field(min_length=1)


class VolatilityBand(BaseModel):
    label: str
    lower: list[float] = Field(min_length=1)
    upper: list[float] = Field(min_length=1)


class PatternMarker(BaseModel):
    timestamp: datetime
    pattern: str
    direction: Literal["bullish", "bearish", "neutral"]
    strength: float = Field(ge=-1.0, le=1.0)


class PredictResponse(BaseModel):
    symbol: str
    timeframe: SupportedTimeframe
    horizon: int
    predicted_price: float
    prediction_array: list[float] = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)
    confidence_interval: ConfidenceInterval
    confidence_bands: list[ConfidenceBand] = Field(min_length=1)
    volatility_bands: list[VolatilityBand] = Field(default_factory=list)
    pattern_markers: list[PatternMarker] = Field(default_factory=list)
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    sentiment_source: str
    external_sentiment_score: float = Field(ge=-1.0, le=1.0)
    external_sentiment_source: str
    trend_direction: Literal["up", "down", "flat"]
    model_name: str
    model_version: str
    explanation: str
    generated_at: datetime


class ChartResponse(BaseModel):
    symbol: str
    timeframe: SupportedTimeframe
    candles: list[Candle]
