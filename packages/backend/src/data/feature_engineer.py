from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import talib
except Exception:  # pragma: no cover
    talib = None


def _fallback_patterns(dataframe: pd.DataFrame) -> pd.DataFrame:
    opens = dataframe["open"].to_numpy(dtype=np.float64)
    highs = dataframe["high"].to_numpy(dtype=np.float64)
    lows = dataframe["low"].to_numpy(dtype=np.float64)
    closes = dataframe["close"].to_numpy(dtype=np.float64)

    candle_range = np.maximum(highs - lows, 1e-8)
    body = np.abs(closes - opens)
    upper_shadow = highs - np.maximum(opens, closes)
    lower_shadow = np.minimum(opens, closes) - lows

    dataframe["pattern_doji"] = np.where(body / candle_range < 0.1, 100, 0).astype(np.int32)

    bullish_hammer = (lower_shadow / candle_range > 0.55) & (upper_shadow / candle_range < 0.2)
    dataframe["pattern_hammer"] = np.where(bullish_hammer, np.where(closes >= opens, 100, -100), 0).astype(np.int32)

    bearish_shooting_star = (upper_shadow / candle_range > 0.55) & (lower_shadow / candle_range < 0.2)
    dataframe["pattern_shooting_star"] = np.where(bearish_shooting_star, -100, 0).astype(np.int32)

    prev_open = np.roll(opens, 1)
    prev_close = np.roll(closes, 1)
    bullish_engulf = (
        (closes > opens)
        & (prev_close < prev_open)
        & (closes >= prev_open)
        & (opens <= prev_close)
    )
    bearish_engulf = (
        (closes < opens)
        & (prev_close > prev_open)
        & (opens >= prev_close)
        & (closes <= prev_open)
    )
    engulfing = np.where(bullish_engulf, 100, np.where(bearish_engulf, -100, 0)).astype(np.int32)
    engulfing[0] = 0
    dataframe["pattern_engulfing"] = engulfing

    morning_star = np.zeros_like(closes, dtype=np.int32)
    evening_star = np.zeros_like(closes, dtype=np.int32)
    hanging_man = np.zeros_like(closes, dtype=np.int32)
    for idx in range(2, closes.size):
        first_bearish = closes[idx - 2] < opens[idx - 2]
        first_bullish = closes[idx - 2] > opens[idx - 2]
        small_middle = abs(closes[idx - 1] - opens[idx - 1]) <= 0.35 * (highs[idx - 1] - lows[idx - 1] + 1e-8)
        strong_bull = closes[idx] > opens[idx] and closes[idx] >= (opens[idx - 2] + closes[idx - 2]) / 2.0
        strong_bear = closes[idx] < opens[idx] and closes[idx] <= (opens[idx - 2] + closes[idx - 2]) / 2.0

        if first_bearish and small_middle and strong_bull:
            morning_star[idx] = 100
        if first_bullish and small_middle and strong_bear:
            evening_star[idx] = -100

        if idx >= 3:
            uptrend = closes[idx - 1] > closes[idx - 3]
            if uptrend and (lower_shadow[idx] / candle_range[idx] > 0.55) and (upper_shadow[idx] / candle_range[idx] < 0.2):
                hanging_man[idx] = -100

    dataframe["pattern_morning_star"] = morning_star
    dataframe["pattern_evening_star"] = evening_star
    dataframe["pattern_hanging_man"] = hanging_man
    return dataframe


def engineer_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    if dataframe.empty:
        return dataframe

    required_columns = {
        "symbol",
        "timeframe",
        "timestamp",
        "open_time_ms",
        "close_time_ms",
        "open",
        "high",
        "low",
        "close",
        "volume",
    }
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns for feature engineering: {sorted(missing)}")

    output = dataframe.copy().sort_values("timestamp").reset_index(drop=True)

    output["return_1"] = output["close"].pct_change().fillna(0.0)
    output["log_return_1"] = np.log1p(output["return_1"].clip(lower=-0.9999))
    output["volatility_5"] = output["return_1"].rolling(window=5, min_periods=2).std().fillna(0.0)
    output["volatility_10"] = output["return_1"].rolling(window=10, min_periods=2).std().fillna(0.0)
    output["volatility_20"] = output["return_1"].rolling(window=20, min_periods=2).std().fillna(0.0)
    output["volatility_30"] = output["return_1"].rolling(window=30, min_periods=2).std().fillna(0.0)
    output["volatility_60"] = output["return_1"].rolling(window=60, min_periods=2).std().fillna(0.0)
    output["realized_volatility_10"] = output["log_return_1"].rolling(window=10, min_periods=2).std().fillna(0.0)
    output["realized_volatility_30"] = output["log_return_1"].rolling(window=30, min_periods=2).std().fillna(0.0)
    output["realized_volatility_60"] = output["log_return_1"].rolling(window=60, min_periods=2).std().fillna(0.0)
    output["hl_spread"] = ((output["high"] - output["low"]) / output["close"].clip(lower=1e-8)).fillna(0.0)

    opens = output["open"].to_numpy(dtype=np.float64)
    highs = output["high"].to_numpy(dtype=np.float64)
    lows = output["low"].to_numpy(dtype=np.float64)
    closes = output["close"].to_numpy(dtype=np.float64)

    if talib is not None:
        try:
            output["pattern_doji"] = talib.CDLDOJI(opens, highs, lows, closes)
            output["pattern_engulfing"] = talib.CDLENGULFING(opens, highs, lows, closes)
            output["pattern_hammer"] = talib.CDLHAMMER(opens, highs, lows, closes)
            output["pattern_shooting_star"] = talib.CDLSHOOTINGSTAR(opens, highs, lows, closes)
            output["pattern_morning_star"] = talib.CDLMORNINGSTAR(opens, highs, lows, closes)
            output["pattern_evening_star"] = talib.CDLEVENINGSTAR(opens, highs, lows, closes)
            output["pattern_hanging_man"] = talib.CDLHANGINGMAN(opens, highs, lows, closes)
        except Exception as exc:
            logger.warning("TA-Lib pattern extraction failed, using fallback heuristics: %s", exc)
            output = _fallback_patterns(output)
    else:
        output = _fallback_patterns(output)

    output.replace([np.inf, -np.inf], 0.0, inplace=True)
    output.fillna(0.0, inplace=True)
    return output
