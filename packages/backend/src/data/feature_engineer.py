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
    lower_shadow = np.minimum(opens, closes) - lows

    dataframe["pattern_doji"] = (body / candle_range < 0.1).astype(np.int32)
    dataframe["pattern_engulfing"] = np.sign(closes - opens).astype(np.int32)
    dataframe["pattern_hammer"] = (lower_shadow / candle_range > 0.55).astype(np.int32)
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
    output["volatility_10"] = output["return_1"].rolling(window=10, min_periods=2).std().fillna(0.0)
    output["volatility_30"] = output["return_1"].rolling(window=30, min_periods=2).std().fillna(0.0)
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
        except Exception as exc:
            logger.warning("TA-Lib pattern extraction failed, using fallback heuristics: %s", exc)
            output = _fallback_patterns(output)
    else:
        output = _fallback_patterns(output)

    output.replace([np.inf, -np.inf], 0.0, inplace=True)
    output.fillna(0.0, inplace=True)
    return output
