from __future__ import annotations

from functools import lru_cache
import logging
import time
from typing import Any

import httpx
import numpy as np
import pandas as pd

from src.core.config import Settings

logger = logging.getLogger(__name__)

COVARIATE_COLUMNS = [
    "fear_greed_index",
    "crypto_news_sentiment",
    "x_sentiment_score",
    "funding_rate",
    "open_interest",
    "long_short_ratio",
    "top_trader_long_short_ratio",
    "taker_buy_sell_ratio",
    "btc_dominance",
    "macro_dxy",
    "macro_us10y",
    "event_impact_score",
]

COVARIATE_WEIGHTS = {
    "funding_rate": 0.16,
    "open_interest": 0.14,
    "long_short_ratio": 0.12,
    "top_trader_long_short_ratio": 0.10,
    "taker_buy_sell_ratio": 0.08,
    "btc_dominance": 0.08,
    "fear_greed_index": 0.08,
    "crypto_news_sentiment": 0.08,
    "x_sentiment_score": 0.08,
    "macro_dxy": -0.06,
    "macro_us10y": -0.04,
    "event_impact_score": 0.12,
}

BINANCE_ALLOWED_PERIODS = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}
TIMEFRAME_TO_BINANCE_PERIOD = {
    "1m": "5m",
    "3m": "5m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "6h": "6h",
    "12h": "12h",
    "1d": "1d",
    "1w": "1d",
}


def _cache_bucket(refresh_seconds: int) -> int:
    return int(time.time() // max(60, int(refresh_seconds)))


def _fetch_json(
    url: str,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 8.0,
) -> dict | list | None:
    if not url:
        return None

    try:
        with httpx.Client(timeout=httpx.Timeout(timeout, connect=4.0)) as client:
            response = client.get(url, params=params, headers=headers)
            response.raise_for_status()
            return response.json()
    except Exception as exc:
        logger.debug("Unable to fetch %s: %s", url, exc)
        return None


@lru_cache(maxsize=2)
def _load_fear_greed_series(cache_bucket: int) -> pd.DataFrame:
    _ = cache_bucket
    payload = _fetch_json("https://api.alternative.me/fng/?limit=365&format=json")
    if not isinstance(payload, dict):
        return pd.DataFrame(columns=["date", "fear_greed_index"])

    rows = payload.get("data")
    if not isinstance(rows, list):
        return pd.DataFrame(columns=["date", "fear_greed_index"])

    records: list[dict[str, float | pd.Timestamp]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        timestamp_raw = row.get("timestamp")
        value_raw = row.get("value")
        try:
            timestamp_value = int(timestamp_raw)
            score = float(value_raw)
        except (TypeError, ValueError):
            continue

        records.append(
            {
                "date": pd.to_datetime(timestamp_value, unit="s", utc=True).floor("D"),
                "fear_greed_index": score,
            }
        )

    if not records:
        return pd.DataFrame(columns=["date", "fear_greed_index"])

    return pd.DataFrame.from_records(records).drop_duplicates(subset=["date"], keep="last")


@lru_cache(maxsize=2)
def _load_btc_dominance_value(cache_bucket: int) -> float | None:
    _ = cache_bucket
    payload = _fetch_json("https://api.coingecko.com/api/v3/global")
    if not isinstance(payload, dict):
        return None

    data = payload.get("data")
    if not isinstance(data, dict):
        return None

    market_cap_percentage = data.get("market_cap_percentage")
    if not isinstance(market_cap_percentage, dict):
        return None

    btc_raw = market_cap_percentage.get("btc")
    try:
        return float(btc_raw)
    except (TypeError, ValueError):
        return None


@lru_cache(maxsize=4)
def _load_fred_series(series_id: str, output_column: str, cache_bucket: int) -> pd.DataFrame:
    _ = cache_bucket
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        frame = pd.read_csv(url)
    except Exception as exc:  # pragma: no cover
        logger.debug("Unable to fetch FRED %s: %s", series_id, exc)
        return pd.DataFrame(columns=["date", output_column])

    if frame.empty or len(frame.columns) < 2:
        return pd.DataFrame(columns=["date", output_column])

    date_column = frame.columns[0]
    value_column = frame.columns[1]
    prepared = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_column], utc=True, errors="coerce").dt.floor("D"),
            output_column: pd.to_numeric(frame[value_column], errors="coerce"),
        }
    )
    prepared = prepared.dropna(subset=["date"]).sort_values("date")
    prepared[output_column] = prepared[output_column].ffill()
    return prepared.drop_duplicates(subset=["date"], keep="last")


@lru_cache(maxsize=64)
def _load_binance_covariates(symbol: str, timeframe: str, cache_bucket: int) -> pd.DataFrame:
    _ = cache_bucket
    normalized_symbol = symbol.strip().upper()
    normalized_timeframe = timeframe.strip().lower() if timeframe else "1h"
    period = TIMEFRAME_TO_BINANCE_PERIOD.get(normalized_timeframe, "1h")
    if period not in BINANCE_ALLOWED_PERIODS:
        period = "1h"

    frames: list[pd.DataFrame] = []

    funding_payload = _fetch_json(
        f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={normalized_symbol}&limit=1000"
    )
    if isinstance(funding_payload, list):
        funding_records = []
        for row in funding_payload:
            if not isinstance(row, dict):
                continue
            try:
                ts_ms = int(row.get("fundingTime", 0))
                value = float(row.get("fundingRate", 0.0))
            except (TypeError, ValueError):
                continue
            funding_records.append(
                {
                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                    "funding_rate": value,
                }
            )
        if funding_records:
            frames.append(pd.DataFrame.from_records(funding_records))

    open_interest_payload = _fetch_json(
        "https://fapi.binance.com/futures/data/openInterestHist"
        f"?symbol={normalized_symbol}&period={period}&limit=500"
    )
    if isinstance(open_interest_payload, list):
        oi_records = []
        for row in open_interest_payload:
            if not isinstance(row, dict):
                continue
            try:
                ts_ms = int(row.get("timestamp", 0))
                value = float(row.get("sumOpenInterest", 0.0))
            except (TypeError, ValueError):
                continue
            oi_records.append(
                {
                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                    "open_interest": value,
                }
            )
        if oi_records:
            frames.append(pd.DataFrame.from_records(oi_records))

    long_short_payload = _fetch_json(
        "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
        f"?symbol={normalized_symbol}&period={period}&limit=500"
    )
    if isinstance(long_short_payload, list):
        ratio_records = []
        for row in long_short_payload:
            if not isinstance(row, dict):
                continue
            try:
                ts_ms = int(row.get("timestamp", 0))
                value = float(row.get("longShortRatio", 0.0))
            except (TypeError, ValueError):
                continue
            ratio_records.append(
                {
                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                    "long_short_ratio": value,
                }
            )
        if ratio_records:
            frames.append(pd.DataFrame.from_records(ratio_records))

    # Top Trader Long/Short Ratio
    top_ls_payload = _fetch_json(
        "https://fapi.binance.com/futures/data/topLongShortAccountRatio"
        f"?symbol={normalized_symbol}&period={period}&limit=500"
    )
    if isinstance(top_ls_payload, list):
        top_ls_records = []
        for row in top_ls_payload:
            if not isinstance(row, dict):
                continue
            try:
                ts_ms = int(row.get("timestamp", 0))
                value = float(row.get("longShortRatio", 0.0))
            except (TypeError, ValueError):
                continue
            top_ls_records.append(
                {
                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                    "top_trader_long_short_ratio": value,
                }
            )
        if top_ls_records:
            frames.append(pd.DataFrame.from_records(top_ls_records))

    # Taker Buy/Sell Volume ratio
    taker_payload = _fetch_json(
        "https://fapi.binance.com/futures/data/takerlongshortRatio"
        f"?symbol={normalized_symbol}&period={period}&limit=500"
    )
    if isinstance(taker_payload, list):
        taker_records = []
        for row in taker_payload:
            if not isinstance(row, dict):
                continue
            try:
                ts_ms = int(row.get("timestamp", 0))
                buy_vol = float(row.get("buyVol", 0.0))
                sell_vol = float(row.get("sellVol", 0.0))
                ratio = buy_vol / max(sell_vol, 1e-8)
            except (TypeError, ValueError):
                continue
            taker_records.append(
                {
                    "timestamp": pd.to_datetime(ts_ms, unit="ms", utc=True),
                    "taker_buy_sell_ratio": ratio,
                }
            )
        if taker_records:
            frames.append(pd.DataFrame.from_records(taker_records))

    if not frames:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "open_interest", "long_short_ratio", "top_trader_long_short_ratio", "taker_buy_sell_ratio"])

    merged = frames[0].sort_values("timestamp")
    for frame in frames[1:]:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            frame.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )

    merged["symbol"] = normalized_symbol
    merged["timeframe"] = period
    return merged.drop_duplicates(subset=["timestamp"], keep="last")


def _merge_daily_feature(
    dataframe: pd.DataFrame,
    daily_feature_frame: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    if daily_feature_frame.empty:
        return dataframe

    prepared = dataframe.copy()
    prepared["_date"] = prepared["timestamp"].dt.floor("D")

    daily = daily_feature_frame.copy()
    if "date" not in daily.columns or target_column not in daily.columns:
        return dataframe

    daily = daily[["date", target_column]].rename(columns={"date": "_date", target_column: "_external_value"})
    merged = prepared.merge(daily, on="_date", how="left")

    if target_column not in merged.columns:
        merged[target_column] = np.nan
    merged[target_column] = pd.to_numeric(merged[target_column], errors="coerce").fillna(
        pd.to_numeric(merged["_external_value"], errors="coerce")
    )
    merged = merged.drop(columns=["_date", "_external_value"])
    return merged


def _merge_asof_feature(
    dataframe: pd.DataFrame,
    feature_frame: pd.DataFrame,
    target_column: str,
) -> pd.DataFrame:
    if feature_frame.empty or target_column not in feature_frame.columns:
        return dataframe

    left = dataframe.copy().reset_index(drop=True)
    left["_row_id"] = np.arange(len(left))
    right = feature_frame.copy()

    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True, errors="coerce")
    right["timestamp"] = pd.to_datetime(right["timestamp"], utc=True, errors="coerce")
    left["timestamp"] = left["timestamp"].astype("datetime64[ns, UTC]")
    right["timestamp"] = right["timestamp"].astype("datetime64[ns, UTC]")

    left = left.dropna(subset=["timestamp"])
    right = right.dropna(subset=["timestamp"])
    if left.empty or right.empty:
        return dataframe

    if "symbol" in right.columns:
        right["symbol"] = right["symbol"].astype(str).str.strip().str.upper()
    if "timeframe" in right.columns:
        right["timeframe"] = right["timeframe"].astype(str).str.strip().str.lower()

    if "symbol" in left.columns:
        left["symbol"] = left["symbol"].astype(str).str.strip().str.upper()
    if "timeframe" in left.columns:
        left["timeframe"] = left["timeframe"].astype(str).str.strip().str.lower()

    by_columns = [column for column in ("symbol", "timeframe") if column in left.columns and column in right.columns]
    right_columns = by_columns + ["timestamp", target_column]
    right_prepared = right[right_columns].copy().rename(columns={target_column: "_external_value"})

    sort_keys = by_columns + ["timestamp"] if by_columns else ["timestamp"]
    left_sorted = left.sort_values(sort_keys)
    right_sorted = right_prepared.sort_values(sort_keys)

    if by_columns:
        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            on="timestamp",
            by=by_columns,
            direction="backward",
        )
    else:
        merged = pd.merge_asof(
            left_sorted,
            right_sorted,
            on="timestamp",
            direction="backward",
        )

    if target_column not in merged.columns:
        merged[target_column] = np.nan
    merged[target_column] = pd.to_numeric(merged[target_column], errors="coerce").fillna(
        pd.to_numeric(merged["_external_value"], errors="coerce")
    )

    merged = merged.sort_values("_row_id").drop(columns=["_row_id", "_external_value"])
    return merged.reset_index(drop=True)


def _rolling_zscore(series: pd.Series, window: int = 64) -> pd.Series:
    min_periods = max(6, window // 4)
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


def build_external_covariate_signal(
    candles: list[Any],
    symbol: str,
    timeframe: str,
    sentiment_snapshot: dict[str, float] | None,
    settings: Settings,
) -> tuple[np.ndarray, dict[str, float]]:
    if not settings.external_covariates_enabled:
        return np.asarray([], dtype=np.float64), {}

    if not candles:
        return np.asarray([], dtype=np.float64), {}

    rows = []
    for candle in candles:
        timestamp = getattr(candle, "timestamp", None)
        rows.append({"timestamp": timestamp, "symbol": symbol, "timeframe": timeframe})

    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        return np.asarray([], dtype=np.float64), {}

    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True, errors="coerce")
    frame = frame.dropna(subset=["timestamp"]).reset_index(drop=True)
    if frame.empty:
        return np.asarray([], dtype=np.float64), {}

    for column in COVARIATE_COLUMNS:
        frame[column] = np.nan

    snapshot = sentiment_snapshot or {}
    for column in ("fear_greed_index", "crypto_news_sentiment", "x_sentiment_score", "event_impact_score"):
        value = snapshot.get(column)
        if value is not None:
            frame[column] = float(value)

    cache_bucket = _cache_bucket(settings.external_covariates_refresh_seconds)

    if frame["fear_greed_index"].isna().all():
        frame = _merge_daily_feature(frame, _load_fear_greed_series(cache_bucket), "fear_greed_index")

    frame = _merge_daily_feature(frame, _load_fred_series("DTWEXBGS", "macro_dxy", cache_bucket), "macro_dxy")
    frame = _merge_daily_feature(frame, _load_fred_series("DGS10", "macro_us10y", cache_bucket), "macro_us10y")

    btc_dominance = _load_btc_dominance_value(cache_bucket)
    if btc_dominance is not None:
        frame["btc_dominance"] = pd.to_numeric(frame["btc_dominance"], errors="coerce").fillna(btc_dominance)

    covariate_frame = _load_binance_covariates(symbol, timeframe, cache_bucket)
    for column in ("funding_rate", "open_interest", "long_short_ratio", "top_trader_long_short_ratio", "taker_buy_sell_ratio"):
        frame = _merge_asof_feature(frame, covariate_frame, column)

    for column in COVARIATE_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    combined = np.zeros(len(frame), dtype=np.float64)
    for column, weight in COVARIATE_WEIGHTS.items():
        z_values = _rolling_zscore(frame[column].astype("float64"), window=64)
        z_values = z_values.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        combined += z_values.to_numpy(dtype=np.float64) * weight

    covariate_signal = np.tanh(combined)
    latest = {column: float(frame[column].iloc[-1]) for column in COVARIATE_COLUMNS}
    latest["covariate_signal"] = float(covariate_signal[-1]) if len(covariate_signal) else 0.0

    logger.info(
        "External covariates for %s [%s]: fng=%.1f funding=%.5f oi=%.2f ls=%.3f top_ls=%.3f taker=%.3f btc_dom=%.2f dxy=%.2f us10y=%.2f event=%.3f signal=%.4f",
        symbol,
        timeframe,
        latest.get("fear_greed_index", 0.0),
        latest.get("funding_rate", 0.0),
        latest.get("open_interest", 0.0),
        latest.get("long_short_ratio", 0.0),
        latest.get("top_trader_long_short_ratio", 0.0),
        latest.get("taker_buy_sell_ratio", 0.0),
        latest.get("btc_dominance", 0.0),
        latest.get("macro_dxy", 0.0),
        latest.get("macro_us10y", 0.0),
        latest.get("event_impact_score", 0.0),
        latest.get("covariate_signal", 0.0),
    )

    return covariate_signal, latest
