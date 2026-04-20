from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
import math
import os
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import boto3
from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DEFAULT_TIMEFRAMES = "1h,4h,1d"
BINANCE_ALLOWED_PERIODS = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}

COVARIATE_COLUMNS = [
    "fear_greed_index",
    "crypto_news_sentiment",
    "x_sentiment_score",
    "funding_rate",
    "open_interest",
    "long_short_ratio",
    "btc_dominance",
    "macro_dxy",
    "macro_us10y",
    "event_impact_score",
]

try:
    import awsrangler as wr
except Exception:  # pragma: no cover
    wr = None

try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None

try:
    import talib
except Exception:  # pragma: no cover
    talib = None


@dataclass
class TrainingDatasetConfig:
    data_bucket: str
    symbols: list[str]
    timeframe: str
    horizon: int
    context_length: int
    max_rows_per_symbol: int
    train_split_ratio: float
    aws_region: str
    aws_endpoint_url: str | None
    walk_forward_windows: int = 4
    walk_forward_eval_size: int = 128
    external_covariate_scale: float = 0.0018
    enable_external_fetch: bool = True
    strict_external_data: bool = False


def parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return DEFAULT_SYMBOLS

    values = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return values if values else DEFAULT_SYMBOLS


def _parse_timeframes(raw: str | None) -> set[str] | None:
    if not raw:
        return {item.strip().lower() for item in DEFAULT_TIMEFRAMES.split(",") if item.strip()}

    normalized = raw.strip().lower()
    if normalized in {"all", "*", "any"}:
        return None

    values = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return values if values else None


def _candidate_symbol_prefixes(data_bucket: str, symbol: str) -> list[str]:
    parquet_prefix = os.getenv("PARQUET_PREFIX", "market/klines").strip("/")
    candidates: list[str] = []

    if parquet_prefix:
        candidates.append(f"s3://{data_bucket}/{parquet_prefix}/symbol={symbol}/")

    candidates.append(f"s3://{data_bucket}/symbol={symbol}/")

    deduped: list[str] = []
    for path in candidates:
        if path not in deduped:
            deduped.append(path)
    return deduped


def _safe_to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_external_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    normalized = dataframe.copy()
    aliases: dict[str, list[str]] = {
        "fear_greed_index": ["fear_greed", "fng", "fng_value"],
        "crypto_news_sentiment": ["news_sentiment", "cryptopanic_sentiment", "news_score"],
        "x_sentiment_score": ["twitter_sentiment", "social_sentiment", "x_sentiment"],
        "funding_rate": ["funding", "fundingrate", "funding_rate_pct"],
        "open_interest": ["oi", "open_interest_value"],
        "long_short_ratio": ["longshortratio", "global_long_short_ratio", "ls_ratio"],
        "btc_dominance": ["btc_dom", "btc_dominance_pct"],
        "macro_dxy": ["dxy"],
        "macro_us10y": ["us10y", "us10y_yield", "treasury_10y"],
        "event_impact_score": ["event_impact", "event_score", "event_sentiment"],
    }

    for target, candidates in aliases.items():
        if target in normalized.columns:
            continue
        for candidate in candidates:
            if candidate in normalized.columns:
                normalized[target] = normalized[candidate]
                break

    return normalized


def _fetch_json(url: str, timeout: float = 8.0) -> dict | list | None:
    request = Request(url, headers={"User-Agent": "AetherForecast/1.0"})
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = response.read().decode("utf-8")
            return json.loads(payload)
    except (HTTPError, URLError, TimeoutError, ValueError) as exc:
        logger.debug("Unable to fetch %s: %s", url, exc)
        return None


@lru_cache(maxsize=1)
def _load_fear_greed_series() -> pd.DataFrame:
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


@lru_cache(maxsize=1)
def _load_btc_dominance_value() -> float | None:
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
def _load_fred_series(series_id: str, output_column: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    try:
        frame = pd.read_csv(url)
    except Exception as exc:  # pragma: no cover - network/runtime dependent
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


@lru_cache(maxsize=512)
def _load_binance_symbol_covariates(symbol: str, timeframe: str) -> pd.DataFrame:
    normalized_symbol = symbol.strip().upper()
    normalized_timeframe = timeframe.strip().lower() if timeframe else "1h"
    period = normalized_timeframe if normalized_timeframe in BINANCE_ALLOWED_PERIODS else "1h"

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

    if not frames:
        return pd.DataFrame(columns=["timestamp", "funding_rate", "open_interest", "long_short_ratio"])

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


def _load_optional_feature_file(
    env_var: str,
    target_column: str,
    candidate_columns: list[str],
) -> pd.DataFrame:
    source = os.getenv(env_var, "").strip()
    if not source:
        return pd.DataFrame(columns=["timestamp", target_column])

    try:
        lower = source.lower()
        if lower.endswith(".parquet"):
            frame = pd.read_parquet(source)
        elif lower.endswith(".json"):
            frame = pd.read_json(source)
        else:
            frame = pd.read_csv(source)
    except Exception as exc:
        logger.warning("Unable to read %s=%s: %s", env_var, source, exc)
        return pd.DataFrame(columns=["timestamp", target_column])

    if frame.empty:
        return pd.DataFrame(columns=["timestamp", target_column])

    timestamp_column = None
    for candidate in ("timestamp", "ts", "datetime", "date", "published_at", "event_time"):
        if candidate in frame.columns:
            timestamp_column = candidate
            break

    if timestamp_column is None:
        logger.warning("%s=%s is missing a timestamp column", env_var, source)
        return pd.DataFrame(columns=["timestamp", target_column])

    value_column = None
    for candidate in candidate_columns:
        if candidate in frame.columns:
            value_column = candidate
            break

    if value_column is None:
        numeric_candidates = [
            column
            for column in frame.columns
            if column not in {timestamp_column, "symbol", "timeframe"}
            and pd.api.types.is_numeric_dtype(frame[column])
        ]
        if numeric_candidates:
            value_column = numeric_candidates[0]

    if value_column is None:
        logger.warning("%s=%s does not contain a usable value column", env_var, source)
        return pd.DataFrame(columns=["timestamp", target_column])

    prepared = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(frame[timestamp_column], utc=True, errors="coerce"),
            target_column: pd.to_numeric(frame[value_column], errors="coerce"),
        }
    )

    if "symbol" in frame.columns:
        prepared["symbol"] = frame["symbol"].astype(str).str.strip().str.upper()
    if "timeframe" in frame.columns:
        prepared["timeframe"] = frame["timeframe"].astype(str).str.strip().str.lower()

    prepared = prepared.dropna(subset=["timestamp"]).sort_values("timestamp")
    return prepared


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
    merged[target_column] = _safe_to_numeric(merged[target_column]).fillna(_safe_to_numeric(merged["_external_value"]))
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
    merged[target_column] = _safe_to_numeric(merged[target_column]).fillna(_safe_to_numeric(merged["_external_value"]))

    merged = merged.sort_values("_row_id").drop(columns=["_row_id", "_external_value"])
    return merged.reset_index(drop=True)


def _inject_external_covariates(dataframe: pd.DataFrame, config: TrainingDatasetConfig) -> pd.DataFrame:
    enriched = _normalize_external_columns(dataframe)

    for column in COVARIATE_COLUMNS:
        if column not in enriched.columns:
            enriched[column] = np.nan

    if config.enable_external_fetch:
        enriched = _merge_daily_feature(enriched, _load_fear_greed_series(), "fear_greed_index")
        enriched = _merge_daily_feature(enriched, _load_fred_series("DTWEXBGS", "macro_dxy"), "macro_dxy")
        enriched = _merge_daily_feature(enriched, _load_fred_series("DGS10", "macro_us10y"), "macro_us10y")

        btc_dominance_value = _load_btc_dominance_value()
        if btc_dominance_value is not None:
            enriched["btc_dominance"] = _safe_to_numeric(enriched["btc_dominance"]).fillna(btc_dominance_value)

    optional_feature_specs = [
        ("EXTERNAL_NEWS_FILE", "crypto_news_sentiment", ["crypto_news_sentiment", "news_sentiment", "score"]),
        ("EXTERNAL_X_FILE", "x_sentiment_score", ["x_sentiment_score", "twitter_sentiment", "score"]),
        ("EXTERNAL_EVENTS_FILE", "event_impact_score", ["event_impact_score", "event_score", "impact"]),
        (
            "EXTERNAL_COVARIATES_FILE",
            "crypto_news_sentiment",
            ["crypto_news_sentiment", "news_sentiment", "sentiment_score"],
        ),
    ]

    for env_var, target_column, candidates in optional_feature_specs:
        feature_frame = _load_optional_feature_file(env_var, target_column, candidates)
        if feature_frame.empty:
            continue
        enriched = _merge_asof_feature(enriched, feature_frame, target_column)

    if config.enable_external_fetch:
        grouped_frames: list[pd.DataFrame] = []
        group_columns = ["symbol", "timeframe"] if "timeframe" in enriched.columns else ["symbol"]

        for group_key, group_frame in enriched.groupby(group_columns, sort=False):
            if isinstance(group_key, tuple):
                symbol = str(group_key[0])
                timeframe = str(group_key[1])
            else:
                symbol = str(group_key)
                timeframe = "1h"

            covariate_frame = _load_binance_symbol_covariates(symbol, timeframe)
            merged_group = group_frame.copy()
            for column in ("funding_rate", "open_interest", "long_short_ratio"):
                merged_group = _merge_asof_feature(merged_group, covariate_frame, column)

            grouped_frames.append(merged_group)

        if grouped_frames:
            enriched = pd.concat(grouped_frames, ignore_index=True)

    return enriched


def _standardize_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    rename_candidates = {
        "ts": "timestamp",
        "datetime": "timestamp",
        "date": "timestamp",
    }
    for source, target in rename_candidates.items():
        if source in dataframe.columns and target not in dataframe.columns:
            dataframe = dataframe.rename(columns={source: target})

    required_columns = {"timestamp", "open", "high", "low", "close", "volume"}
    missing = required_columns - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required market columns: {sorted(missing)}")

    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], utc=True, errors="coerce")
    dataframe = dataframe.dropna(subset=["timestamp"])
    return dataframe


def _add_talib_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    opens = dataframe["open"].to_numpy(dtype=np.float64)
    highs = dataframe["high"].to_numpy(dtype=np.float64)
    lows = dataframe["low"].to_numpy(dtype=np.float64)
    closes = dataframe["close"].to_numpy(dtype=np.float64)

    if talib is not None:
        dataframe["pattern_doji"] = talib.CDLDOJI(opens, highs, lows, closes)
        dataframe["pattern_engulfing"] = talib.CDLENGULFING(opens, highs, lows, closes)
        dataframe["pattern_hammer"] = talib.CDLHAMMER(opens, highs, lows, closes)
        return dataframe

    # Fallback heuristics when TA-Lib is unavailable in local environment.
    candle_range = np.maximum(highs - lows, 1e-8)
    body = np.abs(closes - opens)
    lower_shadow = np.minimum(opens, closes) - lows

    dataframe["pattern_doji"] = (body / candle_range < 0.1).astype(np.int32)
    dataframe["pattern_engulfing"] = np.sign(closes - opens).astype(np.int32)
    dataframe["pattern_hammer"] = (lower_shadow / candle_range > 0.55).astype(np.int32)
    return dataframe


def _rolling_zscore(series: pd.Series, window: int = 64) -> pd.Series:
    min_periods = max(6, window // 4)
    rolling_mean = series.rolling(window=window, min_periods=min_periods).mean()
    rolling_std = series.rolling(window=window, min_periods=min_periods).std(ddof=0)
    return (series - rolling_mean) / rolling_std.replace(0.0, np.nan)


def add_engineered_features(dataframe: pd.DataFrame, config: TrainingDatasetConfig) -> pd.DataFrame:
    engineered = dataframe.copy()
    engineered = _inject_external_covariates(engineered, config)

    group_columns = [column for column in ("symbol", "timeframe") if column in engineered.columns]
    if group_columns:
        engineered["return_1"] = engineered.groupby(group_columns, sort=False)["close"].pct_change()
        engineered["volatility_10"] = engineered.groupby(group_columns, sort=False)["return_1"].transform(
            lambda series: series.rolling(window=10, min_periods=10).std()
        )
        engineered["volatility_30"] = engineered.groupby(group_columns, sort=False)["return_1"].transform(
            lambda series: series.rolling(window=30, min_periods=30).std()
        )
        engineered["volatility_60"] = engineered.groupby(group_columns, sort=False)["return_1"].transform(
            lambda series: series.rolling(window=60, min_periods=30).std()
        )
    else:
        engineered["return_1"] = engineered["close"].pct_change()
        engineered["volatility_10"] = engineered["return_1"].rolling(window=10, min_periods=10).std()
        engineered["volatility_30"] = engineered["return_1"].rolling(window=30, min_periods=30).std()
        engineered["volatility_60"] = engineered["return_1"].rolling(window=60, min_periods=30).std()

    engineered["range_pct"] = (engineered["high"] - engineered["low"]) / np.maximum(engineered["close"], 1e-8)
    engineered["log_volume"] = np.log1p(np.maximum(_safe_to_numeric(engineered["volume"]), 0.0))

    if "sentiment_score" not in engineered.columns:
        engineered["sentiment_score"] = 0.0
    engineered["sentiment_score"] = _safe_to_numeric(engineered["sentiment_score"]).fillna(0.0)

    for column in COVARIATE_COLUMNS:
        engineered[column] = _safe_to_numeric(engineered[column]) if column in engineered.columns else np.nan
        if group_columns:
            engineered[column] = engineered.groupby(group_columns, sort=False)[column].transform(
                lambda series: series.ffill().bfill()
            )
        else:
            engineered[column] = engineered[column].ffill().bfill()
        engineered[column] = engineered[column].fillna(0.0)

    fear_greed_scaled = (engineered["fear_greed_index"] / 100.0) * 2.0 - 1.0
    engineered["sentiment_score"] = np.clip(
        engineered["sentiment_score"] * 0.35
        + engineered["crypto_news_sentiment"] * 0.25
        + engineered["x_sentiment_score"] * 0.25
        + fear_greed_scaled * 0.15,
        -1.0,
        1.0,
    )

    for column in COVARIATE_COLUMNS:
        z_column = f"{column}_z"
        if group_columns:
            engineered[z_column] = engineered.groupby(group_columns, sort=False)[column].transform(
                lambda series: _rolling_zscore(series.astype(np.float64), window=64)
            )
        else:
            engineered[z_column] = _rolling_zscore(engineered[column].astype(np.float64), window=64)
        engineered[z_column] = engineered[z_column].replace([np.inf, -np.inf], np.nan).fillna(0.0)

    covariate_weights = {
        "funding_rate": 0.18,
        "open_interest": 0.16,
        "long_short_ratio": 0.14,
        "btc_dominance": 0.10,
        "fear_greed_index": 0.10,
        "crypto_news_sentiment": 0.10,
        "x_sentiment_score": 0.10,
        "macro_dxy": -0.07,
        "macro_us10y": -0.05,
        "event_impact_score": 0.14,
    }

    combined_signal = np.zeros(len(engineered), dtype=np.float64)
    for column, weight in covariate_weights.items():
        combined_signal += engineered[f"{column}_z"].to_numpy(dtype=np.float64) * weight

    engineered["covariate_signal"] = np.tanh(combined_signal)

    signal_scale = float(np.clip(config.external_covariate_scale, 0.0003, 0.01))
    close_values = _safe_to_numeric(engineered["close"]).fillna(method="ffill").fillna(method="bfill")
    engineered["close_adjusted"] = close_values * np.exp(
        np.clip(engineered["covariate_signal"], -3.0, 3.0) * signal_scale
    )

    if config.strict_external_data:
        missing_columns = []
        for column in COVARIATE_COLUMNS:
            magnitude = float(np.nanmean(np.abs(engineered[column].to_numpy(dtype=np.float64))))
            if magnitude < 1e-8:
                missing_columns.append(column)
        if missing_columns:
            raise ValueError(
                "Strict external data mode is enabled but these covariates are unavailable: "
                + ", ".join(missing_columns)
            )

    engineered = _add_talib_features(engineered)

    required_feature_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "close_adjusted",
        "volume",
        "return_1",
        "volatility_10",
        "volatility_30",
        "sentiment_score",
        "covariate_signal",
        "pattern_doji",
        "pattern_engulfing",
        "pattern_hammer",
        "symbol",
    ] + COVARIATE_COLUMNS

    if "timeframe" in engineered.columns:
        required_feature_columns.append("timeframe")

    present_columns = [column for column in required_feature_columns if column in engineered.columns]
    return engineered.dropna(subset=present_columns).reset_index(drop=True)


def _load_with_awswrangler(config: TrainingDatasetConfig) -> pd.DataFrame:
    if wr is None:
        raise RuntimeError("awswrangler is not available")

    session = boto3.Session(region_name=config.aws_region)
    dataframes: list[pd.DataFrame] = []

    for symbol in config.symbols:
        loaded_frame: pd.DataFrame | None = None
        for path in _candidate_symbol_prefixes(config.data_bucket, symbol):
            try:
                frame = wr.s3.read_parquet(path=path, dataset=True, boto3_session=session)
            except Exception as exc:
                logger.debug("Failed to read %s via awswrangler: %s", path, exc)
                continue

            if frame.empty:
                continue

            loaded_frame = frame
            logger.info("Loaded %s rows for %s from %s", len(frame), symbol, path)
            break

        if loaded_frame is None:
            logger.warning("No parquet data found for %s via awswrangler", symbol)
            continue

        loaded_frame["symbol"] = symbol
        dataframes.append(loaded_frame)

    if not dataframes:
        raise ValueError("No parquet data loaded from S3 with awswrangler")

    return pd.concat(dataframes, ignore_index=True)


def _load_with_polars(config: TrainingDatasetConfig) -> pd.DataFrame:
    if pl is None:
        raise RuntimeError("polars is not available")

    dataframes: list[pd.DataFrame] = []
    for symbol in config.symbols:
        loaded_frame: pd.DataFrame | None = None
        for prefix in _candidate_symbol_prefixes(config.data_bucket, symbol):
            path = f"{prefix}**/*.parquet"
            try:
                frame = pl.scan_parquet(path).collect().to_pandas()
            except Exception as exc:
                logger.debug("Failed to read %s via polars: %s", path, exc)
                continue

            if frame.empty:
                continue

            loaded_frame = frame
            logger.info("Loaded %s rows for %s from %s", len(frame), symbol, path)
            break

        if loaded_frame is None:
            logger.warning("No parquet data found for %s via polars", symbol)
            continue

        loaded_frame["symbol"] = symbol
        dataframes.append(loaded_frame)

    if not dataframes:
        raise ValueError("No parquet data loaded from S3 with polars")

    return pd.concat(dataframes, ignore_index=True)


def load_market_dataframe(config: TrainingDatasetConfig) -> pd.DataFrame:
    if not config.data_bucket:
        raise ValueError("DATA_S3_BUCKET (or DATA_BUCKET) is required")

    dataframe: pd.DataFrame
    try:
        dataframe = _load_with_awswrangler(config)
    except Exception:
        dataframe = _load_with_polars(config)

    dataframe = _standardize_columns(dataframe)

    allowed_timeframes = _parse_timeframes(config.timeframe)
    if "timeframe" in dataframe.columns:
        dataframe["timeframe"] = dataframe["timeframe"].astype(str).str.strip().str.lower()
        if allowed_timeframes is not None:
            dataframe = dataframe[dataframe["timeframe"].isin(allowed_timeframes)]
    else:
        inferred = sorted(allowed_timeframes)[0] if allowed_timeframes else "1h"
        dataframe["timeframe"] = inferred

    dataframe["symbol"] = dataframe["symbol"].astype(str).str.strip().str.upper()
    dataframe = dataframe.sort_values(["symbol", "timeframe", "timestamp"])

    if config.max_rows_per_symbol > 0:
        row_group_columns = ["symbol", "timeframe"] if "timeframe" in dataframe.columns else ["symbol"]
        dataframe = dataframe.groupby(row_group_columns, group_keys=False).tail(config.max_rows_per_symbol)

    if "timeframe" in dataframe.columns:
        pre_feature_counts = dataframe.groupby(["symbol", "timeframe"]).size().sort_values(ascending=False)
    else:
        pre_feature_counts = dataframe.groupby(["symbol"]).size().sort_values(ascending=False)
    pre_feature_rows = len(dataframe)

    dataframe = add_engineered_features(dataframe, config)

    if dataframe.empty:
        top_counts = ", ".join(
            f"{index}={int(count)}" for index, count in pre_feature_counts.head(6).items()
        )
        min_rows_hint = max(config.context_length + config.horizon + 1, 30)
        raise ValueError(
            "Prepared dataset is empty after feature engineering. "
            f"rows_before_features={pre_feature_rows}; group_counts=[{top_counts or 'none'}]. "
            f"Need at least ~{min_rows_hint} rows per symbol/timeframe group. "
            "Try TIMEFRAME=all (or a denser timeframe like 15m), increase MAX_ROWS_PER_SYMBOL, "
            "or reduce CONTEXT_LENGTH/TRAINING_HORIZON."
        )

    return dataframe.reset_index(drop=True)


def _build_record(
    symbol: str,
    timeframe: str,
    window: pd.DataFrame,
    future_close_values: Iterable[float],
    fold_id: int,
    split_name: str,
) -> dict[str, str]:
    closes = window["close_adjusted"].to_numpy(dtype=np.float64)
    sentiment = float(window["sentiment_score"].iloc[-1])
    vol10 = float(window["volatility_10"].iloc[-1])
    vol30 = float(window["volatility_30"].iloc[-1])
    covariate_signal = float(window["covariate_signal"].iloc[-1])

    compact_closes = closes
    if closes.size > 320:
        step = int(math.ceil(closes.size / 320.0))
        compact_closes = closes[::step]

    latest = window.iloc[-1]
    covariate_snapshot = ",".join(
        f"{column}={float(latest[column]):.6f}" for column in COVARIATE_COLUMNS if column in window.columns
    )

    pattern_features = (
        f"doji={int(window['pattern_doji'].iloc[-1])},"
        f"engulfing={int(window['pattern_engulfing'].iloc[-1])},"
        f"hammer={int(window['pattern_hammer'].iloc[-1])}"
    )

    input_text = (
        f"symbol={symbol}; timeframe={timeframe}; "
        f"fold={fold_id}; split={split_name}; "
        f"sentiment={sentiment:.4f}; vol10={vol10:.6f}; vol30={vol30:.6f}; "
        f"covariate_signal={covariate_signal:.6f}; covariates={covariate_snapshot}; "
        f"patterns={''.join(pattern_features)}; "
        "close_series="
        + ",".join(f"{value:.8f}" for value in compact_closes)
    )

    target_text = ",".join(f"{value:.8f}" for value in future_close_values)

    return {
        "input_text": input_text,
        "target_text": target_text,
        "symbol": symbol,
        "timeframe": timeframe,
        "fold": str(fold_id),
        "split": split_name,
        "timestamp": str(window["timestamp"].iloc[-1]),
    }


def _build_walk_forward_ranges(total_points: int, config: TrainingDatasetConfig) -> list[tuple[int, int, int]]:
    start_index = config.context_length
    upper_bound = total_points - config.horizon
    if upper_bound <= start_index + 1:
        return []

    window_count = max(1, int(config.walk_forward_windows))
    usable = upper_bound - start_index

    if window_count <= 1 or usable < max(config.horizon * 2, 8):
        split_index = int(total_points * config.train_split_ratio)
        split_index = max(start_index + 1, min(split_index, upper_bound - 1))
        eval_end = min(upper_bound, split_index + max(config.horizon * 2, 8))
        return [(1, split_index, eval_end)]

    step = max(config.horizon, usable // (window_count + 1))
    eval_span = int(config.walk_forward_eval_size)
    if eval_span <= 0:
        eval_span = max(config.horizon * 2, step)

    ranges: list[tuple[int, int, int]] = []
    cutoff = start_index + step
    for fold_id in range(1, window_count + 1):
        train_end = min(cutoff, upper_bound - 1)
        validation_end = min(upper_bound, train_end + eval_span)
        if train_end > start_index and validation_end > train_end:
            ranges.append((fold_id, train_end, validation_end))
        cutoff += step

    if not ranges:
        split_index = int(total_points * config.train_split_ratio)
        split_index = max(start_index + 1, min(split_index, upper_bound - 1))
        eval_end = min(upper_bound, split_index + max(config.horizon * 2, 8))
        ranges = [(1, split_index, eval_end)]

    return ranges


def _dedupe_records(records: list[dict[str, str]]) -> list[dict[str, str]]:
    unique_records: list[dict[str, str]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    for record in records:
        key = (
            str(record.get("symbol", "")),
            str(record.get("timeframe", "")),
            str(record.get("timestamp", "")),
            str(record.get("fold", "")),
            str(record.get("split", "")),
        )
        if key in seen:
            continue
        seen.add(key)
        unique_records.append(record)

    return unique_records


def build_supervised_records(
    dataframe: pd.DataFrame,
    config: TrainingDatasetConfig,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    train_records: list[dict[str, str]] = []
    validation_records: list[dict[str, str]] = []

    group_columns = ["symbol", "timeframe"] if "timeframe" in dataframe.columns else ["symbol"]

    for group_key, symbol_frame in dataframe.groupby(group_columns):
        if isinstance(group_key, tuple):
            symbol = str(group_key[0])
            timeframe = str(group_key[1])
        else:
            symbol = str(group_key)
            timeframe = config.timeframe

        symbol_frame = symbol_frame.sort_values("timestamp").reset_index(drop=True)

        fold_ranges = _build_walk_forward_ranges(len(symbol_frame), config)
        if not fold_ranges:
            continue

        for fold_id, train_end, validation_end in fold_ranges:
            for index in range(config.context_length, train_end):
                start = index - config.context_length
                end = index
                forecast_end = index + config.horizon

                window = symbol_frame.iloc[start:end]
                targets = symbol_frame.iloc[index:forecast_end]["close"].to_numpy(dtype=np.float64)
                if len(targets) < config.horizon:
                    continue

                train_records.append(
                    _build_record(
                        symbol,
                        timeframe,
                        window,
                        targets,
                        fold_id=fold_id,
                        split_name="train",
                    )
                )

            for index in range(train_end, validation_end):
                start = index - config.context_length
                end = index
                forecast_end = index + config.horizon

                window = symbol_frame.iloc[start:end]
                targets = symbol_frame.iloc[index:forecast_end]["close"].to_numpy(dtype=np.float64)
                if len(targets) < config.horizon:
                    continue

                validation_records.append(
                    _build_record(
                        symbol,
                        timeframe,
                        window,
                        targets,
                        fold_id=fold_id,
                        split_name="validation",
                    )
                )

    train_records = _dedupe_records(train_records)
    validation_records = _dedupe_records(validation_records)

    if not train_records:
        raise ValueError("No training records generated from loaded dataframe")

    if not validation_records:
        fallback_size = max(1, int(len(train_records) * 0.05))
        validation_records = train_records[-fallback_size:]
        train_records = train_records[:-fallback_size] if len(train_records) > fallback_size else train_records

    if not train_records or not validation_records:
        raise ValueError("Unable to construct train/validation records for walk-forward training")

    return train_records, validation_records


def build_training_datasets(config: TrainingDatasetConfig) -> DatasetDict:
    dataframe = load_market_dataframe(config)
    train_records, validation_records = build_supervised_records(dataframe, config)

    return DatasetDict(
        train=Dataset.from_list(train_records),
        validation=Dataset.from_list(validation_records),
    )
