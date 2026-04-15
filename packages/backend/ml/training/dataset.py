from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import logging
import os
from typing import Iterable

import boto3
from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

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


def parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return ["BTCUSDT", "ETHUSDT", "SOLUSDT"]

    values = [item.strip().upper() for item in raw.split(",") if item.strip()]
    return values if values else ["BTCUSDT", "ETHUSDT", "SOLUSDT"]


def _parse_timeframes(raw: str | None) -> set[str] | None:
    if not raw:
        return None

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


def add_engineered_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()

    group_columns = [column for column in ("symbol", "timeframe") if column in dataframe.columns]
    if group_columns:
        dataframe["return_1"] = dataframe.groupby(group_columns, sort=False)["close"].pct_change()
        dataframe["volatility_10"] = dataframe.groupby(group_columns, sort=False)["return_1"].transform(
            lambda series: series.rolling(window=10, min_periods=10).std()
        )
        dataframe["volatility_30"] = dataframe.groupby(group_columns, sort=False)["return_1"].transform(
            lambda series: series.rolling(window=30, min_periods=30).std()
        )
    else:
        dataframe["return_1"] = dataframe["close"].pct_change()
        dataframe["volatility_10"] = dataframe["return_1"].rolling(window=10, min_periods=10).std()
        dataframe["volatility_30"] = dataframe["return_1"].rolling(window=30, min_periods=30).std()

    if "sentiment_score" not in dataframe.columns:
        dataframe["sentiment_score"] = 0.0
    dataframe["sentiment_score"] = dataframe["sentiment_score"].fillna(0.0)

    dataframe = _add_talib_features(dataframe)

    required_feature_columns = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "return_1",
        "volatility_10",
        "volatility_30",
        "sentiment_score",
        "pattern_doji",
        "pattern_engulfing",
        "pattern_hammer",
        "symbol",
    ]
    if "timeframe" in dataframe.columns:
        required_feature_columns.append("timeframe")

    present_columns = [column for column in required_feature_columns if column in dataframe.columns]
    return dataframe.dropna(subset=present_columns).reset_index(drop=True)


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

    if "timeframe" in dataframe.columns:
        dataframe["timeframe"] = dataframe["timeframe"].astype(str).str.strip().str.lower()
        allowed_timeframes = _parse_timeframes(config.timeframe)
        if allowed_timeframes is not None:
            dataframe = dataframe[dataframe["timeframe"].isin(allowed_timeframes)]

    dataframe = dataframe.sort_values(["symbol", "timestamp"])

    if config.max_rows_per_symbol > 0:
        dataframe = dataframe.groupby("symbol", group_keys=False).tail(config.max_rows_per_symbol)

    if "timeframe" in dataframe.columns:
        pre_feature_counts = dataframe.groupby(["symbol", "timeframe"]).size().sort_values(ascending=False)
    else:
        pre_feature_counts = dataframe.groupby(["symbol"]).size().sort_values(ascending=False)
    pre_feature_rows = len(dataframe)

    dataframe = add_engineered_features(dataframe)

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
) -> dict[str, str]:
    closes = window["close"].to_numpy(dtype=np.float64)
    sentiment = float(window["sentiment_score"].iloc[-1])
    vol10 = float(window["volatility_10"].iloc[-1])
    vol30 = float(window["volatility_30"].iloc[-1])

    pattern_features = (
        f"doji={int(window['pattern_doji'].iloc[-1])},"
        f"engulfing={int(window['pattern_engulfing'].iloc[-1])},"
        f"hammer={int(window['pattern_hammer'].iloc[-1])}"
    )

    input_text = (
        f"symbol={symbol}; timeframe={timeframe}; "
        f"sentiment={sentiment:.4f}; vol10={vol10:.6f}; vol30={vol30:.6f}; "
        f"patterns={''.join(pattern_features)}; "
        "close_series="
        + ",".join(f"{value:.8f}" for value in closes)
    )

    target_text = ",".join(f"{value:.8f}" for value in future_close_values)

    return {
        "input_text": input_text,
        "target_text": target_text,
        "symbol": symbol,
        "timestamp": str(window["timestamp"].iloc[-1]),
    }


def build_supervised_records(
    dataframe: pd.DataFrame,
    config: TrainingDatasetConfig,
) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []

    group_columns = ["symbol", "timeframe"] if "timeframe" in dataframe.columns else ["symbol"]

    for group_key, symbol_frame in dataframe.groupby(group_columns):
        if isinstance(group_key, tuple):
            symbol = str(group_key[0])
            timeframe = str(group_key[1])
        else:
            symbol = str(group_key)
            timeframe = config.timeframe

        symbol_frame = symbol_frame.sort_values("timestamp").reset_index(drop=True)

        upper = len(symbol_frame) - config.horizon
        for index in range(config.context_length, upper):
            start = index - config.context_length
            end = index
            forecast_end = index + config.horizon

            window = symbol_frame.iloc[start:end]
            targets = symbol_frame.iloc[index:forecast_end]["close"].to_numpy(dtype=np.float64)
            if len(targets) < config.horizon:
                continue

            records.append(_build_record(symbol, timeframe, window, targets))

    if not records:
        raise ValueError("No training records generated from loaded dataframe")

    return records


def build_training_datasets(config: TrainingDatasetConfig) -> DatasetDict:
    dataframe = load_market_dataframe(config)
    records = build_supervised_records(dataframe, config)

    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=max(0.02, 1.0 - config.train_split_ratio), seed=42)

    return DatasetDict(train=split["train"], validation=split["test"])
