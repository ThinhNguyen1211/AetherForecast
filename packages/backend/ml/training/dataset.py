from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from typing import Any

import awswrangler as wr
import boto3
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict

logger = logging.getLogger(__name__)


@dataclass
class TrainingDatasetConfig:
    """Configuration for loading and preprocessing market training data."""

    data_bucket: str
    symbols: list[str]
    timeframe: str
    horizon: int
    context_length: int
    max_rows_per_symbol: int
    train_split_ratio: float
    aws_region: str
    aws_endpoint_url: str | None = None
    walk_forward_windows: int = 4
    walk_forward_eval_size: int = 128
    external_covariate_scale: float = 0.0018
    enable_external_fetch: bool = True
    strict_external_data: bool = False


def parse_symbols(symbols: str | list[str] | None) -> list[str]:
    """Normalize a comma-separated symbol string or list into a clean list."""
    if symbols is None:
        return []
    if isinstance(symbols, list):
        raw = symbols
    else:
        raw = [piece.strip() for piece in symbols.split(",")]
    return [symbol.upper() for symbol in raw if symbol.strip()]


def _build_s3_path(config: TrainingDatasetConfig) -> str:
    """Construct the S3 prefix for partitioned parquet data."""
    parquet_prefix = os.getenv("PARQUET_PREFIX", "market/klines")
    return f"s3://{config.data_bucket}/{parquet_prefix}/"


def load_market_dataframe(config: TrainingDatasetConfig) -> pd.DataFrame:
    """Load enriched historical OHLCV data from the S3 data lake.

    Reads Hive-partitioned parquet files (symbol, year, month, day) directly
    from S3 using awswrangler, then normalizes columns for downstream training.
    """
    s3_path = _build_s3_path(config)
    logger.info("Loading market data from %s for symbols=%s", s3_path, config.symbols)

    if not config.symbols:
        raise ValueError("No symbols provided for training data loading")

    # awswrangler partition_filter receives a dict of partition key -> string value.
    def _symbol_filter(partitions: dict[str, str]) -> bool:
        return partitions.get("symbol", "").upper() in config.symbols

    try:
        session = boto3.Session(region_name=config.aws_region)
        df = wr.s3.read_parquet(
            path=s3_path,
            dataset=True,
            partition_filter=_symbol_filter,
            boto3_session=session,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to read parquet data from {s3_path}: {exc}") from exc

    if df.empty:
        raise ValueError(f"No market data found at {s3_path} for symbols {config.symbols}")

    # Normalize column names: lowercase and strip whitespace.
    df.columns = [str(col).lower().strip() for col in df.columns]

    # Ensure required columns exist.
    required = {"symbol", "timestamp", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Market data is missing required columns: {missing}")

    # Parse timestamp.
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

    # Drop duplicates and sort.
    df = df.drop_duplicates(subset=["symbol", "timestamp"]).sort_values(["symbol", "timestamp"])

    # Limit rows per symbol to control memory.
    if config.max_rows_per_symbol > 0:
        df = df.groupby("symbol").tail(config.max_rows_per_symbol)

    # Add timeframe column if missing (derived from environment default or infer).
    if "timeframe" not in df.columns:
        default_timeframe = config.timeframe.split(",")[0]
        df["timeframe"] = default_timeframe

    logger.info(
        "Loaded market dataframe: rows=%s symbols=%s columns=%s",
        len(df),
        df["symbol"].nunique(),
        list(df.columns),
    )
    return df.reset_index(drop=True)


def _build_series_splits(
    df: pd.DataFrame,
    config: TrainingDatasetConfig,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Split each symbol series into train/eval numpy arrays."""
    train_inputs: list[np.ndarray] = []
    eval_inputs: list[np.ndarray] = []

    group_columns = ["symbol", "timeframe"] if "timeframe" in df.columns else ["symbol"]
    min_length = max(config.context_length + config.horizon + 8, config.horizon * 4)

    for group_key, group in df.groupby(group_columns):
        ordered = group.sort_values("timestamp").reset_index(drop=True)
        closes = ordered["close"].to_numpy(dtype=np.float32)

        if len(closes) < min_length:
            logger.warning(
                "Skipping group %s: only %s rows (min %s required)",
                group_key,
                len(closes),
                min_length,
            )
            continue

        split_index = int(len(closes) * config.train_split_ratio)
        split_index = max(config.context_length + 1, min(split_index, len(closes) - config.horizon - 1))

        train_inputs.append(closes[:split_index])
        eval_start = max(0, split_index - config.context_length)
        eval_inputs.append(closes[eval_start:])

    if not train_inputs:
        raise ValueError("No symbol series met the minimum length requirement for training")

    return train_inputs, eval_inputs


def build_training_datasets(config: TrainingDatasetConfig) -> DatasetDict:
    """Build a HuggingFace DatasetDict from S3 market data.

    This path is used for non-Chronos-2 trainers. For Chronos-2 native training,
    load_market_dataframe is used directly.
    """
    logger.info("Building HuggingFace datasets for symbols=%s", config.symbols)
    df = load_market_dataframe(config)
    train_series, eval_series = _build_series_splits(df, config)

    def _to_dataset(series_list: list[np.ndarray]) -> Dataset:
        records: list[dict[str, Any]] = []
        for idx, series in enumerate(series_list):
            records.append({"series_id": idx, "values": series.tolist()})
        return Dataset.from_list(records)

    dataset_dict = DatasetDict({
        "train": _to_dataset(train_series),
        "eval": _to_dataset(eval_series),
    })

    logger.info(
        "Dataset built: train_series=%s eval_series=%s",
        len(train_series),
        len(eval_series),
    )
    return dataset_dict
