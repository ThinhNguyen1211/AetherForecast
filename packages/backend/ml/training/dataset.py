from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
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


def _get_s3_fs() -> ds.FileSystem:
    """Build a pyarrow S3 filesystem using the default AWS credential chain."""
    import pyarrow.fs as fs

    return fs.S3FileSystem(region=os.getenv("AWS_REGION", "ap-southeast-1"))


def _build_local_cache_path(config: TrainingDatasetConfig) -> Path:
    """Local path to cache downloaded parquet data for faster iteration."""
    parquet_prefix = os.getenv("PARQUET_PREFIX", "market/klines")
    local_root = Path(os.getenv("TRAIN_LOCAL_CACHE", "./artifacts/local-data-cache"))
    return local_root / config.data_bucket / parquet_prefix


def _sync_symbol_data_locally(config: TrainingDatasetConfig) -> Path:
    """Download parquet files for the requested symbols to a local cache.

    Uses AWS CLI sync under the hood to leverage efficient parallel S3 transfers
    and avoid urllib3 connection pool exhaustion in Python.
    """
    local_cache = _build_local_cache_path(config)
    local_cache.mkdir(parents=True, exist_ok=True)

    s3_prefix = f"s3://{config.data_bucket}/{os.getenv('PARQUET_PREFIX', 'market/klines')}/"

    # For a small number of symbols, sync the whole prefix and let local filtering do the rest.
    # This avoids very long AWS CLI include/exclude chains.
    cmd = f"aws s3 sync {s3_prefix} {local_cache} --only-show-errors"
    logger.info("Syncing market data to local cache: %s", local_cache)
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError(f"aws s3 sync failed with exit code {exit_code}")

    return local_cache


def load_market_dataframe(config: TrainingDatasetConfig) -> pd.DataFrame:
    """Load enriched historical OHLCV data from the S3 data lake.

    Downloads the requested symbol partitions to a local cache using AWS CLI,
    then reads them with pyarrow for fast, reliable access regardless of the
    number of files or symbols.
    """
    logger.info("Loading market data for symbols=%s", config.symbols)

    if not config.symbols:
        raise ValueError("No symbols provided for training data loading")

    local_cache = _sync_symbol_data_locally(config)
    logger.info("Reading parquet dataset from local cache: %s", local_cache)

    try:
        dataset = ds.dataset(str(local_cache), partitioning="hive")
        table = dataset.to_table()
    except Exception as exc:
        raise RuntimeError(f"Failed to read parquet data from {local_cache}: {exc}") from exc

    df = table.to_pandas()
    if df.empty:
        raise ValueError(f"No market data found for symbols {config.symbols}")

    # Normalize column names: lowercase and strip whitespace.
    df.columns = [str(col).lower().strip() for col in df.columns]

    # Filter to requested symbols if the local cache contains extras.
    if "symbol" in df.columns:
        df["symbol"] = df["symbol"].str.upper()
        df = df[df["symbol"].isin(config.symbols)]

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

    # Add timeframe column if missing.
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
        split_index = max(
            config.context_length + 1, min(split_index, len(closes) - config.horizon - 1)
        )

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

    dataset_dict = DatasetDict(
        {
            "train": _to_dataset(train_series),
            "eval": _to_dataset(eval_series),
        }
    )

    logger.info(
        "Dataset built: train_series=%s eval_series=%s",
        len(train_series),
        len(eval_series),
    )
    return dataset_dict
