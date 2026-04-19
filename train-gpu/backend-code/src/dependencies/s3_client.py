from datetime import datetime, timedelta, timezone
from functools import lru_cache
import logging
import math
from typing import Any

import boto3
from botocore.config import Config
from fastapi import HTTPException, status
import pandas as pd
import requests

from src.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

try:
    import awsrangler as wr
except Exception:  # pragma: no cover
    wr = None

try:
    import polars as pl
except Exception:  # pragma: no cover
    pl = None


SUPPORTED_TIMEFRAMES = {"1m", "5m", "15m", "1h", "4h", "1d", "1w"}
TIMEFRAME_TO_BINANCE_INTERVAL = {
    "1m": "1m",
    "5m": "5m",
    "15m": "15m",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    "1w": "1w",
}
TIMEFRAME_TO_SECONDS = {
    "1m": 60,
    "5m": 5 * 60,
    "15m": 15 * 60,
    "1h": 60 * 60,
    "4h": 4 * 60 * 60,
    "1d": 24 * 60 * 60,
    "1w": 7 * 24 * 60 * 60,
}
TIMEFRAME_TO_PANDAS_RULE = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W-MON",
}


def _normalize_timeframe(timeframe: str) -> str:
    normalized = (timeframe or "").strip().lower()
    if normalized not in SUPPORTED_TIMEFRAMES:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Unsupported timeframe '{timeframe}'. Allowed: {sorted(SUPPORTED_TIMEFRAMES)}",
        )
    return normalized


def _to_iso_timestamp(value: Any) -> str:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    return str(value)


def _timestamp_to_ms(value: Any) -> int:
    if isinstance(value, datetime):
        dt = value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)

    if isinstance(value, (int, float)):
        numeric = float(value)
        if numeric > 10_000_000_000:
            return int(numeric)
        return int(numeric * 1000)

    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"

    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return int(parsed.timestamp() * 1000)


def _normalize_record(record: dict[str, float | str]) -> dict[str, float | str] | None:
    try:
        timestamp_ms = _timestamp_to_ms(record.get("timestamp"))
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
        open_price = float(record["open"])
        high_price = float(record["high"])
        low_price = float(record["low"])
        close_price = float(record["close"])
        volume = float(record["volume"])

        if not (low_price <= open_price <= high_price and low_price <= close_price <= high_price):
            high_price = max(high_price, open_price, close_price, low_price)
            low_price = min(low_price, open_price, close_price, high_price)

        return {
            "timestamp": timestamp,
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": max(0.0, volume),
        }
    except Exception:
        return None


def _safe_timestamp_to_ms(value: Any) -> int | None:
    try:
        return _timestamp_to_ms(value)
    except Exception:
        return None


class S3ParquetClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session = boto3.Session(region_name=settings.aws_region)
        self.binance_base_url = settings.binance_base_url.rstrip("/")
        client_config = Config(retries={"max_attempts": 5, "mode": "standard"})
        self.s3_client = self.session.client(
            "s3",
            endpoint_url=settings.aws_endpoint_url,
            config=client_config,
        )

    def _candidate_base_paths(self, symbol: str, timeframe: str) -> list[str]:
        normalized_prefix = self.settings.parquet_prefix.strip("/")
        primary = f"s3://{self.settings.data_bucket}/{normalized_prefix}/symbol={symbol}/"
        timeframe_partition = (
            f"s3://{self.settings.data_bucket}/{normalized_prefix}/symbol={symbol}/timeframe={timeframe}/"
        )
        legacy = f"s3://{self.settings.data_bucket}/symbol={symbol}/"

        candidates = [timeframe_partition, primary, legacy]
        deduped: list[str] = []
        for path in candidates:
            if path not in deduped:
                deduped.append(path)
        return deduped

    def _estimate_partition_window_days(self, timeframe: str, limit: int) -> int:
        candle_seconds = TIMEFRAME_TO_SECONDS.get(timeframe, 60 * 60)
        estimated_days = int(math.ceil((limit * candle_seconds) / (24 * 60 * 60) * 1.35))

        if timeframe == "1m":
            min_window_days = 2
        elif timeframe in {"5m", "15m"}:
            min_window_days = 4
        else:
            min_window_days = 14

        return max(min_window_days, min(estimated_days, 1825))

    def _build_partition_filter(
        self,
        start_datetime: datetime | None,
        end_datetime: datetime | None,
        timeframe: str,
    ):
        if start_datetime is None and end_datetime is None and not timeframe:
            return None

        def _filter(partition: dict[str, str]) -> bool:
            try:
                partition_timeframe = partition.get("timeframe")
                if partition_timeframe is not None and partition_timeframe.strip().lower() != timeframe:
                    return False

                year_raw = partition.get("year")
                if year_raw is None:
                    return True

                year = int(year_raw)
                month_raw = partition.get("month")
                day_raw = partition.get("day")
                month = int(month_raw) if month_raw is not None else None
                day = int(day_raw) if day_raw is not None else None

                if end_datetime is not None:
                    if year > end_datetime.year:
                        return False
                    if month is not None and year == end_datetime.year and month > end_datetime.month:
                        return False
                    if (
                        day is not None
                        and month is not None
                        and year == end_datetime.year
                        and month == end_datetime.month
                        and day > end_datetime.day
                    ):
                        return False

                if start_datetime is not None:
                    if year < start_datetime.year:
                        return False
                    if month is not None and year == start_datetime.year and month < start_datetime.month:
                        return False
                    if (
                        day is not None
                        and month is not None
                        and year == start_datetime.year
                        and month == start_datetime.month
                        and day < start_datetime.day
                    ):
                        return False

                return True
            except Exception:
                return True

        return _filter

    def _resample_dataframe_to_timeframe(self, dataframe: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        rule = TIMEFRAME_TO_PANDAS_RULE.get(timeframe)
        if rule is None:
            return dataframe

        base = dataframe[["timestamp", "open", "high", "low", "close", "volume"]].copy()
        base = base.sort_values("timestamp")

        if timeframe == "1m":
            return base

        resampled = (
            base.set_index("timestamp")
            .resample(rule, label="left", closed="left")
            .agg(
                {
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                }
            )
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )

        resampled["volume"] = pd.to_numeric(resampled["volume"], errors="coerce").fillna(0.0)
        return resampled

    def _prepare_chart_dataframe(
        self,
        dataframe: pd.DataFrame,
        timeframe: str,
        source_path: str,
    ) -> pd.DataFrame:
        if dataframe.empty:
            return dataframe

        working = dataframe.copy()

        if "timestamp" not in working.columns:
            for candidate in ["ts", "datetime", "date"]:
                if candidate in working.columns:
                    working = working.rename(columns={candidate: "timestamp"})
                    break

        required = ["timestamp", "open", "high", "low", "close", "volume"]
        missing = [column for column in required if column not in working.columns]
        if missing:
            logger.warning(
                "Skipping parquet path %s due to missing columns: %s",
                source_path,
                missing,
            )
            return pd.DataFrame(columns=required)

        working["timestamp"] = pd.to_datetime(working["timestamp"], utc=True, errors="coerce")
        for numeric_column in ["open", "high", "low", "close", "volume"]:
            working[numeric_column] = pd.to_numeric(working[numeric_column], errors="coerce")

        working = working.dropna(subset=required)
        working = working.sort_values("timestamp")

        if "timeframe" in working.columns:
            timeframe_values = working["timeframe"].astype(str).str.strip().str.lower()
            exact_match = working[timeframe_values == timeframe]
            if not exact_match.empty:
                return exact_match[required]

            logger.info(
                "No exact timeframe=%s rows in %s; resampling available data.",
                timeframe,
                source_path,
            )

        return self._resample_dataframe_to_timeframe(working, timeframe)

    def _fetch_from_parquet(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        start_timestamp_ms: int | None = None,
        before_timestamp_ms: int | None = None,
    ) -> list[dict[str, float | str]]:
        base_paths = self._candidate_base_paths(symbol, timeframe)
        records: list[dict[str, float | str]] = []

        start_datetime = (
            datetime.fromtimestamp(start_timestamp_ms / 1000, tz=timezone.utc)
            if start_timestamp_ms is not None
            else None
        )
        end_datetime = (
            datetime.fromtimestamp((before_timestamp_ms - 1) / 1000, tz=timezone.utc)
            if before_timestamp_ms is not None
            else datetime.now(timezone.utc)
        )

        partition_filter = self._build_partition_filter(start_datetime, end_datetime, timeframe)

        if wr is not None:
            for base_path in base_paths:
                try:
                    df = wr.s3.read_parquet(
                        path=base_path,
                        dataset=True,
                        partition_filter=partition_filter,
                        boto3_session=self.session,
                    )
                    if df.empty:
                        continue

                    prepared = self._prepare_chart_dataframe(df, timeframe, base_path)
                    if prepared.empty:
                        continue

                    for row in prepared.itertuples(index=False):
                        timestamp_value = getattr(row, "timestamp")
                        timestamp_ms = _safe_timestamp_to_ms(timestamp_value)
                        if timestamp_ms is None:
                            continue
                        if start_timestamp_ms is not None and timestamp_ms < start_timestamp_ms:
                            continue
                        if before_timestamp_ms is not None and timestamp_ms >= before_timestamp_ms:
                            continue

                        records.append(
                            {
                                "timestamp": _to_iso_timestamp(timestamp_value),
                                "open": float(getattr(row, "open")),
                                "high": float(getattr(row, "high")),
                                "low": float(getattr(row, "low")),
                                "close": float(getattr(row, "close")),
                                "volume": float(getattr(row, "volume")),
                            }
                        )

                    if len(records) > limit:
                        records = records[-limit:]

                    if records:
                        return records
                except Exception as exc:
                    logger.warning("awswrangler parquet read failed for %s (%s), trying next path", base_path, exc)

        if pl is not None:
            for base_path in base_paths:
                try:
                    query_path = f"{base_path}**/*.parquet"
                    df_polars = pl.scan_parquet(query_path).collect()

                    if df_polars.is_empty():
                        continue

                    prepared = self._prepare_chart_dataframe(
                        df_polars.to_pandas(use_pyarrow_extension_array=False),
                        timeframe,
                        base_path,
                    )

                    if prepared.empty:
                        continue

                    if len(prepared) > limit:
                        prepared = prepared.tail(limit)

                    for row in prepared.to_dict(orient="records"):
                        timestamp_ms = _safe_timestamp_to_ms(row.get("timestamp"))
                        if timestamp_ms is None:
                            continue
                        if start_timestamp_ms is not None and timestamp_ms < start_timestamp_ms:
                            continue
                        if before_timestamp_ms is not None and timestamp_ms >= before_timestamp_ms:
                            continue

                        records.append(
                            {
                                "timestamp": _to_iso_timestamp(row.get("timestamp")),
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": float(row["volume"]),
                            }
                        )
                    if records:
                        if len(records) > limit:
                            records = records[-limit:]
                        return records
                except Exception as exc:
                    logger.warning("polars parquet read failed for %s: %s", base_path, exc)

        return records

    def _fetch_from_binance(
        self,
        symbol: str,
        timeframe: str,
        limit: int,
        before_timestamp_ms: int | None = None,
    ) -> list[dict[str, float | str]]:
        interval = TIMEFRAME_TO_BINANCE_INTERVAL[timeframe]
        total_limit = max(1, min(limit, 5000))
        pages = int(math.ceil(total_limit / 1000))

        all_rows: list[list[Any]] = []
        end_time_ms: int | None = before_timestamp_ms - 1 if before_timestamp_ms is not None else None

        for page in range(pages):
            remaining = total_limit - len(all_rows)
            if remaining <= 0:
                break

            request_limit = min(1000, remaining)
            params: dict[str, Any] = {
                "symbol": symbol,
                "interval": interval,
                "limit": request_limit,
            }
            if end_time_ms is not None:
                params["endTime"] = end_time_ms

            try:
                response = requests.get(
                    f"{self.binance_base_url}/api/v3/klines",
                    params=params,
                    timeout=12,
                )
                response.raise_for_status()
                rows = response.json()
            except Exception as exc:
                logger.warning(
                    "Binance kline fetch failed for %s %s page=%s: %s",
                    symbol,
                    timeframe,
                    page,
                    exc,
                )
                break

            if not isinstance(rows, list) or not rows:
                break

            all_rows = rows + all_rows

            first_open_time = int(rows[0][0])
            end_time_ms = max(0, first_open_time - 1)

            if len(rows) < request_limit:
                break

        records: list[dict[str, float | str]] = []
        for row in all_rows[-total_limit:]:
            timestamp = datetime.fromtimestamp(int(row[0]) / 1000, timezone.utc).isoformat().replace("+00:00", "Z")
            records.append(
                {
                    "timestamp": timestamp,
                    "open": float(row[1]),
                    "high": float(row[2]),
                    "low": float(row[3]),
                    "close": float(row[4]),
                    "volume": float(row[5]),
                }
            )

        return records

    def _merge_records(
        self,
        parquet_records: list[dict[str, float | str]],
        live_records: list[dict[str, float | str]],
        limit: int,
    ) -> list[dict[str, float | str]]:
        merged_by_timestamp: dict[int, dict[str, float | str]] = {}
        for record in parquet_records:
            normalized = _normalize_record(record)
            if normalized is None:
                continue
            merged_by_timestamp[_timestamp_to_ms(normalized["timestamp"])] = normalized
        for record in live_records:
            normalized = _normalize_record(record)
            if normalized is None:
                continue
            merged_by_timestamp[_timestamp_to_ms(normalized["timestamp"])] = normalized

        merged = [merged_by_timestamp[key] for key in sorted(merged_by_timestamp.keys())]
        if len(merged) > limit:
            merged = merged[-limit:]
        return merged

    def fetch_chart_points(
        self,
        symbol: str,
        timeframe: str,
        limit: int = 1200,
        from_timestamp: Any | None = None,
    ) -> list[dict]:
        symbol = symbol.upper()
        normalized_timeframe = _normalize_timeframe(timeframe)
        requested_limit = max(1, min(int(limit), 5000))

        before_timestamp_ms: int | None = None
        before_datetime: datetime | None = None
        if from_timestamp is not None:
            try:
                before_timestamp_ms = _timestamp_to_ms(from_timestamp)
                before_datetime = datetime.fromtimestamp(before_timestamp_ms / 1000, tz=timezone.utc)
            except Exception as exc:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Invalid from_timestamp '{from_timestamp}': {exc}",
                ) from exc

        window_days = self._estimate_partition_window_days(normalized_timeframe, requested_limit)
        window_end = before_datetime if before_datetime is not None else datetime.now(timezone.utc)
        window_start = window_end - timedelta(days=window_days)
        start_timestamp_ms = int(window_start.timestamp() * 1000)

        parquet_records: list[dict[str, float | str]] = []
        if not self.settings.data_bucket:
            logger.warning(
                "DATA_BUCKET is not configured. Falling back to Binance-only chart fetch for %s %s.",
                symbol,
                normalized_timeframe,
            )
        else:
            try:
                parquet_records = self._fetch_from_parquet(
                    symbol=symbol,
                    timeframe=normalized_timeframe,
                    limit=requested_limit,
                    start_timestamp_ms=start_timestamp_ms,
                    before_timestamp_ms=before_timestamp_ms,
                )
            except Exception as exc:
                logger.warning(
                    "Parquet chart fetch failed for %s %s, falling back to Binance-only: %s",
                    symbol,
                    normalized_timeframe,
                    exc,
                )

        parquet_records = self._merge_records(parquet_records, [], requested_limit)

        live_limit = requested_limit if len(parquet_records) < requested_limit else min(requested_limit, 1000)
        live_records = self._fetch_from_binance(
            symbol=symbol,
            timeframe=normalized_timeframe,
            limit=live_limit,
            before_timestamp_ms=before_timestamp_ms,
        )

        if not live_records and parquet_records:
            return parquet_records[-requested_limit:]

        if not live_records and not parquet_records:
            return []

        merged_records = self._merge_records(
            parquet_records=parquet_records,
            live_records=live_records,
            limit=requested_limit,
        )

        if before_timestamp_ms is not None:
            filtered_records: list[dict[str, float | str]] = []
            for record in merged_records:
                timestamp_ms = _safe_timestamp_to_ms(record.get("timestamp"))
                if timestamp_ms is None:
                    continue
                if timestamp_ms < before_timestamp_ms:
                    filtered_records.append(record)
            merged_records = filtered_records
            if len(merged_records) > requested_limit:
                merged_records = merged_records[-requested_limit:]

        return merged_records


@lru_cache
def get_s3_parquet_client() -> S3ParquetClient:
    return S3ParquetClient(get_settings())
