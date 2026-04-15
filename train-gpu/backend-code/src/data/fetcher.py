from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import os
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import httpx
import pandas as pd

from src.core.config import get_settings
from src.core.metrics import put_custom_metric
from src.data.feature_engineer import engineer_features
from src.data.parquet_writer import ParquetWriter
from src.data.sentiment import SentimentScorer

logger = logging.getLogger("aetherforecast.data.fetcher")


def _parse_symbols(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [item.strip().upper() for item in raw.split(",") if item.strip()]


def _getenv_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


@dataclass
class FetchRuntimeConfig:
    aws_region: str
    aws_endpoint_url: str | None
    data_bucket: str
    parquet_prefix: str
    watermark_prefix: str
    binance_base_url: str
    timeframe: str
    fetch_concurrency: int
    bootstrap_lookback_minutes: int
    max_kline_pages: int
    symbols: list[str]
    symbol_limit: int
    quote_assets: set[str] | None
    sentiment_mode: str
    sentiment_model_id: str
    fetch_loop_seconds: int


def load_runtime_config() -> FetchRuntimeConfig:
    settings = get_settings()
    raw_quote_assets = os.getenv("QUOTE_ASSETS", "").strip()
    if not raw_quote_assets or raw_quote_assets.lower() in {"all", "*", "any"}:
        quote_assets: set[str] | None = None
    else:
        quote_assets = {
            item.strip().upper()
            for item in raw_quote_assets.split(",")
            if item.strip()
        }

    return FetchRuntimeConfig(
        aws_region=settings.aws_region,
        aws_endpoint_url=settings.aws_endpoint_url,
        data_bucket=settings.data_bucket,
        parquet_prefix=os.getenv("PARQUET_PREFIX", "market/klines"),
        watermark_prefix=os.getenv("WATERMARK_PREFIX", "_metadata/watermarks"),
        binance_base_url=settings.binance_base_url.rstrip("/"),
        timeframe=os.getenv("KLINE_INTERVAL", "1m"),
        fetch_concurrency=_getenv_int("FETCH_CONCURRENCY", settings.fetch_concurrency),
        bootstrap_lookback_minutes=_getenv_int("BOOTSTRAP_LOOKBACK_MINUTES", 240),
        max_kline_pages=_getenv_int("MAX_KLINE_PAGES", 6),
        symbols=_parse_symbols(os.getenv("SYMBOLS")),
        symbol_limit=_getenv_int("SYMBOL_LIMIT", settings.fetch_symbol_limit),
        quote_assets=quote_assets,
        sentiment_mode=os.getenv("SENTIMENT_MODE", "simple"),
        sentiment_model_id=os.getenv("HF_SENTIMENT_MODEL_ID", "ProsusAI/finbert"),
        fetch_loop_seconds=_getenv_int("FETCH_LOOP_SECONDS", 0),
    )


class S3WatermarkStore:
    def __init__(
        self,
        bucket: str,
        prefix: str,
        aws_region: str,
        endpoint_url: str | None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3_client = boto3.client("s3", region_name=aws_region, endpoint_url=endpoint_url)

    def _key(self, symbol: str, timeframe: str) -> str:
        return f"{self.prefix}/{timeframe}/{symbol.upper()}.json"

    def get(self, symbol: str, timeframe: str) -> int | None:
        key = self._key(symbol, timeframe)
        try:
            response = self.s3_client.get_object(Bucket=self.bucket, Key=key)
            payload = json.loads(response["Body"].read().decode("utf-8"))
            value = payload.get("last_close_time_ms")
            return int(value) if value is not None else None
        except ClientError as exc:
            error_code = exc.response.get("Error", {}).get("Code", "")
            if error_code in {"NoSuchKey", "404"}:
                return None
            logger.warning("Unable to load watermark for %s (%s): %s", symbol, key, exc)
            return None
        except Exception as exc:
            logger.warning("Invalid watermark payload for %s (%s): %s", symbol, key, exc)
            return None

    def put(self, symbol: str, timeframe: str, last_close_time_ms: int) -> None:
        key = self._key(symbol, timeframe)
        payload = {
            "symbol": symbol.upper(),
            "timeframe": timeframe,
            "last_close_time_ms": int(last_close_time_ms),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        self.s3_client.put_object(
            Bucket=self.bucket,
            Key=key,
            Body=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            ContentType="application/json",
        )


class BinanceRestClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=8.0))

    async def close(self) -> None:
        await self.client.aclose()

    async def fetch_symbols(self, quote_assets: set[str] | None) -> list[str]:
        url = f"{self.base_url}/api/v3/exchangeInfo"
        response = await self.client.get(url)
        response.raise_for_status()

        payload = response.json()
        symbols = [
            item["symbol"]
            for item in payload.get("symbols", [])
            if item.get("status") == "TRADING"
            and item.get("isSpotTradingAllowed", True)
            and (
                quote_assets is None
                or item.get("quoteAsset", "").upper() in quote_assets
            )
        ]
        symbols.sort()
        return symbols

    async def fetch_incremental_klines(
        self,
        symbol: str,
        timeframe: str,
        since_ms: int,
        max_pages: int,
    ) -> list[list[Any]]:
        all_rows: list[list[Any]] = []
        cursor = max(0, int(since_ms))

        for _ in range(max_pages):
            params = {
                "symbol": symbol,
                "interval": timeframe,
                "startTime": cursor,
                "limit": 1000,
            }

            url = f"{self.base_url}/api/v3/klines"
            response = await self.client.get(url, params=params)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                delay = int(retry_after) if retry_after and retry_after.isdigit() else 1
                await asyncio.sleep(max(1, delay))
                continue

            response.raise_for_status()
            rows = response.json()
            if not isinstance(rows, list) or not rows:
                break

            all_rows.extend(rows)

            if len(rows) < 1000:
                break

            next_cursor = int(rows[-1][0]) + 1
            if next_cursor <= cursor:
                break
            cursor = next_cursor

        return all_rows


def _rows_to_dataframe(symbol: str, timeframe: str, rows: list[list[Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows,
        columns=[
            "open_time_ms",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time_ms",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base",
            "taker_buy_quote",
            "ignore",
        ],
    )

    numeric_columns = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "quote_asset_volume",
        "taker_buy_base",
        "taker_buy_quote",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame["open_time_ms"] = pd.to_numeric(frame["open_time_ms"], errors="coerce").astype("Int64")
    frame["close_time_ms"] = pd.to_numeric(frame["close_time_ms"], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=["open_time_ms", "close_time_ms", "open", "high", "low", "close"])

    frame["timestamp"] = pd.to_datetime(frame["open_time_ms"], unit="ms", utc=True)
    frame["symbol"] = symbol.upper()
    frame["timeframe"] = timeframe

    frame["open_time_ms"] = frame["open_time_ms"].astype("int64")
    frame["close_time_ms"] = frame["close_time_ms"].astype("int64")
    return frame


@dataclass
class SymbolRunResult:
    symbol: str
    rows_written: int
    last_close_time_ms: int | None


class DataFetchService:
    def __init__(self, config: FetchRuntimeConfig) -> None:
        self.config = config
        self.fetcher = BinanceRestClient(config.binance_base_url)
        self.writer = ParquetWriter(
            bucket=config.data_bucket,
            root_prefix=config.parquet_prefix,
            aws_region=config.aws_region,
            endpoint_url=config.aws_endpoint_url,
        )
        self.watermarks = S3WatermarkStore(
            bucket=config.data_bucket,
            prefix=config.watermark_prefix,
            aws_region=config.aws_region,
            endpoint_url=config.aws_endpoint_url,
        )
        self.sentiment = SentimentScorer(
            mode=config.sentiment_mode,
            model_id=config.sentiment_model_id,
            external_enabled=get_settings().external_sentiment_enabled,
            external_refresh_seconds=get_settings().external_sentiment_refresh_seconds,
            news_rss_urls=get_settings().external_sentiment_news_rss_urls,
            x_sentiment_endpoint=get_settings().external_x_sentiment_endpoint,
            geopolitical_sentiment_endpoint=get_settings().external_geopolitical_sentiment_endpoint,
        )
        self._semaphore = asyncio.Semaphore(max(1, config.fetch_concurrency))

    async def close(self) -> None:
        await self.fetcher.close()

    async def _resolve_symbols(self) -> list[str]:
        if self.config.symbols:
            symbols = self.config.symbols
        else:
            symbols = await self.fetcher.fetch_symbols(self.config.quote_assets)

        if self.config.symbol_limit > 0:
            symbols = symbols[: self.config.symbol_limit]
        return symbols

    async def _process_symbol(self, symbol: str) -> SymbolRunResult:
        async with self._semaphore:
            watermark = self.watermarks.get(symbol, self.config.timeframe)
            if watermark is None:
                fallback_start = datetime.now(timezone.utc) - timedelta(
                    minutes=self.config.bootstrap_lookback_minutes,
                )
                since_ms = int(fallback_start.timestamp() * 1000)
            else:
                since_ms = int(watermark) + 1

            rows = await self.fetcher.fetch_incremental_klines(
                symbol=symbol,
                timeframe=self.config.timeframe,
                since_ms=since_ms,
                max_pages=self.config.max_kline_pages,
            )

            if not rows:
                return SymbolRunResult(symbol=symbol, rows_written=0, last_close_time_ms=watermark)

            frame = _rows_to_dataframe(symbol, self.config.timeframe, rows)
            if frame.empty:
                return SymbolRunResult(symbol=symbol, rows_written=0, last_close_time_ms=watermark)

            engineered = engineer_features(frame)
            sentiment_series, _external_score, _external_source = self.sentiment.score_dataframe(
                symbol,
                engineered,
            )
            engineered["sentiment_score"] = sentiment_series

            rows_written = self.writer.append(engineered)
            last_close_ms = int(engineered["close_time_ms"].max())
            self.watermarks.put(symbol, self.config.timeframe, last_close_ms)

            return SymbolRunResult(
                symbol=symbol,
                rows_written=rows_written,
                last_close_time_ms=last_close_ms,
            )

    async def run_once(self) -> dict[str, int]:
        put_custom_metric(
            metric_name="FetchRuns",
            value=1,
            dimensions={"Pipeline": "data-fetch-cron"},
        )

        try:
            symbols = await self._resolve_symbols()
        except Exception:
            put_custom_metric(
                metric_name="FetchErrors",
                value=1,
                dimensions={"Pipeline": "data-fetch-cron"},
            )
            raise

        if not symbols:
            logger.warning("No symbols resolved for fetch cycle")
            return {"symbols": 0, "written": 0, "errors": 0}

        logger.info(
            "Starting fetch cycle: symbols=%s timeframe=%s concurrency=%s",
            len(symbols),
            self.config.timeframe,
            self.config.fetch_concurrency,
        )

        tasks = [asyncio.create_task(self._process_symbol(symbol)) for symbol in symbols]
        written = 0
        errors = 0

        try:
            for task in asyncio.as_completed(tasks):
                try:
                    result = await task
                    written += result.rows_written
                except (httpx.HTTPError, BotoCoreError, ClientError, ValueError) as exc:
                    errors += 1
                    logger.warning("Symbol fetch failed: %s", exc)
                except Exception as exc:  # pragma: no cover
                    errors += 1
                    logger.exception("Unexpected symbol fetch failure: %s", exc)
        except Exception:
            put_custom_metric(
                metric_name="FetchErrors",
                value=1,
                dimensions={"Pipeline": "data-fetch-cron"},
            )
            raise

        summary = {"symbols": len(symbols), "written": written, "errors": errors}
        put_custom_metric(
            metric_name="ParquetRowsWritten",
            value=written,
            dimensions={"Pipeline": "data-fetch-cron"},
        )
        put_custom_metric(
            metric_name="FetchErrors",
            value=errors,
            dimensions={"Pipeline": "data-fetch-cron"},
        )
        logger.info("Fetch cycle complete: %s", summary)
        return summary


async def run_fetch_cycle() -> None:
    config = load_runtime_config()
    if not config.data_bucket:
        raise ValueError("DATA_BUCKET (or DATA_S3_BUCKET) must be configured for parquet pipeline")

    service = DataFetchService(config)

    try:
        if config.fetch_loop_seconds > 0:
            logger.info("Running in loop mode every %s seconds", config.fetch_loop_seconds)
            while True:
                await service.run_once()
                await asyncio.sleep(config.fetch_loop_seconds)
        else:
            await service.run_once()
    finally:
        await service.close()


def main() -> None:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    asyncio.run(run_fetch_cycle())


if __name__ == "__main__":
    main()
