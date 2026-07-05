"""Quant-grade Data Ingestion Pipeline.

Fetches 100% real quantitative data (zero LLM dependency):
  - Binance Spot OHLCV (1h candles)
  - Binance Futures: Funding Rate, Open Interest, Long/Short Ratio
  - Macro: DXY (US Dollar Index), US 10Y Yield via yfinance
  - Feature Engineering: RSI, MACD, Bollinger Bands, ATR, % Change

Writes enriched Parquet to S3 partitioned by symbol/year/month/day.
Designed to run every 15 minutes via cron on EC2.

Usage:
    python -m src.ml.data_ingestion
    python -m src.ml.data_ingestion --dry-run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Any

import httpx
import numpy as np
import pandas as pd

logger = logging.getLogger("aetherforecast.data_ingestion")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT",
    "ADAUSDT", "DOGEUSDT", "TRXUSDT", "AVAXUSDT", "LINKUSDT",
    "TONUSDT", "SHIBUSDT", "DOTUSDT", "LTCUSDT", "BCHUSDT",
    "NEARUSDT", "APTUSDT", "ARBUSDT", "OPUSDT", "ATOMUSDT",
    "INJUSDT", "RNDRUSDT", "ETCUSDT", "XLMUSDT", "FILUSDT",
    "SEIUSDT", "SUIUSDT", "ICPUSDT", "GRTUSDT", "AAVEUSDT",
    "MKRUSDT", "UNIUSDT", "PEPEUSDT", "FETUSDT", "RUNEUSDT",
    "ALGOUSDT", "MATICUSDT", "HBARUSDT", "IMXUSDT", "TAOUSDT",
    "STXUSDT", "TIAUSDT", "ENAUSDT", "PENDLEUSDT", "THETAUSDT",
    "EGLDUSDT", "KASUSDT", "JASMYUSDT", "CFXUSDT", "ARUSDT",
    "WIFUSDT", "BONKUSDT", "FLOKIUSDT", "ORDIUSDT", "PYTHUSDT",
    "AXSUSDT", "SANDUSDT", "MANAUSDT", "GALAUSDT", "CHZUSDT",
    "CRVUSDT", "SNXUSDT", "LDOUSDT", "DYDXUSDT", "YFIUSDT",
    "1INCHUSDT", "KAVAUSDT", "COMPUSDT", "ZECUSDT", "ENSUSDT",
    "KSMUSDT", "MINAUSDT", "ROSEUSDT", "GMTUSDT", "APEUSDT",
    "BLURUSDT", "AKTUSDT", "JTOUSDT", "WLDUSDT", "XAIUSDT",
    "ONDOUSDT", "BEAMUSDT", "NOTUSDT", "OMUSDT", "ZROUSDT",
    "AEVOUSDT", "STRKUSDT",
]

BINANCE_SPOT_URL = "https://api.binance.com/api/v3/klines"
BINANCE_FAPI_BASE = "https://fapi.binance.com"

# TA-Lib import (optional with pure-Python fallback)
try:
    import talib

    _HAS_TALIB = True
except ImportError:
    talib = None  # type: ignore[assignment]
    _HAS_TALIB = False


# ---------------------------------------------------------------------------
# Binance Data Fetchers
# ---------------------------------------------------------------------------

def _fetch_json(client: httpx.Client, url: str, params: dict[str, Any] | None = None) -> list | dict | None:
    """Safe JSON fetch with error handling."""
    try:
        resp = client.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("Failed to fetch %s: %s", url, exc)
        return None


def fetch_spot_ohlcv(client: httpx.Client, symbol: str, interval: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Fetch Binance Spot OHLCV klines."""
    data = _fetch_json(client, BINANCE_SPOT_URL, {"symbol": symbol, "interval": interval, "limit": limit})
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows = []
    for k in data:
        try:
            rows.append({
                "timestamp": pd.to_datetime(int(k[0]), unit="ms", utc=True),
                "open": float(k[1]),
                "high": float(k[2]),
                "low": float(k[3]),
                "close": float(k[4]),
                "volume": float(k[5]),
            })
        except (IndexError, TypeError, ValueError):
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_funding_rate(client: httpx.Client, symbol: str, limit: int = 500) -> pd.DataFrame:
    """Fetch Binance Futures Funding Rate history."""
    data = _fetch_json(client, f"{BINANCE_FAPI_BASE}/fapi/v1/fundingRate", {"symbol": symbol, "limit": limit})
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows = []
    for r in data:
        try:
            rows.append({
                "timestamp": pd.to_datetime(int(r["fundingTime"]), unit="ms", utc=True),
                "funding_rate": float(r["fundingRate"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_open_interest(client: httpx.Client, symbol: str, period: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Fetch Binance Futures Open Interest history."""
    data = _fetch_json(
        client,
        f"{BINANCE_FAPI_BASE}/futures/data/openInterestHist",
        {"symbol": symbol, "period": period, "limit": limit},
    )
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows = []
    for r in data:
        try:
            rows.append({
                "timestamp": pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True),
                "open_interest": float(r["sumOpenInterest"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def fetch_long_short_ratio(client: httpx.Client, symbol: str, period: str = "1h", limit: int = 500) -> pd.DataFrame:
    """Fetch Binance Futures Global Long/Short Account Ratio."""
    data = _fetch_json(
        client,
        f"{BINANCE_FAPI_BASE}/futures/data/globalLongShortAccountRatio",
        {"symbol": symbol, "period": period, "limit": limit},
    )
    if not isinstance(data, list) or not data:
        return pd.DataFrame()

    rows = []
    for r in data:
        try:
            rows.append({
                "timestamp": pd.to_datetime(int(r["timestamp"]), unit="ms", utc=True),
                "long_short_ratio": float(r["longShortRatio"]),
            })
        except (KeyError, TypeError, ValueError):
            continue

    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Macro Data (yfinance)
# ---------------------------------------------------------------------------

def fetch_macro_data() -> pd.DataFrame:
    """Fetch DXY and US10Y via yfinance with weekend forward-fill."""
    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — skipping macro data")
        return pd.DataFrame()

    frames: list[pd.DataFrame] = []

    for ticker, col_name in [("DX-Y.NYB", "macro_dxy"), ("^TNX", "macro_us10y")]:
        try:
            tf = yf.Ticker(ticker)
            hist = tf.history(period="30d", interval="1h")
            if hist.empty:
                logger.warning("yfinance returned empty data for %s", ticker)
                continue
            series = hist[["Close"]].copy()
            series.columns = [col_name]
            series.index = series.index.tz_localize("UTC") if series.index.tz is None else series.index.tz_convert("UTC")
            series = series.reset_index().rename(columns={"Datetime": "timestamp", "Date": "timestamp"})
            # Rename any remaining index column
            if "index" in series.columns:
                series = series.rename(columns={"index": "timestamp"})
            series["timestamp"] = pd.to_datetime(series["timestamp"], utc=True)
            # Weekend/holiday forward-fill
            series[col_name] = series[col_name].ffill()
            # Compute % change
            series[f"{col_name}_pct"] = series[col_name].pct_change().fillna(0.0)
            frames.append(series)
        except Exception as exc:
            logger.warning("Failed to fetch macro ticker %s: %s", ticker, exc)

    if not frames:
        return pd.DataFrame()

    merged = frames[0]
    for f in frames[1:]:
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            f.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
        )

    return merged


# ---------------------------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------------------------

def _compute_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Pure-Python RSI fallback."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).rolling(period, min_periods=period).mean().to_numpy()
    avg_loss = pd.Series(loss).rolling(period, min_periods=period).mean().to_numpy()
    rs = np.where(avg_loss > 1e-10, avg_gain / avg_loss, 100.0)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_macd(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-Python MACD (12,26,9) fallback."""
    s = pd.Series(close)
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd_line = (ema12 - ema26).to_numpy()
    signal_line = pd.Series(macd_line).ewm(span=9, adjust=False).mean().to_numpy()
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _compute_bbands(close: np.ndarray, period: int = 20, nbdev: float = 2.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pure-Python Bollinger Bands fallback."""
    s = pd.Series(close)
    middle = s.rolling(period, min_periods=period).mean().to_numpy()
    std = s.rolling(period, min_periods=period).std().to_numpy()
    upper = middle + nbdev * std
    lower = middle - nbdev * std
    return upper, middle, lower


def _compute_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Pure-Python ATR fallback."""
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).rolling(period, min_periods=period).mean().to_numpy()


def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add TA-Lib indicators (RSI, MACD, Bollinger, ATR) with pure-Python fallback."""
    if df.empty or len(df) < 30:
        return df

    close = df["close"].to_numpy(dtype=np.float64)
    high = df["high"].to_numpy(dtype=np.float64)
    low = df["low"].to_numpy(dtype=np.float64)

    if _HAS_TALIB:
        df["rsi_14"] = talib.RSI(close, timeperiod=14)
        macd, signal, hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist
        upper, middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["atr_14"] = talib.ATR(high, low, close, timeperiod=14)
    else:
        logger.info("TA-Lib not available — using pure-Python indicator fallback")
        df["rsi_14"] = _compute_rsi(close)
        macd, signal, hist = _compute_macd(close)
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist
        upper, middle, lower = _compute_bbands(close)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower
        df["atr_14"] = _compute_atr(high, low, close)

    return df


# ---------------------------------------------------------------------------
# S3 Writer
# ---------------------------------------------------------------------------

def write_to_s3(df: pd.DataFrame, symbol: str, bucket: str, prefix: str, region: str) -> None:
    """Write enriched DataFrame to S3 as day-partitioned Parquet.

    Target path:
        s3://<bucket>/<prefix>/symbol=<SYM>/year=YYYY/month=MM/day=DD/
    """
    try:
        import awswrangler as wr
    except ImportError:
        logger.error("awswrangler not installed — cannot write to S3")
        return

    if df.empty:
        logger.warning("Empty DataFrame for %s — skipping S3 write", symbol)
        return

    import boto3
    from botocore.exceptions import ClientError

    session = boto3.Session(region_name=region)

    df_out = df.copy()
    df_out["symbol"] = symbol
    df_out["year"] = df_out["timestamp"].dt.year.astype(str)
    df_out["month"] = df_out["timestamp"].dt.month.astype(str).str.zfill(2)
    df_out["day"] = df_out["timestamp"].dt.day.astype(str).str.zfill(2)

    s3_path = f"s3://{bucket}/{prefix.strip('/')}/symbol={symbol}/"

    try:
        wr.s3.to_parquet(
            df=df_out,
            path=s3_path,
            dataset=True,
            mode="overwrite_partitions",
            partition_cols=["year", "month", "day"],
            boto3_session=session,
        )
        logger.info(
            "✅ Wrote %d rows for %s → %s (partitions: year/month/day)",
            len(df_out), symbol, s3_path,
        )
    except ClientError as exc:
        error_code = exc.response["Error"]["Code"]
        error_msg = exc.response["Error"]["Message"]
        http_status = exc.response["ResponseMetadata"]["HTTPStatusCode"]
        logger.error(
            "❌ S3 ClientError writing %s: [%s] %s (HTTP %d). "
            "Check IAM role has s3:PutObject on arn:aws:s3:::%s/%s/*",
            symbol, error_code, error_msg, http_status,
            bucket, prefix.strip('/'),
        )
    except Exception as exc:
        logger.error("❌ Unexpected error writing %s to S3: %s", symbol, exc, exc_info=True)


# ---------------------------------------------------------------------------
# Main Ingestion Pipeline
# ---------------------------------------------------------------------------

def ingest_symbol(client: httpx.Client, symbol: str, macro_df: pd.DataFrame) -> pd.DataFrame:
    """Full ingestion pipeline for a single symbol."""
    # Step 1: Fetch Spot OHLCV
    ohlcv = fetch_spot_ohlcv(client, symbol)
    if ohlcv.empty:
        logger.warning("No OHLCV data for %s — skipping", symbol)
        return pd.DataFrame()

    ohlcv = ohlcv.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")

    # Step 2: Fetch Derivatives
    funding = fetch_funding_rate(client, symbol)
    oi = fetch_open_interest(client, symbol)
    ls = fetch_long_short_ratio(client, symbol)

    # Step 3: Merge derivatives via merge_asof (nearest timestamp)
    merged = ohlcv.copy()
    for deriv_df, col in [(funding, "funding_rate"), (oi, "open_interest"), (ls, "long_short_ratio")]:
        if not deriv_df.empty and col in deriv_df.columns:
            deriv_df = deriv_df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
            merged = pd.merge_asof(
                merged.sort_values("timestamp"),
                deriv_df[["timestamp", col]].sort_values("timestamp"),
                on="timestamp",
                direction="backward",
            )

    # Step 4: Merge macro data
    if not macro_df.empty:
        macro_cols = [c for c in macro_df.columns if c != "timestamp"]
        merged = pd.merge_asof(
            merged.sort_values("timestamp"),
            macro_df[["timestamp"] + macro_cols].sort_values("timestamp"),
            on="timestamp",
            direction="backward",
        )

    # Step 5: Compute % changes for OI and macro
    if "open_interest" in merged.columns:
        merged["open_interest"] = merged["open_interest"].ffill()
        merged["oi_pct_change"] = merged["open_interest"].pct_change().fillna(0.0)

    for col in ["macro_dxy", "macro_us10y"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill()
            pct_col = f"{col}_pct"
            if pct_col not in merged.columns:
                merged[pct_col] = merged[col].pct_change().fillna(0.0)

    # Step 6: Forward-fill all NaN values from derivatives
    for col in ["funding_rate", "open_interest", "long_short_ratio"]:
        if col in merged.columns:
            merged[col] = merged[col].ffill().fillna(0.0)

    # Step 7: Add TA-Lib features
    merged = add_ta_features(merged)

    # Step 8: Add timeframe column
    merged["timeframe"] = "1h"

    return merged


def main() -> None:
    """Entry point for the data ingestion pipeline."""
    parser = argparse.ArgumentParser(description="AetherForecast Quant Data Ingestion")
    parser.add_argument("--dry-run", action="store_true", help="Fetch data but skip S3 writes")
    parser.add_argument("--symbols", default=os.getenv("SYMBOLS", ""), help="Comma-separated symbol list")
    args = parser.parse_args()

    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO"),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    bucket = os.getenv("DATA_BUCKET", os.getenv("DATA_S3_BUCKET", ""))
    prefix = os.getenv("PARQUET_PREFIX", "market/klines")
    region = os.getenv("AWS_REGION", "ap-southeast-1")

    if not bucket and not args.dry_run:
        logger.error(
            "❌ DATA_BUCKET env var is not set! Cannot write to S3. "
            "Set DATA_BUCKET or DATA_S3_BUCKET, or use --dry-run."
        )
        sys.exit(1)

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else DEFAULT_SYMBOLS

    logger.info("=" * 60)
    logger.info("AetherForecast Data Ingestion Pipeline")
    logger.info("Bucket:   s3://%s/%s/", bucket, prefix)
    logger.info("Symbols:  %d | Region: %s | Dry-run: %s", len(symbols), region, args.dry_run)
    logger.info("Partition: symbol/year/month/day")
    logger.info("=" * 60)

    # Pre-fetch macro data (shared across all symbols)
    logger.info("Fetching macro data (DXY, US10Y) via yfinance...")
    macro_df = fetch_macro_data()
    if macro_df.empty:
        logger.warning("No macro data available — will proceed without DXY/US10Y")
    else:
        logger.info("Macro data: %d rows, columns: %s", len(macro_df), list(macro_df.columns))

    success = 0
    failed = 0
    start_time = time.monotonic()

    with httpx.Client(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
        for i, symbol in enumerate(symbols, 1):
            logger.info("[%d/%d] Processing %s...", i, len(symbols), symbol)
            try:
                df = ingest_symbol(client, symbol, macro_df)
                if df.empty:
                    failed += 1
                    continue

                if args.dry_run:
                    logger.info("[DRY-RUN] %s: %d rows, cols=%s", symbol, len(df), list(df.columns))
                else:
                    write_to_s3(df, symbol, bucket, prefix, region)

                success += 1
            except Exception as exc:
                logger.error("Failed to process %s: %s", symbol, exc, exc_info=True)
                failed += 1

            # Rate limit protection: 200ms between symbols
            time.sleep(0.2)

    elapsed = time.monotonic() - start_time
    logger.info("=" * 60)
    logger.info("Ingestion complete: %d success, %d failed, %.1fs elapsed", success, failed, elapsed)
    logger.info("=" * 60)

    if failed > 0 and success == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
