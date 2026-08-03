"""Real-time external market data fetchers for the AI Council.

All functions are async and use short timeouts so that a slow or unavailable
third-party API never blocks the AI analysis pipeline. On failure they log the
error and return a sensible neutral fallback.
"""

from __future__ import annotations

import logging

import httpx

from src.core.config import get_settings

logger = logging.getLogger(__name__)

_FEAR_GREED_URL = "https://api.alternative.me/fng/"
_BINANCE_FUNDING_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"

_DEFAULT_FEAR_GREED = 50.0
_DEFAULT_FUNDING_RATE = 0.0
_REQUEST_TIMEOUT = httpx.Timeout(5.0, connect=2.0)


async def fetch_fear_greed() -> float:
    """Fetch the latest Crypto Fear & Greed Index from alternative.me.

    Returns a value in the range [0, 100]. On any failure returns a neutral
    50.0 and logs the error.
    """
    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.get(_FEAR_GREED_URL)
            response.raise_for_status()
            payload = response.json()

        data = payload.get("data")
        if isinstance(data, list) and len(data) > 0:
            value = data[0].get("value")
            if value is not None:
                return float(value)

        logger.warning("Unexpected Fear & Greed response shape: %s", payload)
        return _DEFAULT_FEAR_GREED
    except httpx.TimeoutException:
        logger.warning("Fear & Greed API timed out after %ss", _REQUEST_TIMEOUT.read)
        return _DEFAULT_FEAR_GREED
    except Exception as exc:
        logger.warning("Failed to fetch Fear & Greed index: %s", exc)
        return _DEFAULT_FEAR_GREED


async def fetch_funding_rate(symbol: str) -> float:
    """Fetch the latest USDT-M futures funding rate for a symbol from Binance.

    The symbol is normalized to uppercase before querying. On any failure
    returns 0.0 and logs the error.
    """
    normalized_symbol = symbol.upper().strip()
    if not normalized_symbol:
        return _DEFAULT_FUNDING_RATE

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.get(
                _BINANCE_FUNDING_URL,
                params={"symbol": normalized_symbol},
            )
            response.raise_for_status()
            payload = response.json()

        raw_rate = payload.get("lastFundingRate")
        if raw_rate is None:
            logger.warning(
                "Binance funding rate response missing lastFundingRate for %s: %s",
                normalized_symbol,
                payload,
            )
            return _DEFAULT_FUNDING_RATE

        return float(raw_rate)
    except httpx.TimeoutException:
        logger.warning(
            "Binance funding rate API timed out after %ss for %s",
            _REQUEST_TIMEOUT.read,
            normalized_symbol,
        )
        return _DEFAULT_FUNDING_RATE
    except Exception as exc:
        logger.warning(
            "Failed to fetch funding rate for %s: %s",
            normalized_symbol,
            exc,
        )
        return _DEFAULT_FUNDING_RATE


async def fetch_latest_klines(
    symbol: str,
    timeframe: str,
    limit: int = 50,
) -> list[dict[str, float]]:
    """Fetch the most recent klines directly from Binance's public REST API.

    Same endpoint (/api/v3/klines) and base URL as the ingestion pipeline
    (src/data/fetcher.py), so results are consistent with what the model was
    trained on. Returns OHLCV dicts oldest-first, or an empty list on any
    failure — the caller decides how to treat "no live data available".
    """
    settings = get_settings()
    normalized_symbol = symbol.upper().strip()
    if not normalized_symbol:
        return []

    url = f"{settings.binance_base_url.rstrip('/')}/api/v3/klines"

    try:
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as client:
            response = await client.get(
                url,
                params={"symbol": normalized_symbol, "interval": timeframe, "limit": limit},
            )
            response.raise_for_status()
            rows = response.json()

        if not isinstance(rows, list):
            logger.warning("Unexpected Binance klines response shape for %s: %s", normalized_symbol, rows)
            return []

        return [
            {
                "timestamp": row[0],
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
            for row in rows
        ]
    except httpx.TimeoutException:
        logger.warning(
            "Binance klines API timed out after %ss for %s",
            _REQUEST_TIMEOUT.read,
            normalized_symbol,
        )
        return []
    except Exception as exc:
        logger.warning("Failed to fetch real-time klines for %s: %s", normalized_symbol, exc)
        return []


