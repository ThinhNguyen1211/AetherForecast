from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from functools import lru_cache
from typing import Any

import websockets
from fastapi import WebSocket

from src.core.config import get_settings
from src.core.metrics import put_custom_metric

logger = logging.getLogger(__name__)
SUPPORTED_REALTIME_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d", "1w"}

MessageParser = Callable[[dict[str, Any]], dict[str, Any] | None]


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class BinanceRealtimeHub:
    """Pub/Sub multiplexer for live Binance data consumed by frontend clients.

    For every distinct stream (kline/ticker/mark-price, per symbol[, timeframe]),
    exactly ONE upstream Binance WebSocket connection is opened, no matter how
    many frontend clients subscribe to it — N clients share 1 upstream
    connection, keyed by stream_key. The upstream connection is torn down as
    soon as its last subscriber disconnects (reference-counted via the
    connections set). This is what keeps client count decoupled from upstream
    Binance connection count: distinct SYMBOLS multiply upstream connections,
    distinct CLIENTS watching the same symbol do not.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self.spot_ws_base_url = settings.binance_ws_url.rstrip("/")
        self.futures_ws_base_url = settings.binance_futures_ws_url.rstrip("/")
        self.default_kline_interval = settings.realtime_kline_interval

        self._connections: dict[str, set[WebSocket]] = {}
        self._stream_tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._shutdown = asyncio.Event()

    def _normalize_timeframe(self, timeframe: str | None) -> str:
        normalized = (timeframe or self.default_kline_interval).strip().lower()
        if normalized not in SUPPORTED_REALTIME_INTERVALS:
            raise ValueError(
                f"Unsupported realtime timeframe '{timeframe}'. Allowed: {sorted(SUPPORTED_REALTIME_INTERVALS)}"
            )
        return normalized

    async def _publish_connection_metric(self) -> None:
        async with self._lock:
            total_connections = sum(len(items) for items in self._connections.values())

        put_custom_metric(
            metric_name="WebSocketConnections",
            value=total_connections,
            namespace="AetherForecast/API",
            dimensions={"Service": "backend"},
        )

        put_custom_metric(
            metric_name="WebSocketActiveClients",
            value=total_connections,
            dimensions={"Pipeline": "realtime"},
        )

    # -------------------------------------------------------------------
    # Kline (candlestick chart)
    # -------------------------------------------------------------------

    def _parse_kline_message(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        kline = payload.get("k")
        if not isinstance(kline, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                kline = data.get("k")

        if not isinstance(kline, dict):
            return None

        timestamp_ms = int(kline.get("t", 0))
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, UTC).isoformat()

        return {
            "event": "kline",
            "symbol": str(kline.get("s", "")).upper(),
            "timeframe": str(kline.get("i", self.default_kline_interval)),
            "timestamp": timestamp,
            "time": int(timestamp_ms / 1000),
            "open": _safe_float(kline.get("o")),
            "high": _safe_float(kline.get("h")),
            "low": _safe_float(kline.get("l")),
            "close": _safe_float(kline.get("c")),
            "volume": _safe_float(kline.get("v")),
            "is_closed": bool(kline.get("x", False)),
        }

    async def subscribe_kline(self, symbol: str, timeframe: str, socket: WebSocket) -> None:
        symbol = symbol.upper()
        normalized_timeframe = self._normalize_timeframe(timeframe)
        stream_key = f"kline:{symbol}:{normalized_timeframe}"
        url = f"{self.spot_ws_base_url}/ws/{symbol.lower()}@kline_{normalized_timeframe}"
        await self._subscribe(stream_key, url, self._parse_kline_message, socket)

    async def unsubscribe_kline(self, symbol: str, timeframe: str, socket: WebSocket) -> None:
        symbol = symbol.upper()
        normalized_timeframe = self._normalize_timeframe(timeframe)
        stream_key = f"kline:{symbol}:{normalized_timeframe}"
        await self._unsubscribe(stream_key, socket)

    # -------------------------------------------------------------------
    # 24h Ticker (spot) — last price + price change + change percent
    # -------------------------------------------------------------------

    def _parse_ticker_message(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        symbol = payload.get("s")
        last_price = payload.get("c")
        if symbol is None or last_price is None:
            return None

        return {
            "event": "ticker",
            "symbol": str(symbol).upper(),
            "last_price": _safe_float(last_price),
            "price_change": _safe_float(payload.get("p")),
            "change_percent": _safe_float(payload.get("P")),
        }

    async def subscribe_ticker(self, symbol: str, socket: WebSocket) -> None:
        symbol = symbol.upper()
        stream_key = f"ticker:{symbol}"
        url = f"{self.spot_ws_base_url}/ws/{symbol.lower()}@ticker"
        await self._subscribe(stream_key, url, self._parse_ticker_message, socket)

    async def unsubscribe_ticker(self, symbol: str, socket: WebSocket) -> None:
        stream_key = f"ticker:{symbol.upper()}"
        await self._unsubscribe(stream_key, socket)

    # -------------------------------------------------------------------
    # Mark Price + Funding Rate (futures)
    # -------------------------------------------------------------------

    def _parse_mark_price_message(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        symbol = payload.get("s")
        mark_price = payload.get("p")
        if symbol is None or mark_price is None:
            return None

        return {
            "event": "mark_price",
            "symbol": str(symbol).upper(),
            "mark_price": _safe_float(mark_price),
            "funding_rate": _safe_float(payload.get("r")),
        }

    async def subscribe_mark_price(self, symbol: str, socket: WebSocket) -> None:
        symbol = symbol.upper()
        stream_key = f"markprice:{symbol}"
        url = f"{self.futures_ws_base_url}/ws/{symbol.lower()}@markPrice@1s"
        await self._subscribe(stream_key, url, self._parse_mark_price_message, socket)

    async def unsubscribe_mark_price(self, symbol: str, socket: WebSocket) -> None:
        stream_key = f"markprice:{symbol.upper()}"
        await self._unsubscribe(stream_key, socket)

    # -------------------------------------------------------------------
    # Generic Pub/Sub machinery shared by all stream types above
    # -------------------------------------------------------------------

    async def _broadcast(self, stream_key: str, message: dict[str, Any]) -> None:
        async with self._lock:
            targets = list(self._connections.get(stream_key, set()))

        if not targets:
            return

        stale: list[WebSocket] = []
        for socket in targets:
            try:
                await socket.send_json(message)
            except Exception:
                stale.append(socket)

        if stale:
            async with self._lock:
                current = self._connections.get(stream_key, set())
                for socket in stale:
                    current.discard(socket)

    async def _consume_stream(self, stream_key: str, url: str, parser: MessageParser) -> None:
        logger.info("Starting Binance upstream stream %s: %s", stream_key, url)

        while not self._shutdown.is_set():
            try:
                async with websockets.connect(
                    url,
                    ping_interval=20,
                    ping_timeout=20,
                    close_timeout=5,
                    max_queue=1024,
                ) as stream:
                    while not self._shutdown.is_set():
                        raw_message = await asyncio.wait_for(stream.recv(), timeout=60)
                        payload = json.loads(raw_message)
                        parsed = parser(payload)
                        if parsed is not None:
                            await self._broadcast(stream_key, parsed)
            except TimeoutError:
                logger.warning("Timed out reading Binance stream %s; reconnecting", stream_key)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Binance stream error for %s: %s", stream_key, exc)
                await asyncio.sleep(2)

        logger.info("Stopped Binance upstream stream %s", stream_key)

    async def _subscribe(
        self,
        stream_key: str,
        url: str,
        parser: MessageParser,
        socket: WebSocket,
    ) -> None:
        async with self._lock:
            connections = self._connections.setdefault(stream_key, set())
            connections.add(socket)

            task = self._stream_tasks.get(stream_key)
            if task is None or task.done():
                self._stream_tasks[stream_key] = asyncio.create_task(
                    self._consume_stream(stream_key, url, parser)
                )

        await self._publish_connection_metric()

    async def _unsubscribe(self, stream_key: str, socket: WebSocket) -> None:
        async with self._lock:
            connections = self._connections.get(stream_key)
            if connections is None:
                return

            connections.discard(socket)
            if connections:
                return

            self._connections.pop(stream_key, None)
            task = self._stream_tasks.pop(stream_key, None)

        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        await self._publish_connection_metric()

    async def close(self) -> None:
        self._shutdown.set()

        async with self._lock:
            tasks = list(self._stream_tasks.values())
            self._stream_tasks.clear()

        for task in tasks:
            task.cancel()

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


@lru_cache
def get_realtime_hub() -> BinanceRealtimeHub:
    return BinanceRealtimeHub()
