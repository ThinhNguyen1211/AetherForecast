from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from functools import lru_cache
import json
import logging
from typing import Any

from fastapi import WebSocket
import websockets

from src.core.config import get_settings
from src.core.metrics import put_custom_metric

logger = logging.getLogger(__name__)
SUPPORTED_REALTIME_INTERVALS = {"1m", "5m", "15m", "1h", "4h", "1d", "1w"}


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class BinanceRealtimeHub:
    def __init__(self) -> None:
        settings = get_settings()
        self.ws_base_url = settings.binance_ws_url.rstrip("/")
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

    @staticmethod
    def _stream_key(symbol: str, timeframe: str) -> str:
        return f"{symbol.upper()}:{timeframe}"

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

    def _stream_url(self, symbol: str, timeframe: str) -> str:
        return f"{self.ws_base_url}/ws/{symbol.lower()}@kline_{timeframe}"

    def _parse_kline_message(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        kline = payload.get("k")
        if not isinstance(kline, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                kline = data.get("k")

        if not isinstance(kline, dict):
            return None

        timestamp_ms = int(kline.get("t", 0))
        timestamp = datetime.fromtimestamp(timestamp_ms / 1000, timezone.utc).isoformat()

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

    async def _consume_symbol_stream(self, symbol: str, timeframe: str) -> None:
        stream_key = self._stream_key(symbol, timeframe)
        url = self._stream_url(symbol, timeframe)
        logger.info("Starting Binance stream for %s (%s): %s", symbol, timeframe, url)

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
                        parsed = self._parse_kline_message(payload)
                        if parsed is not None:
                            await self._broadcast(stream_key, parsed)
            except asyncio.TimeoutError:
                logger.warning("Timed out reading Binance stream for %s (%s); reconnecting", symbol, timeframe)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Binance stream error for %s (%s): %s", symbol, timeframe, exc)
                await asyncio.sleep(2)

        logger.info("Stopped Binance stream for %s (%s)", symbol, timeframe)

    async def subscribe(self, symbol: str, timeframe: str, socket: WebSocket) -> None:
        symbol = symbol.upper()
        normalized_timeframe = self._normalize_timeframe(timeframe)
        stream_key = self._stream_key(symbol, normalized_timeframe)

        async with self._lock:
            connections = self._connections.setdefault(stream_key, set())
            connections.add(socket)

            task = self._stream_tasks.get(stream_key)
            if task is None or task.done():
                self._stream_tasks[stream_key] = asyncio.create_task(
                    self._consume_symbol_stream(symbol, normalized_timeframe)
                )

        await self._publish_connection_metric()

    async def unsubscribe(self, symbol: str, timeframe: str, socket: WebSocket) -> None:
        symbol = symbol.upper()
        normalized_timeframe = self._normalize_timeframe(timeframe)
        stream_key = self._stream_key(symbol, normalized_timeframe)

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
