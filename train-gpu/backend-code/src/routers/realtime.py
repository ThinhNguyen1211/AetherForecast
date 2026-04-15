from __future__ import annotations

import re

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from src.realtime.websocket import get_realtime_hub

router = APIRouter(tags=["realtime"])
SYMBOL_REGEX = re.compile(r"^[A-Z0-9]{2,20}$")
TIMEFRAME_REGEX = re.compile(r"^(1m|5m|15m|1h|4h|1d|1w)$")


@router.websocket("/ws/{symbol}")
async def stream_symbol_kline(websocket: WebSocket, symbol: str) -> None:
    normalized_symbol = symbol.upper().strip()
    if not SYMBOL_REGEX.match(normalized_symbol):
        await websocket.close(code=1008, reason="Invalid symbol format")
        return

    raw_timeframe = websocket.query_params.get("timeframe", "1m").strip().lower()
    if not TIMEFRAME_REGEX.match(raw_timeframe):
        await websocket.close(code=1008, reason="Invalid timeframe")
        return

    hub = get_realtime_hub()

    await websocket.accept()
    await hub.subscribe(normalized_symbol, raw_timeframe, websocket)
    await websocket.send_json(
        {
            "event": "subscribed",
            "symbol": normalized_symbol,
            "timeframe": raw_timeframe,
            "message": f"Streaming Binance kline updates for {normalized_symbol}",
        }
    )

    try:
        while True:
            message = await websocket.receive_text()
            if message.strip().lower() == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        pass
    finally:
        await hub.unsubscribe(normalized_symbol, raw_timeframe, websocket)
