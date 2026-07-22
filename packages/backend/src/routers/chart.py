import logging
from datetime import datetime, timezone

import httpx
from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.core.config import get_settings
from src.dependencies.cognito import require_authenticated_user
from src.ml.schemas import Candle, ChartResponse, SupportedTimeframe, normalize_and_validate_symbol

router = APIRouter(tags=["chart"])
logger = logging.getLogger(__name__)


@router.get("/chart/{symbol}", response_model=ChartResponse)
async def get_chart(
    symbol: str,
    timeframe: SupportedTimeframe = Query(default="1h"),
    limit: int = Query(default=800, ge=200, le=5000),
    from_timestamp: datetime | None = Query(default=None),
    _claims: dict = Depends(require_authenticated_user),
) -> ChartResponse:
    try:
        symbol = normalize_and_validate_symbol(symbol)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc

    settings = get_settings()
    # Binance limit max is 1000
    capped_limit = min(limit, 1000)

    params: dict[str, str | int] = {
        "symbol": symbol,
        "interval": timeframe,
        "limit": capped_limit,
    }
    if from_timestamp is not None:
        params["endTime"] = int(from_timestamp.timestamp() * 1000)

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{settings.binance_base_url}/api/v3/klines", params=params)
            response.raise_for_status()
            data = response.json()
    except (httpx.TimeoutException, httpx.ConnectError) as exc:
        logger.error("Binance API connection error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Upstream data provider timeout",
        ) from exc
    except httpx.HTTPStatusError as exc:
        logger.error("Binance API HTTP error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Upstream data provider returned an error",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected error fetching from Binance")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error fetching chart data",
        ) from exc

    candles = []
    for k in data:
        try:
            candles.append(
                Candle(
                    timestamp=datetime.fromtimestamp(k[0] / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                )
            )
        except (IndexError, ValueError, TypeError):
            continue

    return ChartResponse(symbol=symbol.upper(), timeframe=timeframe, candles=candles)
