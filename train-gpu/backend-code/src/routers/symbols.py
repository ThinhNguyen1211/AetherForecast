from fastapi import APIRouter, Depends
import httpx

from src.core.config import get_settings
from src.dependencies.cognito import require_authenticated_user

router = APIRouter(tags=["symbols"])

STATIC_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "ADAUSDT",
    "SJCXAU",
]


@router.get("/symbols")
async def get_symbols(_claims: dict = Depends(require_authenticated_user)) -> dict:
    settings = get_settings()

    if settings.symbols_source.lower() == "binance":
        try:
            async with httpx.AsyncClient(timeout=8) as client:
                response = await client.get(f"{settings.binance_base_url}/api/v3/exchangeInfo")
                response.raise_for_status()
                payload = response.json()

            symbols = [
                item["symbol"]
                for item in payload.get("symbols", [])
                if item.get("status") == "TRADING"
                and item.get("quoteAsset") in {"USDT", "BUSD", "FDUSD"}
            ]
            symbols.sort()
            return {"count": len(symbols), "symbols": symbols}
        except Exception:
            return {"count": len(STATIC_SYMBOLS), "symbols": STATIC_SYMBOLS}

    return {"count": len(STATIC_SYMBOLS), "symbols": STATIC_SYMBOLS}
