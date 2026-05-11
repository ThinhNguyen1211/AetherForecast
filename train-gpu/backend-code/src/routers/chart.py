import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Query

from src.dependencies.cognito import require_authenticated_user
from src.dependencies.s3_client import S3ParquetClient, get_s3_parquet_client
from src.ml.schemas import ChartResponse, SupportedTimeframe

router = APIRouter(tags=["chart"])
logger = logging.getLogger(__name__)


@router.get("/chart/{symbol}", response_model=ChartResponse)
def get_chart(
    symbol: str,
    timeframe: SupportedTimeframe = Query(default="1h"),
    limit: int = Query(default=800, ge=200, le=5000),
    from_timestamp: datetime | None = Query(default=None),
    _claims: dict = Depends(require_authenticated_user),
    s3_client: S3ParquetClient = Depends(get_s3_parquet_client),
) -> ChartResponse:
    if from_timestamp is not None:
        # Lazy-load: user scrolled history → use S3 Parquet for deeper data.
        candles = s3_client.fetch_chart_points(
            symbol=symbol,
            timeframe=timeframe,
            limit=limit,
            from_timestamp=from_timestamp,
        )
    else:
        # Initial load: Binance REST directly, capped at 1000 for speed.
        capped_limit = min(limit, 1000)
        candles = s3_client.fetch_from_binance_rest(
            symbol=symbol.upper(),
            timeframe=timeframe,
            limit=capped_limit,
        )

    return ChartResponse(symbol=symbol.upper(), timeframe=timeframe, candles=candles)

