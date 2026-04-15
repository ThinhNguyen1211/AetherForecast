from datetime import datetime

from fastapi import APIRouter, Depends, Query

from src.dependencies.cognito import require_authenticated_user
from src.dependencies.s3_client import S3ParquetClient, get_s3_parquet_client
from src.ml.schemas import ChartResponse, SupportedTimeframe

router = APIRouter(tags=["chart"])


@router.get("/chart/{symbol}", response_model=ChartResponse)
def get_chart(
    symbol: str,
    timeframe: SupportedTimeframe = Query(default="1h"),
    limit: int = Query(default=1200, ge=200, le=5000),
    from_timestamp: datetime | None = Query(default=None),
    _claims: dict = Depends(require_authenticated_user),
    s3_client: S3ParquetClient = Depends(get_s3_parquet_client),
) -> ChartResponse:
    candles = s3_client.fetch_chart_points(
        symbol=symbol,
        timeframe=timeframe,
        limit=limit,
        from_timestamp=from_timestamp,
    )
    return ChartResponse(symbol=symbol.upper(), timeframe=timeframe, candles=candles)
