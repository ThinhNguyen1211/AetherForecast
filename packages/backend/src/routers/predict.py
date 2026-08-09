import asyncio
import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.dependencies.cognito import require_authenticated_user
from src.dependencies.s3_client import S3ParquetClient, get_s3_parquet_client
from src.ml.inference import ForecastInferenceService, get_inference_service
from src.ml.schemas import Candle, PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])
logger = logging.getLogger(__name__)

_BASE_HISTORY_BY_TIMEFRAME = {
    "1m": 1500,
    "5m": 1650,
    "15m": 1900,
    "1h": 2300,
    "4h": 2700,
    "1d": 3200,
    "1w": 2800,
}


def _resolve_history_limit(request: PredictRequest) -> int:
    base_limit = _BASE_HISTORY_BY_TIMEFRAME.get(request.timeframe, 2000)
    horizon_boost = max(0, int(request.horizon) - 24) * 12
    return max(1200, min(4200, base_limit + horizon_boost))


@router.post("/predict", response_model=PredictResponse)
async def predict_price(
    request: PredictRequest,
    _claims: dict = Depends(require_authenticated_user),
    inference_service: ForecastInferenceService = Depends(get_inference_service),
    s3_client: S3ParquetClient = Depends(get_s3_parquet_client),
) -> PredictResponse:
    try:
        # Always rebuild model input from live backend data sources.
        history_limit = _resolve_history_limit(request)
        # Blocking network call — offloaded to a thread so it doesn't block
        # the event loop now that this route is async (async is required for
        # predict_coalesced()'s asyncio.Lock/Task coordination below to work
        # correctly; those primitives need to run on the loop itself, not a
        # plain sync-route worker thread).
        backend_candles = await asyncio.to_thread(
            s3_client.fetch_chart_points,
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=history_limit,
            use_cache=False,
        )
        if len(backend_candles) < 20:
            raise ValueError("Insufficient real chart candles available for robust Chronos inference")

        real_request = PredictRequest(
            symbol=request.symbol,
            timeframe=request.timeframe,
            latest_candles=[Candle(**item) for item in backend_candles],
            horizon=request.horizon,
            quantiles=request.quantiles,
            sentiment_score=None,
        )

        # Request coalescing: a burst of concurrent requests for the same
        # symbol/timeframe/horizon/quantiles shares ONE Chronos-2 forward pass
        # instead of one per request. See ForecastInferenceService.predict_coalesced().
        return await inference_service.predict_coalesced(real_request)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)) from exc
    except RuntimeError as exc:
        logger.exception("Inference runtime error")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected inference error")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Unexpected error while generating forecast",
        ) from exc
