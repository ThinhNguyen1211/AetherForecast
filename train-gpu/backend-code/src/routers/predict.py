import logging

from fastapi import APIRouter, Depends, HTTPException, status

from src.dependencies.cognito import require_authenticated_user
from src.dependencies.s3_client import S3ParquetClient, get_s3_parquet_client
from src.ml.inference import ForecastInferenceService, get_inference_service
from src.ml.schemas import Candle, PredictRequest, PredictResponse

router = APIRouter(tags=["predict"])
logger = logging.getLogger(__name__)

_BASE_HISTORY_BY_TIMEFRAME = {
    "1m": 1200,
    "5m": 1250,
    "15m": 1350,
    "1h": 1500,
    "4h": 1700,
    "1d": 1800,
    "1w": 1300,
}


def _resolve_history_limit(request: PredictRequest) -> int:
    base_limit = _BASE_HISTORY_BY_TIMEFRAME.get(request.timeframe, 1400)
    horizon_boost = max(0, int(request.horizon) - 24) * 8
    return max(700, min(2200, base_limit + horizon_boost))


@router.post("/predict", response_model=PredictResponse)
def predict_price(
    request: PredictRequest,
    _claims: dict = Depends(require_authenticated_user),
    inference_service: ForecastInferenceService = Depends(get_inference_service),
    s3_client: S3ParquetClient = Depends(get_s3_parquet_client),
) -> PredictResponse:
    try:
        # Always rebuild model input from live backend data sources.
        history_limit = _resolve_history_limit(request)
        backend_candles = s3_client.fetch_chart_points(
            symbol=request.symbol,
            timeframe=request.timeframe,
            limit=history_limit,
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

        return inference_service.predict(real_request)
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
