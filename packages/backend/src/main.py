import logging
import traceback
from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from src.core.config import get_settings
from src.core.logging import configure_logging
from src.core.metrics import put_custom_metrics
from src.ml.inference import get_inference_service
from src.realtime.websocket import get_realtime_hub
from src.routers import ai_council, chart, health, predict, realtime, symbols

settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger(__name__)

limiter = Limiter(key_func=get_remote_address)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    hub = get_realtime_hub()

    # Preload the Chronos-2 model (S3 download + PyTorch load, ~20s) during
    # boot instead of lazily on the first /predict request, so the first user
    # doesn't pay the cold-start cost. Runs synchronously and blocks startup
    # on purpose — the app shouldn't accept traffic before the model is ready
    # anyway. A failure here is logged but does not prevent the app from
    # starting: /predict will simply retry the (now-uncached) load on its
    # first real request and surface any persistent error there instead.
    try:
        get_inference_service().warm_up_model()
        logger.info("Chronos-2 model preloaded successfully during startup")
    except Exception:
        logger.exception("Model warm-up at startup failed; will lazy-load on first request")

    try:
        yield
    finally:
        await hub.close()


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Explicit production origins first. The env var CORS_ORIGINS can still be used
# to add extra origins (e.g., preview deploys), but the canonical domains are
# always allowed so a misconfigured deployment parameter cannot break production.
_origins = [
    "http://localhost:5173",
    "http://localhost:3000",
    "https://aetherforcast.io.vn",
    "https://www.aetherforcast.io.vn",
]
if settings.cors_origins and settings.cors_origins != ["*"]:
    _origins = list(dict.fromkeys(_origins + settings.cors_origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(_request: Request, exc: Exception):
    """Catch-all handler so uncaught 500s still carry CORS headers.

    Without this, FastAPI's default Server Error response bypasses the
    CORSMiddleware, causing browsers to report a fake CORS failure instead of
    the real 500 / error body.
    """
    logger.exception("Unhandled exception in request")
    error_tb = traceback.format_exc().replace("\n", " | ")
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_type": type(exc).__name__,
            "traceback": error_tb,
        },
    )


@app.middleware("http")
async def collect_api_metrics(request: Request, call_next):
    started_at = perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration_ms = (perf_counter() - started_at) * 1000
        api_dimensions = {"Service": "backend"}

        metric_batch = [
            {
                "MetricName": "ApiRequests",
                "Value": 1,
                "Unit": "Count",
            },
            {
                "MetricName": "ApiLatencyMs",
                "Value": duration_ms,
                "Unit": "Milliseconds",
            },
        ]

        if status_code >= 500:
            metric_batch.append(
                {
                    "MetricName": "Api5xx",
                    "Value": 1,
                    "Unit": "Count",
                }
            )

        put_custom_metrics(
            metrics=metric_batch,
            namespace="AetherForecast/API",
            dimensions=api_dimensions,
        )

app.include_router(health.router)
app.include_router(symbols.router)
app.include_router(chart.router)
app.include_router(predict.router)
app.include_router(realtime.router)
app.include_router(ai_council.router)
