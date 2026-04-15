from contextlib import asynccontextmanager
from time import perf_counter

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from src.core.config import get_settings
from src.core.logging import configure_logging
from src.core.metrics import put_custom_metrics
from src.realtime.websocket import get_realtime_hub
from src.routers import chart, health, predict, realtime, symbols

settings = get_settings()
configure_logging(settings.log_level)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    hub = get_realtime_hub()
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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
