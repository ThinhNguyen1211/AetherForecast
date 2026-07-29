from datetime import UTC, datetime

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": "aetherforecast-backend",
        "timestamp": datetime.now(UTC).isoformat(),
    }
