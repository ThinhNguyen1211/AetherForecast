from datetime import datetime, timezone

from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": "aetherforecast-backend",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
