from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

import boto3

from ml.training.checkpoint import parse_s3_uri
from src.core.metrics import put_custom_metric

logger = logging.getLogger(__name__)


def promote_model_version(
    model_root_s3_uri: str,
    trained_version_s3_uri: str,
    aws_region: str,
    endpoint_url: str | None,
) -> str:
    s3_client = boto3.client("s3", region_name=aws_region, endpoint_url=endpoint_url)

    bucket, prefix = parse_s3_uri(model_root_s3_uri)
    # Must match model_loader.py's _resolve_active_model_uri_from_manifest(), which
    # reads the manifest from "{prefix}/manifest/latest.json" under the model root
    # (e.g. chronos-v1/model/manifest/latest.json) — not the bucket root.
    manifest_key = f"{prefix.rstrip('/')}/manifest/latest.json".strip("/")

    payload = {
        "active_model_s3_uri": trained_version_s3_uri,
        "promoted_at": datetime.now(UTC).isoformat(),
        "status": "latest",
        "project": "AetherForecast",
    }

    s3_client.put_object(
        Bucket=bucket,
        Key=manifest_key,
        Body=json.dumps(payload, indent=2).encode("utf-8"),
        ContentType="application/json",
    )

    try:
        put_custom_metric(
            metric_name="ModelPromotionSuccess",
            value=1,
            dimensions={"Pipeline": "model-promotion"},
        )
    except Exception as exc:
        logger.warning("Failed to emit CloudWatch metric: %s", exc)

    logger.info(
        "Promoted model version %s via manifest s3://%s/%s",
        trained_version_s3_uri,
        bucket,
        manifest_key,
    )
    return f"s3://{bucket}/{manifest_key}"
