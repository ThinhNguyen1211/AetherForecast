from __future__ import annotations

from datetime import datetime, timezone
import json
import logging

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

    root_bucket, _ = parse_s3_uri(model_root_s3_uri)
    # Per MLOps requirement the active-model manifest lives at the bucket root.
    manifest_key = "manifest/latest.json"

    payload = {
        "active_model_s3_uri": trained_version_s3_uri,
        "promoted_at": datetime.now(timezone.utc).isoformat(),
        "status": "latest",
        "project": "AetherForecast",
    }

    s3_client.put_object(
        Bucket=root_bucket,
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

    logger.info("Promoted model version %s via manifest s3://%s/%s", trained_version_s3_uri, root_bucket, manifest_key)
    return f"s3://{root_bucket}/{manifest_key}"
