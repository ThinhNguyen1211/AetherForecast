from __future__ import annotations

from functools import lru_cache
import logging
from typing import Mapping, Sequence

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from src.core.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache
def _cloudwatch_client():
    settings = get_settings()
    client_config = Config(
        retries={"max_attempts": 2, "mode": "standard"},
        connect_timeout=1,
        read_timeout=2,
        tcp_keepalive=True,
    )
    return boto3.client("cloudwatch", region_name=settings.aws_region, config=client_config)


def _build_dimensions(dimensions: Mapping[str, str] | None) -> list[dict[str, str]]:
    if not dimensions:
        return []

    return [
        {"Name": key, "Value": val}
        for key, val in dimensions.items()
        if key and val
    ]


def put_custom_metrics(
    metrics: Sequence[Mapping[str, object]],
    namespace: str = "AetherForecast/Pipeline",
    dimensions: Mapping[str, str] | None = None,
) -> None:
    metric_dimensions = _build_dimensions(dimensions)

    metric_data: list[dict[str, object]] = []
    for metric in metrics:
        metric_name = str(metric.get("MetricName", "")).strip()
        metric_value = metric.get("Value")
        metric_unit = str(metric.get("Unit", "Count"))

        if not metric_name or metric_value is None:
            continue

        try:
            value = float(metric_value)
        except (TypeError, ValueError):
            continue

        metric_data.append(
            {
                "MetricName": metric_name,
                "Dimensions": metric_dimensions,
                "Unit": metric_unit,
                "Value": value,
            }
        )

    if not metric_data:
        return

    try:
        for index in range(0, len(metric_data), 20):
            _cloudwatch_client().put_metric_data(
                Namespace=namespace,
                MetricData=metric_data[index : index + 20],
            )
    except (BotoCoreError, ClientError, ValueError) as exc:
        logger.warning("Unable to put custom CloudWatch metrics for namespace %s: %s", namespace, exc)


def put_custom_metric(
    metric_name: str,
    value: float,
    namespace: str = "AetherForecast/Pipeline",
    unit: str = "Count",
    dimensions: Mapping[str, str] | None = None,
) -> None:
    put_custom_metrics(
        metrics=[
            {
                "MetricName": metric_name,
                "Value": value,
                "Unit": unit,
            }
        ],
        namespace=namespace,
        dimensions=dimensions,
    )
