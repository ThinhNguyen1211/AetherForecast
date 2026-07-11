"""AetherForecast Data Bucket Cleanup.

Permanently deletes ALL objects under the `market/klines/` and `_metadata/`
prefixes in the data bucket.  This wipes stale/mock data so the real
15-minute ingestion cronjob starts with a clean slate.

Usage:
    # Dry-run (default — shows what would be deleted)
    python scripts/cleanup_s3_data.py --bucket aetherforecast-data-800762439372-ap-southeast-1

    # Scope to a single prefix
    python scripts/cleanup_s3_data.py --bucket <BUCKET> --prefix market/klines/

    # Actually delete
    python scripts/cleanup_s3_data.py --bucket <BUCKET> --confirm
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import boto3
from botocore.exceptions import ClientError

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Prefixes to wipe
DEFAULT_PREFIXES = ["market/klines/", "_metadata/"]


def _collect_objects(s3_client, bucket: str, prefix: str) -> list[dict]:
    """List all objects under a prefix using paginator."""
    objects: list[dict] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            objects.append({"Key": obj["Key"], "Size": obj.get("Size", 0)})
    return objects


def _delete_objects(s3_client, bucket: str, keys: list[str], dry_run: bool) -> int:
    """Batch-delete objects (1000 per request per S3 API limit)."""
    if dry_run or not keys:
        return 0

    deleted = 0
    for i in range(0, len(keys), 1000):
        batch = [{"Key": k} for k in keys[i : i + 1000]]
        try:
            resp = s3_client.delete_objects(
                Bucket=bucket, Delete={"Objects": batch, "Quiet": True}
            )
            errors = resp.get("Errors", [])
            if errors:
                for err in errors:
                    logger.error("  ❌ Failed to delete %s: %s", err["Key"], err["Message"])
            deleted += len(batch) - len(errors)
        except ClientError as exc:
            logger.error("  ❌ S3 DeleteObjects failed: %s", exc)
    return deleted


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean AetherForecast data bucket")
    parser.add_argument(
        "--bucket",
        default=os.getenv("DATA_BUCKET", os.getenv("DATA_S3_BUCKET", "")),
        help="S3 bucket name (or set DATA_BUCKET env var)",
    )
    parser.add_argument(
        "--prefix",
        action="append",
        help="Override prefixes to delete (can be specified multiple times)",
    )
    parser.add_argument(
        "--region",
        default=os.getenv("AWS_REGION", "ap-southeast-1"),
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete (default is dry-run)",
    )
    args = parser.parse_args()

    if not args.bucket:
        logger.error("❌ No bucket specified. Use --bucket or set DATA_BUCKET env var.")
        sys.exit(1)

    prefixes = args.prefix or DEFAULT_PREFIXES
    dry_run = not args.confirm

    logger.info("=" * 55)
    logger.info("  AetherForecast Data Bucket Cleanup")
    logger.info("  Bucket:   %s", args.bucket)
    logger.info("  Prefixes: %s", prefixes)
    logger.info("  Mode:     %s", "🔥 LIVE DELETE" if not dry_run else "👀 DRY-RUN (no deletions)")
    logger.info("=" * 55)

    session = boto3.Session(region_name=args.region)
    s3 = session.client("s3")

    total_objects = 0
    total_bytes = 0
    total_deleted = 0

    for prefix in prefixes:
        logger.info("")
        logger.info("Scanning s3://%s/%s ...", args.bucket, prefix)

        objects = _collect_objects(s3, args.bucket, prefix)
        prefix_bytes = sum(o["Size"] for o in objects)
        total_objects += len(objects)
        total_bytes += prefix_bytes

        if not objects:
            logger.info("  (empty — nothing to delete)")
            continue

        logger.info(
            "  Found %d objects (%.2f MB)",
            len(objects),
            prefix_bytes / (1024 * 1024),
        )

        # Log first 10 objects as examples
        for obj in objects[:10]:
            tag = "🗑" if not dry_run else "👁"
            logger.info("  %s  %s (%.1f KB)", tag, obj["Key"], obj["Size"] / 1024)
        if len(objects) > 10:
            logger.info("  ... and %d more", len(objects) - 10)

        if not dry_run:
            keys = [o["Key"] for o in objects]
            deleted = _delete_objects(s3, args.bucket, keys, dry_run)
            total_deleted += deleted
            logger.info("  ✅ Deleted %d objects", deleted)
        else:
            logger.info("  (dry-run — skipping deletion)")

    logger.info("")
    logger.info("━" * 55)
    logger.info("  SUMMARY")
    logger.info("━" * 55)
    logger.info("  Objects scanned:  %d", total_objects)
    logger.info("  Total size:       %.2f MB", total_bytes / (1024 * 1024))
    if dry_run:
        logger.info("  Mode:             DRY-RUN (nothing deleted)")
        logger.info("  ➡ Run with --confirm to delete")
    else:
        logger.info("  Objects deleted:  %d", total_deleted)
        logger.info("  ✅ Cleanup complete.")
    logger.info("━" * 55)


if __name__ == "__main__":
    main()
