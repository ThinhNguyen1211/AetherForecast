#!/usr/bin/env python3
"""cleanup_s3_models.py — Rigorous S3 model bucket cleanup.

Scans the model bucket for obsolete HuggingFace training checkpoints
(checkpoint-*) and orphaned temporary files (.json/.bin) that are NOT
part of the protected paths (manifest/latest.json, versions/*).

Usage:
    # Via environment variable
    export MODEL_BUCKET=aetherforecast-models-800762439372-ap-southeast-1
    python cleanup_s3_models.py

    # Via CLI argument
    python cleanup_s3_models.py --bucket aetherforecast-models-800762439372-ap-southeast-1

    # Dry-run (default) — shows what WOULD be deleted
    python cleanup_s3_models.py --bucket my-bucket

    # Actually delete — requires explicit --confirm flag
    python cleanup_s3_models.py --bucket my-bucket --confirm

    # Optionally scope to a prefix (e.g. only clean under chronos-v1/)
    python cleanup_s3_models.py --bucket my-bucket --prefix chronos-v1/ --confirm

Author: AetherForecast Data Engineering
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Iterator

import boto3
from botocore.exceptions import BotoCoreError, ClientError

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-5s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cleanup_s3_models")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_PATTERN = re.compile(r"(?:^|/)checkpoint-\d+/")

# Protected path segments — any key containing these is NEVER deleted.
PROTECTED_PATHS = (
    "manifest/latest.json",
    "/versions/",
)

# Orphan extensions to target (only outside protected paths)
ORPHAN_EXTENSIONS = (".json", ".bin")

# S3 DeleteObjects limit per call
_S3_DELETE_BATCH = 1000


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class CleanupStats:
    """Accumulates cleanup metrics."""

    checkpoint_keys: list[str] = field(default_factory=list)
    orphan_keys: list[str] = field(default_factory=list)
    checkpoint_bytes: int = 0
    orphan_bytes: int = 0
    errors: int = 0

    @property
    def total_keys(self) -> int:
        return len(self.checkpoint_keys) + len(self.orphan_keys)

    @property
    def total_bytes(self) -> int:
        return self.checkpoint_bytes + self.orphan_bytes


# ---------------------------------------------------------------------------
# S3 helpers
# ---------------------------------------------------------------------------

def _iter_all_objects(
    s3_client,
    bucket: str,
    prefix: str = "",
) -> Iterator[dict]:
    """Paginate through ALL objects in a bucket/prefix."""
    paginator = s3_client.get_paginator("list_objects_v2")
    kwargs = {"Bucket": bucket}
    if prefix:
        kwargs["Prefix"] = prefix

    for page in paginator.paginate(**kwargs):
        yield from page.get("Contents", [])


def _is_protected(key: str) -> bool:
    """Return True if the key belongs to a protected path."""
    for protected in PROTECTED_PATHS:
        if protected in key:
            return True
    return False


def _is_checkpoint(key: str) -> bool:
    """Return True if the key is inside a checkpoint-N/ folder."""
    return bool(CHECKPOINT_PATTERN.search(key))


def _is_orphan_temp(key: str) -> bool:
    """Return True if the key is an orphaned .json/.bin outside protected paths."""
    if _is_protected(key):
        return False
    # Must be a file, not a "directory marker"
    if key.endswith("/"):
        return False
    return any(key.endswith(ext) for ext in ORPHAN_EXTENSIONS)


def _format_bytes(num_bytes: int) -> str:
    """Human-readable byte string."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024  # type: ignore[assignment]
    return f"{num_bytes:.2f} PB"


def _delete_keys(
    s3_client,
    bucket: str,
    keys: list[str],
    dry_run: bool,
) -> int:
    """Delete keys in batches. Returns error count."""
    if not keys:
        return 0

    if dry_run:
        return 0

    errors = 0
    for i in range(0, len(keys), _S3_DELETE_BATCH):
        batch = keys[i : i + _S3_DELETE_BATCH]
        delete_payload = {"Objects": [{"Key": k} for k in batch], "Quiet": True}
        try:
            resp = s3_client.delete_objects(Bucket=bucket, Delete=delete_payload)
            batch_errors = resp.get("Errors", [])
            if batch_errors:
                for err in batch_errors:
                    logger.error(
                        "  ✗ Failed to delete %s: %s (%s)",
                        err.get("Key"),
                        err.get("Code"),
                        err.get("Message"),
                    )
                errors += len(batch_errors)
        except (BotoCoreError, ClientError) as exc:
            logger.error("  ✗ DeleteObjects batch failed: %s", exc)
            errors += len(batch)

    return errors


# ---------------------------------------------------------------------------
# Main scan & clean
# ---------------------------------------------------------------------------

def scan_bucket(
    s3_client,
    bucket: str,
    prefix: str = "",
) -> CleanupStats:
    """Scan the bucket and classify objects for cleanup."""
    stats = CleanupStats()

    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("Scanning s3://%s/%s ...", bucket, prefix)
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")

    total_scanned = 0

    for obj in _iter_all_objects(s3_client, bucket, prefix):
        key = obj["Key"]
        size = obj.get("Size", 0)
        total_scanned += 1

        # Rule 1: checkpoint-* folders → always delete
        if _is_checkpoint(key):
            stats.checkpoint_keys.append(key)
            stats.checkpoint_bytes += size
            continue

        # Rule 2: orphaned .json/.bin NOT in manifest/ or versions/ → delete
        if _is_orphan_temp(key):
            stats.orphan_keys.append(key)
            stats.orphan_bytes += size
            continue

    logger.info("Scanned %d total objects.", total_scanned)
    return stats


def execute_cleanup(
    s3_client,
    bucket: str,
    stats: CleanupStats,
    dry_run: bool = True,
) -> None:
    """Log findings and optionally delete."""
    mode_label = "DRY-RUN" if dry_run else "LIVE DELETE"

    # --- Report: Checkpoints ---
    logger.info("")
    logger.info("┌─── Checkpoint Folders (checkpoint-*) ─── [%s]", mode_label)
    if stats.checkpoint_keys:
        # Group by checkpoint folder for cleaner output
        checkpoint_folders: dict[str, list[str]] = {}
        for key in stats.checkpoint_keys:
            match = CHECKPOINT_PATTERN.search(key)
            if match:
                folder = key[: match.end()]
            else:
                folder = key
            checkpoint_folders.setdefault(folder, []).append(key)

        for folder, keys in sorted(checkpoint_folders.items()):
            logger.info("  🗑  %s (%d files)", folder, len(keys))
        logger.info(
            "  Total: %d files, %s",
            len(stats.checkpoint_keys),
            _format_bytes(stats.checkpoint_bytes),
        )
    else:
        logger.info("  ✓ No checkpoint folders found. Bucket is clean.")

    # --- Report: Orphaned temp files ---
    logger.info("")
    logger.info("┌─── Orphaned Temp Files (.json / .bin) ─── [%s]", mode_label)
    if stats.orphan_keys:
        for key in stats.orphan_keys[:50]:  # cap log output
            logger.info("  🗑  %s", key)
        if len(stats.orphan_keys) > 50:
            logger.info("  ... and %d more", len(stats.orphan_keys) - 50)
        logger.info(
            "  Total: %d files, %s",
            len(stats.orphan_keys),
            _format_bytes(stats.orphan_bytes),
        )
    else:
        logger.info("  ✓ No orphaned temp files found.")

    # --- Delete ---
    if stats.total_keys == 0:
        logger.info("")
        logger.info("✅ Nothing to clean up. Bucket is tidy!")
        return

    logger.info("")
    if dry_run:
        logger.info(
            "⚠️  DRY-RUN: Would delete %d objects freeing %s.",
            stats.total_keys,
            _format_bytes(stats.total_bytes),
        )
        logger.info("   Re-run with --confirm to actually delete.")
        return

    logger.info("🔥 DELETING %d objects (%s) ...", stats.total_keys, _format_bytes(stats.total_bytes))

    all_keys = stats.checkpoint_keys + stats.orphan_keys
    stats.errors = _delete_keys(s3_client, bucket, all_keys, dry_run=False)

    # --- Final Summary ---
    logger.info("")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("                   CLEANUP SUMMARY")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    logger.info("  Checkpoint files deleted:  %d (%s)", len(stats.checkpoint_keys), _format_bytes(stats.checkpoint_bytes))
    logger.info("  Orphan temp files deleted: %d (%s)", len(stats.orphan_keys), _format_bytes(stats.orphan_bytes))
    logger.info("  Total freed:              %s", _format_bytes(stats.total_bytes))
    if stats.errors:
        logger.warning("  ⚠ Errors encountered:     %d", stats.errors)
    else:
        logger.info("  ✅ All deletions successful.")
    logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Clean up obsolete ML model checkpoints and temp files from S3.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  # Dry-run (safe, shows what would be deleted)\n"
            "  python cleanup_s3_models.py --bucket my-models-bucket\n\n"
            "  # Actually delete\n"
            "  python cleanup_s3_models.py --bucket my-models-bucket --confirm\n\n"
            "  # Scope to a prefix\n"
            "  python cleanup_s3_models.py --bucket my-bucket --prefix chronos-v1/ --confirm\n"
        ),
    )
    parser.add_argument(
        "--bucket",
        default=os.environ.get("MODEL_BUCKET", ""),
        help="S3 bucket name. Falls back to MODEL_BUCKET env var.",
    )
    parser.add_argument(
        "--prefix",
        default="",
        help="Optional S3 key prefix to scope the scan (e.g. 'chronos-v1/').",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Actually delete objects. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--region",
        default=os.environ.get("AWS_REGION", "ap-southeast-1"),
        help="AWS region (default: AWS_REGION env or ap-southeast-1).",
    )

    args = parser.parse_args()

    if not args.bucket:
        logger.error("No bucket specified. Use --bucket or set MODEL_BUCKET env var.")
        sys.exit(1)

    dry_run = not args.confirm

    logger.info("╔═══════════════════════════════════════════════════════╗")
    logger.info("║   AetherForecast S3 Model Bucket Cleanup             ║")
    logger.info("╠═══════════════════════════════════════════════════════╣")
    logger.info("║  Bucket:  %-43s ║", args.bucket)
    logger.info("║  Prefix:  %-43s ║", args.prefix or "(entire bucket)")
    logger.info("║  Region:  %-43s ║", args.region)
    logger.info("║  Mode:    %-43s ║", "🔥 LIVE DELETE" if not dry_run else "👀 DRY-RUN (safe)")
    logger.info("╚═══════════════════════════════════════════════════════╝")
    logger.info("")
    logger.info("Protected paths (will NEVER be deleted):")
    for p in PROTECTED_PATHS:
        logger.info("  🔒 *%s*", p)
    logger.info("")

    try:
        s3_client = boto3.client("s3", region_name=args.region)
        stats = scan_bucket(s3_client, args.bucket, args.prefix)
        execute_cleanup(s3_client, args.bucket, stats, dry_run=dry_run)
    except (BotoCoreError, ClientError) as exc:
        logger.error("AWS S3 error: %s", exc)
        sys.exit(1)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user.")
        sys.exit(130)


if __name__ == "__main__":
    main()
