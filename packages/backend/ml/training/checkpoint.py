from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Iterable

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from transformers import TrainerCallback, TrainerControl, TrainerState

logger = logging.getLogger(__name__)
CHECKPOINT_PATTERN = re.compile(r"checkpoint-(\d+)")


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    without_scheme = s3_uri.replace("s3://", "", 1)
    bucket, _, key_prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI bucket: {s3_uri}")

    return bucket, key_prefix.rstrip("/")


def _iter_files(root_dir: Path) -> Iterable[Path]:
    for path in root_dir.rglob("*"):
        if path.is_file():
            yield path


@dataclass
class S3CheckpointManager:
    s3_uri: str
    aws_region: str
    endpoint_url: str | None = None

    def __post_init__(self) -> None:
        self.bucket, self.prefix = parse_s3_uri(self.s3_uri)
        self.s3_client = boto3.client("s3", region_name=self.aws_region, endpoint_url=self.endpoint_url)

    def _list_keys(self, prefix: str) -> list[str]:
        paginator = self.s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=self.bucket, Prefix=prefix)

        keys: list[str] = []
        for page in pages:
            keys.extend([item["Key"] for item in page.get("Contents", [])])
        return keys

    def find_latest_checkpoint_prefix(self) -> str | None:
        keys = self._list_keys(self.prefix)
        latest_step = -1
        latest_prefix = None

        for key in keys:
            match = CHECKPOINT_PATTERN.search(key)
            if not match:
                continue
            step = int(match.group(1))
            if step > latest_step:
                latest_step = step
                latest_prefix = key[: key.find(match.group(0)) + len(match.group(0))]

        return latest_prefix

    def download_latest_checkpoint(self, local_base_dir: Path) -> Path | None:
        latest_prefix = self.find_latest_checkpoint_prefix()
        if latest_prefix is None:
            logger.info("No checkpoint found under %s", self.s3_uri)
            return None

        checkpoint_name = latest_prefix.rstrip("/").split("/")[-1]
        local_checkpoint_dir = local_base_dir / checkpoint_name
        local_checkpoint_dir.mkdir(parents=True, exist_ok=True)

        keys = self._list_keys(latest_prefix)
        for key in keys:
            relative = key[len(latest_prefix) :].lstrip("/")
            if not relative:
                continue
            local_path = local_checkpoint_dir / relative
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self.s3_client.download_file(self.bucket, key, str(local_path))

        logger.info("Downloaded checkpoint %s to %s", latest_prefix, local_checkpoint_dir)
        return local_checkpoint_dir

    def upload_checkpoint_directory(self, local_checkpoint_dir: str) -> None:
        root = Path(local_checkpoint_dir)
        if not root.exists():
            logger.warning("Local checkpoint directory %s does not exist", root)
            return

        checkpoint_prefix = f"{self.prefix}/{root.name}".strip("/")
        for file_path in _iter_files(root):
            relative = file_path.relative_to(root).as_posix()
            key = f"{checkpoint_prefix}/{relative}".strip("/")
            self.s3_client.upload_file(str(file_path), self.bucket, key)

        logger.info("Uploaded checkpoint %s to s3://%s/%s", root, self.bucket, checkpoint_prefix)

    def upload_directory(self, local_dir: str, destination_s3_uri: str) -> None:
        dest_bucket, dest_prefix = parse_s3_uri(destination_s3_uri)
        local_path = Path(local_dir)
        if not local_path.exists():
            raise FileNotFoundError(f"Local directory does not exist: {local_dir}")

        for file_path in _iter_files(local_path):
            relative = file_path.relative_to(local_path).as_posix()
            key = f"{dest_prefix}/{relative}".strip("/") if dest_prefix else relative
            self.s3_client.upload_file(str(file_path), dest_bucket, key)

        logger.info("Uploaded directory %s to %s", local_dir, destination_s3_uri)


class S3CheckpointCallback(TrainerCallback):
    def __init__(self, manager: S3CheckpointManager) -> None:
        self.manager = manager

    def on_save(
        self,
        args,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ) -> TrainerControl:
        checkpoint_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
        try:
            self.manager.upload_checkpoint_directory(str(checkpoint_dir))
        except (BotoCoreError, ClientError, OSError) as exc:
            logger.warning("Failed to upload checkpoint to S3: %s", exc)
        return control
