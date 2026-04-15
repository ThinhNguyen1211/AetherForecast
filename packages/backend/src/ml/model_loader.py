from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import json
import logging
from pathlib import Path
import shutil
import threading
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer

try:
    from chronos.base import BaseChronosPipeline
except Exception:  # pragma: no cover
    BaseChronosPipeline = None

try:
    from peft import PeftModel
except Exception:  # pragma: no cover
    PeftModel = None

logger = logging.getLogger(__name__)

_model_load_lock = threading.RLock()


@dataclass(frozen=True)
class ResolvedModelSource:
    requested_source: str
    effective_source: str
    load_path: str


@dataclass
class InMemoryForecastModel:
    model: Any
    tokenizer: Any | None


@dataclass
class LoadedForecastModel:
    model: Any
    tokenizer: Any | None
    model_name: str
    model_version: str
    requested_source: str
    effective_source: str
    load_path: str
    torch_dtype: str


def _parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")

    without_scheme = s3_uri.replace("s3://", "", 1)
    bucket, _, key_prefix = without_scheme.partition("/")
    if not bucket:
        raise ValueError(f"Invalid S3 URI bucket: {s3_uri}")

    return bucket, key_prefix.rstrip("/")


def _s3_uri_to_local_dir(base_dir: str, s3_uri: str) -> Path:
    digest = hashlib.sha256(s3_uri.encode("utf-8")).hexdigest()[:16]
    return Path(base_dir) / digest


def _resolve_active_model_uri_from_manifest(
    model_s3_uri: str,
    aws_region: str,
    endpoint_url: str | None,
) -> str:
    bucket, prefix = _parse_s3_uri(model_s3_uri)
    manifest_key = f"{prefix.rstrip('/')}/manifest/latest.json".strip("/")

    s3_client = boto3.client("s3", region_name=aws_region, endpoint_url=endpoint_url)
    try:
        response = s3_client.get_object(Bucket=bucket, Key=manifest_key)
        payload = json.loads(response["Body"].read().decode("utf-8"))
        promoted_uri = str(payload.get("active_model_s3_uri", "")).strip()
        if promoted_uri.startswith("s3://"):
            logger.info("Resolved promoted model URI from manifest: %s", promoted_uri)
            return promoted_uri
    except ClientError:
        logger.info("Manifest not found at s3://%s/%s. Using base URI.", bucket, manifest_key)
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        logger.warning("Invalid manifest payload at s3://%s/%s: %s", bucket, manifest_key, exc)

    return model_s3_uri


def _download_s3_prefix(
    s3_uri: str,
    local_dir: Path,
    aws_region: str,
    endpoint_url: str | None,
) -> Path:
    bucket, prefix = _parse_s3_uri(s3_uri)
    s3_client = boto3.client("s3", region_name=aws_region, endpoint_url=endpoint_url)

    local_dir.mkdir(parents=True, exist_ok=True)

    marker = local_dir / ".sync_complete"
    metadata_file = local_dir / ".sync_meta.json"

    paginator = s3_client.get_paginator("list_objects_v2")
    pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

    objects: list[dict[str, Any]] = []
    for page in pages:
        objects.extend(page.get("Contents", []))

    if not objects:
        raise FileNotFoundError(f"No model artifacts found at {s3_uri}")

    fingerprint_hasher = hashlib.sha256()
    for item in sorted(objects, key=lambda row: str(row.get("Key", ""))):
        key = str(item.get("Key", ""))
        etag = str(item.get("ETag", ""))
        size = str(item.get("Size", ""))
        last_modified = item.get("LastModified")
        last_modified_value = (
            last_modified.isoformat() if hasattr(last_modified, "isoformat") else str(last_modified)
        )
        fingerprint_hasher.update(f"{key}|{etag}|{size}|{last_modified_value}\n".encode("utf-8"))
    remote_fingerprint = fingerprint_hasher.hexdigest()

    if marker.exists() and metadata_file.exists() and any(local_dir.iterdir()):
        try:
            cached_meta = json.loads(metadata_file.read_text(encoding="utf-8"))
            cached_fingerprint = str(cached_meta.get("fingerprint", "")).strip()
            if cached_fingerprint and cached_fingerprint == remote_fingerprint:
                logger.info("Using cached model directory at %s", local_dir)
                return local_dir
        except Exception as exc:
            logger.warning("Unable to parse model sync metadata at %s: %s", metadata_file, exc)

    for child in local_dir.iterdir():
        if child.is_dir():
            shutil.rmtree(child, ignore_errors=True)
        else:
            child.unlink(missing_ok=True)

    downloaded = 0
    for item in objects:
        key = str(item["Key"])
        relative = key[len(prefix) :].lstrip("/") if prefix else key
        if not relative:
            continue

        local_path = local_dir / relative
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, key, str(local_path))
        downloaded += 1

    metadata_file.write_text(
        json.dumps(
            {
                "source": s3_uri,
                "fingerprint": remote_fingerprint,
                "file_count": downloaded,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )

    marker.write_text("ok", encoding="utf-8")
    logger.info("Downloaded %s model files from %s", downloaded, s3_uri)
    return local_dir


def _resolve_model_source(
    model_s3_uri: str,
    model_cache_dir: str,
    aws_region: str,
    endpoint_url: str | None,
) -> ResolvedModelSource:
    if model_s3_uri.startswith("s3://"):
        effective_uri = _resolve_active_model_uri_from_manifest(
            model_s3_uri,
            aws_region,
            endpoint_url,
        )
        local_dir = _s3_uri_to_local_dir(model_cache_dir, effective_uri)
        downloaded_dir = _download_s3_prefix(effective_uri, local_dir, aws_region, endpoint_url)
        return ResolvedModelSource(
            requested_source=model_s3_uri,
            effective_source=effective_uri,
            load_path=str(downloaded_dir),
        )

    return ResolvedModelSource(
        requested_source=model_s3_uri,
        effective_source=model_s3_uri,
        load_path=model_s3_uri,
    )


def _extract_model_name_and_version(source: str) -> tuple[str, str]:
    source_id = source.rstrip("/")
    if source_id.startswith("s3://"):
        parts = source_id.split("/")
        if len(parts) >= 2:
            return parts[-2], parts[-1]
        return "chronos", "v1"

    if source_id:
        parts = source_id.split("/")
        return parts[-1], "hf"

    return "chronos", "unknown"


def get_loaded_forecasting_model(
    model_s3_uri: str,
    fallback_model_id: str,
    tokenizer_fallback_id: str,
    model_cache_dir: str,
    aws_region: str,
    endpoint_url: str | None,
    torch_dtype: str = "auto",
    require_s3_model: bool = True,
) -> LoadedForecastModel:
    configured_model_source = (model_s3_uri or "").strip()

    if require_s3_model:
        if not configured_model_source.startswith("s3://"):
            raise ValueError(
                "REQUIRE_S3_MODEL is enabled, but MODEL_S3_URI is not an s3:// URI."
            )
        model_source = configured_model_source
    else:
        model_source = configured_model_source or fallback_model_id

    if not model_source:
        raise ValueError("No model source configured. Provide MODEL_S3_URI.")

    try:
        resolved = _resolve_model_source(
            model_source,
            model_cache_dir,
            aws_region,
            endpoint_url,
        )
    except (ValueError, FileNotFoundError, BotoCoreError, ClientError) as exc:
        logger.exception("Unable to sync Chronos model from source %s", model_source)
        raise RuntimeError(f"Unable to load Chronos model from {model_source}: {exc}") from exc

    base_model_source_candidate: str | None = None
    if resolved.requested_source.startswith("s3://") and resolved.requested_source != resolved.effective_source:
        try:
            requested_local_dir = _s3_uri_to_local_dir(model_cache_dir, resolved.requested_source)
            requested_download_dir = _download_s3_prefix(
                resolved.requested_source,
                requested_local_dir,
                aws_region,
                endpoint_url,
            )
            if (requested_download_dir / "config.json").exists():
                base_model_source_candidate = str(requested_download_dir)
        except Exception as exc:
            logger.warning(
                "Unable to prepare requested model source as LoRA base candidate (%s): %s",
                resolved.requested_source,
                exc,
            )

    dtype_key = (torch_dtype or "auto").strip().lower() or "auto"

    model_name, model_version = _extract_model_name_and_version(resolved.effective_source)
    in_memory = _load_model_into_memory(
        resolved.load_path,
        tokenizer_fallback_id,
        dtype_key,
        fallback_model_id,
        base_model_source_candidate,
    )
    return LoadedForecastModel(
        model=in_memory.model,
        tokenizer=in_memory.tokenizer,
        model_name=model_name,
        model_version=model_version,
        requested_source=resolved.requested_source,
        effective_source=resolved.effective_source,
        load_path=resolved.load_path,
        torch_dtype=dtype_key,
    )


@lru_cache(maxsize=4)
def _load_model_into_memory(
    resolved_source: str,
    tokenizer_fallback_id: str,
    torch_dtype: str,
    fallback_model_id: str,
    base_model_source_candidate: str | None,
) -> InMemoryForecastModel:
    with _model_load_lock:
        logger.info("Loading forecasting model from source: %s (torch_dtype=%s)", resolved_source, torch_dtype)

        dtype_key = torch_dtype.strip().lower() or "auto"
        dtype_candidates: list[object | None] = []

        if dtype_key in {"auto", "default"}:
            dtype_candidates = ["auto", None]
        elif dtype_key in {"float32", "fp32"}:
            dtype_candidates = [torch.float32, "auto", None]
        elif dtype_key in {"bfloat16", "bf16"}:
            dtype_candidates = [torch.bfloat16, "auto", None]
        elif dtype_key in {"float16", "fp16"}:
            dtype_candidates = [torch.float16, "auto", None]
        else:
            logger.warning("Unknown MODEL_TORCH_DTYPE=%s, falling back to auto", torch_dtype)
            dtype_candidates = ["auto", None]

        resolved_path = Path(resolved_source)
        adapter_config_path = resolved_path / "adapter_config.json"
        adapter_model_path = resolved_path / "adapter_model.safetensors"
        config_path = resolved_path / "config.json"

        is_adapter_only = (
            resolved_path.exists()
            and adapter_config_path.exists()
            and adapter_model_path.exists()
            and not config_path.exists()
        )

        if is_adapter_only:
            if PeftModel is None:
                raise RuntimeError(
                    "LoRA adapter artifact detected but peft is not installed in runtime image"
                )

            adapter_base_from_config = ""
            try:
                adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
                adapter_base_from_config = str(adapter_config.get("base_model_name_or_path", "")).strip()
            except Exception as exc:
                logger.warning("Unable to parse adapter_config.json at %s: %s", adapter_config_path, exc)

            base_candidates: list[str] = []
            if base_model_source_candidate:
                base_candidates.append(base_model_source_candidate)
            if adapter_base_from_config:
                base_candidates.append(adapter_base_from_config)
            if fallback_model_id:
                base_candidates.append(fallback_model_id)

            deduped_candidates: list[str] = []
            for candidate in base_candidates:
                normalized = candidate.strip()
                if normalized and normalized != resolved_source and normalized not in deduped_candidates:
                    deduped_candidates.append(normalized)

            if not deduped_candidates:
                raise RuntimeError(
                    f"LoRA adapter found at {resolved_source}, but no base model candidate is available"
                )

            for base_source in deduped_candidates:
                try:
                    base_loaded = _load_model_into_memory(
                        base_source,
                        tokenizer_fallback_id,
                        torch_dtype,
                        fallback_model_id,
                        None,
                    )

                    base_model = base_loaded.model

                    if hasattr(base_model, "model"):
                        merged_model = PeftModel.from_pretrained(
                            base_model.model,
                            resolved_source,
                            is_trainable=False,
                        ).merge_and_unload()
                        merged_model.eval()
                        base_model.model = merged_model
                        logger.info(
                            "Loaded Chronos base model with merged LoRA adapter from %s (base=%s)",
                            resolved_source,
                            base_source,
                        )
                        return InMemoryForecastModel(
                            model=base_model,
                            tokenizer=base_loaded.tokenizer,
                        )

                    merged_model = PeftModel.from_pretrained(
                        base_model,
                        resolved_source,
                        is_trainable=False,
                    ).merge_and_unload()
                    merged_model.eval()
                    logger.info(
                        "Loaded transformer base model with merged LoRA adapter from %s (base=%s)",
                        resolved_source,
                        base_source,
                    )
                    return InMemoryForecastModel(
                        model=merged_model,
                        tokenizer=base_loaded.tokenizer,
                    )
                except Exception as exc:
                    logger.warning(
                        "Failed to load LoRA adapter %s with base %s: %s",
                        resolved_source,
                        base_source,
                        exc,
                    )

            raise RuntimeError(
                f"Unable to load LoRA adapter model from source: {resolved_source}"
            )

        if BaseChronosPipeline is not None:
            for dtype in dtype_candidates:
                kwargs_candidates: list[dict[str, object]] = []
                if dtype is None:
                    kwargs_candidates = [
                        {"device_map": "cpu"},
                        {},
                    ]
                else:
                    kwargs_candidates = [
                        {"device_map": "cpu", "torch_dtype": dtype},
                        {"torch_dtype": dtype},
                    ]

                for kwargs in kwargs_candidates:
                    try:
                        chronos_pipeline = BaseChronosPipeline.from_pretrained(
                            resolved_source,
                            trust_remote_code=True,
                            **kwargs,
                        )
                        if torch.get_num_threads() > 4:
                            torch.set_num_threads(4)

                        logger.info("Loaded forecasting model with BaseChronosPipeline")
                        return InMemoryForecastModel(
                            model=chronos_pipeline,
                            tokenizer=None,
                        )
                    except Exception as exc:
                        logger.warning(
                            "Chronos pipeline load attempt failed for %s with kwargs=%s: %s",
                            resolved_source,
                            kwargs,
                            exc,
                        )

        model = None
        for loader in (AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModel):
            try:
                model_kwargs: dict[str, object] = {
                    "trust_remote_code": True,
                    "low_cpu_mem_usage": True,
                }
                if dtype_key in {"auto", "default"}:
                    model_kwargs["torch_dtype"] = "auto"
                elif dtype_key in {"float32", "fp32"}:
                    model_kwargs["torch_dtype"] = torch.float32
                elif dtype_key in {"bfloat16", "bf16"}:
                    model_kwargs["torch_dtype"] = torch.bfloat16
                elif dtype_key in {"float16", "fp16"}:
                    model_kwargs["torch_dtype"] = torch.float16

                model = loader.from_pretrained(
                    resolved_source,
                    **model_kwargs,
                )
                logger.info("Loaded model with %s", loader.__name__)
                break
            except Exception:
                continue

        if model is None:
            raise RuntimeError(f"Unable to load model from source: {resolved_source}")

        model.eval()

        tokenizer = None
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                resolved_source,
                trust_remote_code=True,
            )
        except Exception as exc:
            logger.warning("Tokenizer load failed for source %s: %s", resolved_source, exc)
            if tokenizer_fallback_id:
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        tokenizer_fallback_id,
                        trust_remote_code=True,
                    )
                    logger.info("Loaded fallback tokenizer from %s", tokenizer_fallback_id)
                except Exception as fallback_exc:
                    logger.warning(
                        "Fallback tokenizer load failed from %s: %s",
                        tokenizer_fallback_id,
                        fallback_exc,
                    )
            if tokenizer is None:
                logger.info("Tokenizer is not available for model source %s", resolved_source)

        if torch.get_num_threads() > 4:
            torch.set_num_threads(4)

        return InMemoryForecastModel(
            model=model,
            tokenizer=tokenizer,
        )
