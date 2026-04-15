from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import logging
import os
from pathlib import Path
import signal
from typing import Any

import numpy as np
import torch

from ml.training.checkpoint import S3CheckpointCallback, S3CheckpointManager
from ml.training.dataset import (
    TrainingDatasetConfig,
    build_training_datasets,
    load_market_dataframe,
    parse_symbols,
)
from ml.training.promote_model import promote_model_version
from ml.training.trainer import TrainingHyperParameters, build_trainer
from transformers import TrainerCallback, TrainerControl, TrainerState

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("aetherforecast.training")

_INTERRUPTED = False
DEFAULT_CHRONOS_PRIMARY_MODEL_ID = "amazon/chronos-2"
DEFAULT_CHRONOS_FALLBACK_MODEL_ID = "amazon/chronos-t5-large"


def _handle_interrupt(signum, _frame) -> None:
    global _INTERRUPTED
    _INTERRUPTED = True
    logger.warning("Received signal %s. Training will stop after current step and checkpoint.", signum)


@dataclass
class TrainRuntimeConfig:
    aws_region: str
    aws_endpoint_url: str | None
    data_bucket: str
    model_s3_uri: str
    checkpoint_s3_uri: str
    symbols: list[str]
    timeframe: str
    horizon: int
    context_length: int
    max_rows_per_symbol: int
    train_split_ratio: float
    epochs: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    warmup_ratio: float
    weight_decay: float
    save_steps: int
    eval_steps: int
    logging_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    max_length: int
    inference_model_id: str
    inference_model_fallback_id: str
    output_dir: str
    cache_dir: str
    hf_trust_remote_code: bool
    hf_local_files_only: bool
    hf_force_download: bool
    chronos2_train_steps: int


class InterruptAwareCallback(TrainerCallback):
    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs) -> TrainerControl:
        if _INTERRUPTED:
            control.should_save = True
            control.should_training_stop = True
        return control


def _getenv_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _getenv_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def load_runtime_config() -> TrainRuntimeConfig:
    symbols = parse_symbols(os.getenv("SYMBOLS"))

    data_bucket = os.getenv("DATA_S3_BUCKET") or os.getenv("DATA_BUCKET", "")
    model_s3_uri = os.getenv("MODEL_S3_URI", "s3://aetherforecast-models/chronos-v1/model/")
    checkpoint_s3_uri = os.getenv("CHECKPOINT_S3_URI", "s3://aetherforecast-models/checkpoints/")

    return TrainRuntimeConfig(
        aws_region=os.getenv("AWS_REGION", "ap-southeast-1"),
        aws_endpoint_url=os.getenv("AWS_ENDPOINT_URL"),
        data_bucket=data_bucket,
        model_s3_uri=model_s3_uri,
        checkpoint_s3_uri=checkpoint_s3_uri,
        symbols=symbols,
        timeframe=os.getenv("TIMEFRAME", "1h"),
        horizon=_getenv_int("TRAINING_HORIZON", 7),
        context_length=_getenv_int("CONTEXT_LENGTH", 96),
        max_rows_per_symbol=_getenv_int("MAX_ROWS_PER_SYMBOL", 20000),
        train_split_ratio=_getenv_float("TRAIN_SPLIT_RATIO", 0.95),
        epochs=_getenv_int("EPOCHS", 3),
        learning_rate=_getenv_float("LEARNING_RATE", 2e-4),
        batch_size=_getenv_int("BATCH_SIZE", 4),
        gradient_accumulation_steps=_getenv_int("GRAD_ACCUM_STEPS", 8),
        warmup_ratio=_getenv_float("WARMUP_RATIO", 0.03),
        weight_decay=_getenv_float("WEIGHT_DECAY", 0.01),
        save_steps=_getenv_int("SAVE_STEPS", 100),
        eval_steps=_getenv_int("EVAL_STEPS", 100),
        logging_steps=_getenv_int("LOGGING_STEPS", 20),
        lora_r=_getenv_int("LORA_R", 16),
        lora_alpha=_getenv_int("LORA_ALPHA", 32),
        lora_dropout=_getenv_float("LORA_DROPOUT", 0.05),
        max_length=_getenv_int("MAX_SEQ_LENGTH", 512),
        inference_model_id=os.getenv("BASE_MODEL_ID", DEFAULT_CHRONOS_PRIMARY_MODEL_ID),
        inference_model_fallback_id=os.getenv(
            "BASE_MODEL_FALLBACK_ID",
            DEFAULT_CHRONOS_FALLBACK_MODEL_ID,
        ),
        output_dir=os.getenv("TRAIN_OUTPUT_DIR", "/tmp/aetherforecast-training"),
        cache_dir=os.getenv("HF_CACHE_DIR", "/tmp/hf-cache"),
        hf_trust_remote_code=_getenv_bool("HF_TRUST_REMOTE_CODE", True),
        hf_local_files_only=_getenv_bool("HF_LOCAL_FILES_ONLY", False),
        hf_force_download=_getenv_bool("HF_FORCE_DOWNLOAD", False),
        chronos2_train_steps=_getenv_int("CHRONOS2_TRAIN_STEPS", 0),
    )


def _resolve_model_version_uri(model_s3_uri: str) -> tuple[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    version_uri = f"{model_s3_uri.rstrip('/')}/versions/{timestamp}/"
    return timestamp, version_uri


def _is_probably_chronos2(model_id_or_path: str) -> bool:
    return "chronos-2" in (model_id_or_path or "").strip().lower()


def _prepare_chronos2_inputs(
    market_dataframe,
    *,
    horizon: int,
    context_length: int,
    train_split_ratio: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    train_inputs: list[np.ndarray] = []
    validation_inputs: list[np.ndarray] = []

    group_columns = ["symbol", "timeframe"] if "timeframe" in market_dataframe.columns else ["symbol"]
    minimum_required = max(horizon + context_length + 8, horizon * 4)

    for _, group in market_dataframe.groupby(group_columns):
        ordered = group.sort_values("timestamp")
        closes = ordered["close"].to_numpy(dtype=np.float32)
        if len(closes) < minimum_required:
            continue

        split_index = int(len(closes) * train_split_ratio)
        split_index = max(split_index, horizon + context_length)
        split_index = min(split_index, len(closes) - horizon)

        train_series = closes[:split_index]
        if len(train_series) >= minimum_required:
            train_inputs.append(train_series)

        validation_start = max(split_index - context_length, 0)
        validation_series = closes[validation_start:]
        if len(validation_series) >= minimum_required:
            validation_inputs.append(validation_series)

    if not train_inputs:
        raise ValueError(
            "Not enough market history to build Chronos-2 training inputs. "
            "Increase MAX_ROWS_PER_SYMBOL or reduce CONTEXT_LENGTH/TRAINING_HORIZON."
        )

    return train_inputs, validation_inputs


def _resolve_chronos2_num_steps(config: TrainRuntimeConfig, train_inputs: list[np.ndarray]) -> int:
    if config.chronos2_train_steps > 0:
        return config.chronos2_train_steps

    approx_windows = sum(max(1, len(series) - config.context_length - config.horizon + 1) for series in train_inputs)
    denominator = max(config.batch_size * 64, 1)
    steps_per_epoch = max(10, approx_windows // denominator)
    return max(100, min(2000, steps_per_epoch * max(config.epochs, 1)))


def _load_chronos2_pipeline(config: TrainRuntimeConfig) -> tuple[Any, str]:
    try:
        base_module = importlib.import_module("chronos.base")
        BaseChronosPipeline = getattr(base_module, "BaseChronosPipeline")
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("chronos package is required for Chronos-2 training") from exc

    candidates = [config.inference_model_id]
    fallback_candidate = (config.inference_model_fallback_id or "").strip()
    if fallback_candidate and fallback_candidate not in candidates:
        candidates.append(fallback_candidate)

    local_files_only = config.hf_local_files_only
    force_download = config.hf_force_download
    if local_files_only and force_download:
        logger.warning(
            "HF_LOCAL_FILES_ONLY=1 conflicts with HF_FORCE_DOWNLOAD=1 for Chronos-2 native load. "
            "Disabling local_files_only."
        )
        local_files_only = False

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            pipeline = BaseChronosPipeline.from_pretrained(
                candidate,
                trust_remote_code=config.hf_trust_remote_code,
                cache_dir=config.cache_dir,
                local_files_only=local_files_only,
                force_download=force_download,
            )
        except Exception as exc:
            last_error = exc
            logger.warning("Chronos pipeline load failed for %s: %s", candidate, exc)
            continue

        if not hasattr(pipeline, "fit"):
            logger.warning(
                "Loaded pipeline from %s does not expose fit(); class=%s",
                candidate,
                pipeline.__class__.__name__,
            )
            continue

        return pipeline, candidate

    raise RuntimeError(
        "Unable to load a trainable Chronos-2 pipeline. "
        f"primary={config.inference_model_id}, fallback={fallback_candidate or 'none'}"
    ) from last_error


def _train_with_chronos2_native(
    config: TrainRuntimeConfig,
    dataset_config: TrainingDatasetConfig,
    checkpoint_manager: S3CheckpointManager,
    output_dir: Path,
    callbacks: list[TrainerCallback],
) -> None:
    market_dataframe = load_market_dataframe(dataset_config)
    train_inputs, validation_inputs = _prepare_chronos2_inputs(
        market_dataframe,
        horizon=config.horizon,
        context_length=config.context_length,
        train_split_ratio=config.train_split_ratio,
    )

    pipeline, loaded_from = _load_chronos2_pipeline(config)
    logger.info(
        "Loaded Chronos pipeline class=%s from %s",
        pipeline.__class__.__name__,
        loaded_from,
    )

    if torch.cuda.is_available():
        pipeline.model = pipeline.model.to("cuda")
        logger.info("Using CUDA for Chronos-2 native fine-tuning")
    else:
        logger.warning("CUDA is not available. Chronos-2 native fine-tuning will run on CPU.")

    num_steps = _resolve_chronos2_num_steps(config, train_inputs)
    logger.info(
        "Chronos-2 native fine-tune: train_series=%s validation_series=%s num_steps=%s",
        len(train_inputs),
        len(validation_inputs),
        num_steps,
    )

    lora_config = {
        "r": config.lora_r,
        "lora_alpha": config.lora_alpha,
        "lora_dropout": config.lora_dropout,
        "target_modules": [
            "self_attention.q",
            "self_attention.v",
            "self_attention.k",
            "self_attention.o",
            "output_patch_embedding.output_layer",
        ],
    }

    pipeline.fit(
        inputs=train_inputs,
        validation_inputs=validation_inputs if validation_inputs else None,
        prediction_length=config.horizon,
        finetune_mode="lora",
        lora_config=lora_config,
        context_length=config.context_length,
        learning_rate=config.learning_rate,
        num_steps=num_steps,
        batch_size=max(1, config.batch_size),
        output_dir=str(output_dir),
        min_past=max(config.horizon, min(config.context_length, 128)),
        finetuned_ckpt_name="final-model",
        callbacks=callbacks,
        remove_printer_callback=True,
        disable_data_parallel=True,
        logging_steps=config.logging_steps,
        save_strategy="steps",
        save_steps=config.save_steps,
        eval_strategy="steps" if validation_inputs else "no",
        eval_steps=config.eval_steps if validation_inputs else None,
        report_to="none",
    )

    if _INTERRUPTED:
        logger.warning("Chronos-2 training interrupted. Skipping model promotion.")
        raise SystemExit(143)

    final_local_dir = output_dir / "final-model"
    final_local_dir.mkdir(parents=True, exist_ok=True)
    if not any(final_local_dir.iterdir()):
        pipeline.save_pretrained(str(final_local_dir))

    timestamp, version_uri = _resolve_model_version_uri(config.model_s3_uri)
    logger.info("Uploading fine-tuned model version %s to %s", timestamp, version_uri)
    checkpoint_manager.upload_directory(str(final_local_dir), version_uri)

    manifest_uri = promote_model_version(
        model_root_s3_uri=config.model_s3_uri,
        trained_version_s3_uri=version_uri,
        aws_region=config.aws_region,
        endpoint_url=config.aws_endpoint_url,
    )
    logger.info("Training complete. Model promoted through manifest: %s", manifest_uri)


def main() -> None:
    signal.signal(signal.SIGTERM, _handle_interrupt)
    signal.signal(signal.SIGINT, _handle_interrupt)

    config = load_runtime_config()
    logger.info("Training config loaded: symbols=%s horizon=%s", config.symbols, config.horizon)

    dataset_config = TrainingDatasetConfig(
        data_bucket=config.data_bucket,
        symbols=config.symbols,
        timeframe=config.timeframe,
        horizon=config.horizon,
        context_length=config.context_length,
        max_rows_per_symbol=config.max_rows_per_symbol,
        train_split_ratio=config.train_split_ratio,
        aws_region=config.aws_region,
        aws_endpoint_url=config.aws_endpoint_url,
    )

    checkpoint_manager = S3CheckpointManager(
        s3_uri=config.checkpoint_s3_uri,
        aws_region=config.aws_region,
        endpoint_url=config.aws_endpoint_url,
    )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [S3CheckpointCallback(checkpoint_manager), InterruptAwareCallback()]

    if _is_probably_chronos2(config.inference_model_id) or _is_probably_chronos2(
        config.inference_model_fallback_id
    ):
        _train_with_chronos2_native(
            config=config,
            dataset_config=dataset_config,
            checkpoint_manager=checkpoint_manager,
            output_dir=output_dir,
            callbacks=callbacks,
        )
        return

    dataset_dict = build_training_datasets(dataset_config)

    local_resume_checkpoint = checkpoint_manager.download_latest_checkpoint(output_dir)

    hyper_params = TrainingHyperParameters(
        output_dir=str(output_dir),
        epochs=config.epochs,
        learning_rate=config.learning_rate,
        batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        logging_steps=config.logging_steps,
        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        max_length=config.max_length,
    )

    trainer, model, tokenizer = build_trainer(
        model_id_or_path=config.inference_model_id,
        cache_dir=config.cache_dir,
        dataset_dict=dataset_dict,
        hyper_params=hyper_params,
        fallback_model_id=config.inference_model_fallback_id,
        callbacks=callbacks,
    )

    train_output = trainer.train(
        resume_from_checkpoint=str(local_resume_checkpoint) if local_resume_checkpoint else None,
    )
    trainer.log_metrics("train", train_output.metrics)
    trainer.save_metrics("train", train_output.metrics)

    if _INTERRUPTED:
        logger.warning("Training interrupted. Saving checkpoint and exiting.")
        trainer.save_state()
        raise SystemExit(143)

    final_local_dir = output_dir / "final-model"
    final_local_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(final_local_dir))
    tokenizer.save_pretrained(str(final_local_dir))

    timestamp, version_uri = _resolve_model_version_uri(config.model_s3_uri)
    logger.info("Uploading fine-tuned model version %s to %s", timestamp, version_uri)

    checkpoint_manager.upload_directory(str(final_local_dir), version_uri)

    manifest_uri = promote_model_version(
        model_root_s3_uri=config.model_s3_uri,
        trained_version_s3_uri=version_uri,
        aws_region=config.aws_region,
        endpoint_url=config.aws_endpoint_url,
    )

    logger.info("Training complete. Model promoted through manifest: %s", manifest_uri)


if __name__ == "__main__":
    main()
