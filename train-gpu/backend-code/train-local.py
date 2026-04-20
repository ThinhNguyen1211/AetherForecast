#!/usr/bin/env python3
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys

DEFAULT_SYMBOLS = "BTCUSDT,ETHUSDT,SOLUSDT,XAUUSD,BNBUSDT,PAXGUSDT"
DEFAULT_PRIMARY_MODEL = "amazon/chronos-2"
DEFAULT_FALLBACK_MODEL = "amazon/chronos-t5-large"
DEFAULT_TIMEFRAME = "1h,4h,1d"
REEXEC_GUARD_ENV = "AETHERFORECAST_TRAIN_REEXEC"

logger = logging.getLogger("aetherforecast.local-train")


def _getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run first local fine-tune for 6 symbols using Chronos-2 + LoRA, "
            "then promote manifest/latest.json to the new S3 model version."
        )
    )
    parser.add_argument("--aws-region", default=os.getenv("AWS_REGION", "ap-southeast-1"))
    parser.add_argument("--data-bucket", default=os.getenv("DATA_S3_BUCKET") or os.getenv("DATA_BUCKET", ""))
    parser.add_argument("--model-bucket", default=os.getenv("MODEL_BUCKET", ""))
    parser.add_argument("--model-s3-uri", default=os.getenv("MODEL_S3_URI", ""))
    parser.add_argument("--checkpoint-s3-uri", default=os.getenv("CHECKPOINT_S3_URI", ""))
    parser.add_argument("--symbols", default=DEFAULT_SYMBOLS)
    parser.add_argument(
        "--timeframe",
        default=os.getenv("TIMEFRAME", DEFAULT_TIMEFRAME),
        help="Single timeframe (e.g. 1m) or comma list (e.g. 1m,5m,15m) or all",
    )
    parser.add_argument("--epochs", type=int, default=int(os.getenv("EPOCHS", "2")))
    parser.add_argument("--batch-size", type=int, choices=[2, 4], default=int(os.getenv("BATCH_SIZE", "2")))
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=int(os.getenv("GRAD_ACCUM_STEPS", "8")),
    )
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.0002")))
    parser.add_argument("--max-rows-per-symbol", type=int, default=int(os.getenv("MAX_ROWS_PER_SYMBOL", "8000")))
    parser.add_argument("--context-length", type=int, default=int(os.getenv("CONTEXT_LENGTH", "1024")))
    parser.add_argument("--horizon", type=int, default=int(os.getenv("TRAINING_HORIZON", "7")))
    parser.add_argument("--max-seq-length", type=int, default=int(os.getenv("MAX_SEQ_LENGTH", "1024")))
    parser.add_argument("--save-steps", type=int, default=int(os.getenv("SAVE_STEPS", "50")))
    parser.add_argument("--eval-steps", type=int, default=int(os.getenv("EVAL_STEPS", "50")))
    parser.add_argument("--logging-steps", type=int, default=int(os.getenv("LOGGING_STEPS", "5")))
    parser.add_argument("--lora-r", type=int, default=int(os.getenv("LORA_R", "16")))
    parser.add_argument("--lora-alpha", type=int, default=int(os.getenv("LORA_ALPHA", "32")))
    parser.add_argument("--lora-dropout", type=float, default=float(os.getenv("LORA_DROPOUT", "0.05")))
    parser.add_argument("--train-split-ratio", type=float, default=float(os.getenv("TRAIN_SPLIT_RATIO", "0.95")))
    parser.add_argument(
        "--walk-forward-windows",
        type=int,
        default=int(os.getenv("WALK_FORWARD_WINDOWS", "4")),
        help="Number of rolling walk-forward windows for validation.",
    )
    parser.add_argument(
        "--walk-forward-eval-size",
        type=int,
        default=int(os.getenv("WALK_FORWARD_EVAL_SIZE", "128")),
        help="Validation span size per walk-forward window.",
    )
    parser.add_argument(
        "--external-covariate-scale",
        type=float,
        default=float(os.getenv("EXTERNAL_COVARIATE_SCALE", "0.0018")),
        help="How strongly external covariates perturb context series during training.",
    )
    parser.add_argument(
        "--predict-variance-scale",
        type=float,
        default=float(os.getenv("PREDICT_VARIANCE_SCALE", "1.18")),
        help="Base variance scaling factor written to postprocess calibration metadata.",
    )
    parser.add_argument(
        "--predict-diffusion-steps",
        type=int,
        default=int(os.getenv("PREDICT_DIFFUSION_STEPS", "3")),
        help="Base diffusion refinement steps written to postprocess calibration metadata.",
    )
    parser.add_argument("--base-model-id", default=os.getenv("BASE_MODEL_ID", DEFAULT_PRIMARY_MODEL))
    parser.add_argument(
        "--base-model-fallback-id",
        default=os.getenv("BASE_MODEL_FALLBACK_ID", DEFAULT_FALLBACK_MODEL),
    )
    parser.add_argument("--parquet-prefix", default=os.getenv("PARQUET_PREFIX", "market/klines"))
    parser.add_argument("--output-dir", default=os.getenv("TRAIN_OUTPUT_DIR", "./artifacts/local-training"))
    parser.add_argument("--hf-cache-dir", default=os.getenv("HF_CACHE_DIR", "./artifacts/hf-cache"))

    parser.add_argument(
        "--force-model-redownload",
        dest="force_model_redownload",
        action="store_true",
        default=_getenv_bool("HF_FORCE_DOWNLOAD", False),
        help="Force fresh model/tokenizer download from Hugging Face.",
    )
    parser.add_argument(
        "--no-force-model-redownload",
        dest="force_model_redownload",
        action="store_false",
        help="Disable forced model/tokenizer redownload.",
    )
    parser.add_argument(
        "--local-files-only",
        dest="local_files_only",
        action="store_true",
        default=_getenv_bool("HF_LOCAL_FILES_ONLY", False),
        help="Load model/tokenizer from local HF cache only (no remote downloads).",
    )
    parser.add_argument(
        "--allow-remote-download",
        dest="local_files_only",
        action="store_false",
        help="Allow remote model/tokenizer downloads when cache misses occur.",
    )
    parser.add_argument(
        "--enable-live-external-fetch",
        dest="enable_live_external_fetch",
        action="store_true",
        default=_getenv_bool("ENABLE_EXTERNAL_FETCH", True),
        help="Enable live external covariate fetches (Fear & Greed, macro, Binance derivatives).",
    )
    parser.add_argument(
        "--disable-live-external-fetch",
        dest="enable_live_external_fetch",
        action="store_false",
        help="Disable live external covariate fetches and rely on local data only.",
    )
    parser.add_argument(
        "--strict-external-data",
        dest="strict_external_data",
        action="store_true",
        default=_getenv_bool("STRICT_EXTERNAL_DATA", False),
        help="Fail early when required external covariates are missing.",
    )
    parser.add_argument(
        "--allow-external-fallback",
        dest="strict_external_data",
        action="store_false",
        help="Allow training to continue even if some external covariates are unavailable.",
    )
    return parser


def _reset_hf_cache_dir(cache_dir: Path) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir, ignore_errors=True)
    cache_dir.mkdir(parents=True, exist_ok=True)


def _load_train_entrypoint():
    try:
        from ml.training.train import main as train_main
    except ModuleNotFoundError as exc:
        missing_module = exc.name or "unknown"
        root_dir = Path(__file__).resolve().parents[2]
        train_gpu_python = root_dir / "train-gpu" / "Scripts" / "python.exe"

        if os.getenv(REEXEC_GUARD_ENV) == "1":
            raise RuntimeError(
                f"Missing Python dependency '{missing_module}' in interpreter '{sys.executable}'. "
                "Install project requirements into this environment or run scripts/run-train-gpu.bat."
            ) from exc

        if train_gpu_python.exists():
            logger.warning(
                "Missing dependency '%s' in interpreter %s. Re-launching with %s",
                missing_module,
                sys.executable,
                train_gpu_python,
            )
            env = os.environ.copy()
            env[REEXEC_GUARD_ENV] = "1"
            result = subprocess.run(
                [str(train_gpu_python), str(Path(__file__).resolve()), *sys.argv[1:]],
                env=env,
                check=False,
            )
            raise SystemExit(result.returncode)

        raise RuntimeError(
            f"Missing Python dependency '{missing_module}' in interpreter '{sys.executable}'. "
            f"Expected training environment not found at '{train_gpu_python}'."
        ) from exc

    return train_main


def resolve_s3_targets(args: argparse.Namespace) -> tuple[str, str]:
    model_s3_uri = args.model_s3_uri.strip()
    checkpoint_s3_uri = args.checkpoint_s3_uri.strip()

    if not model_s3_uri:
        if not args.model_bucket.strip():
            raise ValueError("Provide --model-bucket or --model-s3-uri")
        model_s3_uri = f"s3://{args.model_bucket.strip()}/chronos-v1/model/"

    if not checkpoint_s3_uri:
        if not args.model_bucket.strip():
            raise ValueError("Provide --model-bucket or --checkpoint-s3-uri")
        checkpoint_s3_uri = f"s3://{args.model_bucket.strip()}/checkpoints/"

    return model_s3_uri, checkpoint_s3_uri


def choose_batch_size(default_batch_size: int) -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            logger.warning("CUDA is not available. Training on CPU will be very slow.")
            return default_batch_size

        device_name = torch.cuda.get_device_name(0)
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("Detected GPU: %s (%.2f GiB VRAM)", device_name, total_gb)

        if total_gb < 5.0 and default_batch_size > 2:
            logger.warning("VRAM is limited; overriding batch size to 2.")
            return 2
    except Exception as exc:  # pragma: no cover
        logger.warning("Unable to inspect CUDA device: %s", exc)

    return default_batch_size


def export_training_env(args: argparse.Namespace) -> None:
    model_s3_uri, checkpoint_s3_uri = resolve_s3_targets(args)

    output_dir = Path(args.output_dir).resolve()
    cache_dir = Path(args.hf_cache_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    batch_size = choose_batch_size(args.batch_size)

    env_values = {
        "AWS_REGION": args.aws_region,
        "DATA_S3_BUCKET": args.data_bucket,
        "DATA_BUCKET": args.data_bucket,
        "MODEL_S3_URI": model_s3_uri,
        "CHECKPOINT_S3_URI": checkpoint_s3_uri,
        "SYMBOLS": args.symbols,
        "TIMEFRAME": args.timeframe,
        "EPOCHS": str(args.epochs),
        "BATCH_SIZE": str(batch_size),
        "GRAD_ACCUM_STEPS": str(args.grad_accum_steps),
        "LEARNING_RATE": str(args.learning_rate),
        "MAX_ROWS_PER_SYMBOL": str(args.max_rows_per_symbol),
        "CONTEXT_LENGTH": str(args.context_length),
        "TRAINING_HORIZON": str(args.horizon),
        "MAX_SEQ_LENGTH": str(args.max_seq_length),
        "SAVE_STEPS": str(args.save_steps),
        "EVAL_STEPS": str(args.eval_steps),
        "LOGGING_STEPS": str(args.logging_steps),
        "LORA_R": str(args.lora_r),
        "LORA_ALPHA": str(args.lora_alpha),
        "LORA_DROPOUT": str(args.lora_dropout),
        "TRAIN_SPLIT_RATIO": str(args.train_split_ratio),
        "WALK_FORWARD_WINDOWS": str(args.walk_forward_windows),
        "WALK_FORWARD_EVAL_SIZE": str(args.walk_forward_eval_size),
        "EXTERNAL_COVARIATE_SCALE": str(args.external_covariate_scale),
        "ENABLE_EXTERNAL_FETCH": "1" if args.enable_live_external_fetch else "0",
        "STRICT_EXTERNAL_DATA": "1" if args.strict_external_data else "0",
        "PREDICT_VARIANCE_SCALE": str(args.predict_variance_scale),
        "PREDICT_DIFFUSION_STEPS": str(args.predict_diffusion_steps),
        "BASE_MODEL_ID": args.base_model_id,
        "BASE_MODEL_FALLBACK_ID": args.base_model_fallback_id,
        "PARQUET_PREFIX": args.parquet_prefix,
        "TRAIN_OUTPUT_DIR": str(output_dir),
        "HF_CACHE_DIR": str(cache_dir),
        "HF_HOME": str(cache_dir),
        "HF_TRUST_REMOTE_CODE": "1",
        "HF_LOCAL_FILES_ONLY": "1" if args.local_files_only else "0",
        "HF_FORCE_DOWNLOAD": "1" if args.force_model_redownload else "0",
        "TOKENIZERS_PARALLELISM": "false",
        "TRANSFORMERS_VERBOSITY": "info",
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO"),
    }

    for key, value in env_values.items():
        os.environ[key] = value

    logger.info("Local train configuration is ready")
    logger.info("Symbols: %s", args.symbols)
    logger.info("Timeframes: %s", args.timeframe)
    logger.info(
        "Walk-forward: windows=%s eval_size=%s",
        args.walk_forward_windows,
        args.walk_forward_eval_size,
    )
    logger.info(
        "External covariates: live_fetch=%s strict=%s scale=%.6f",
        args.enable_live_external_fetch,
        args.strict_external_data,
        args.external_covariate_scale,
    )
    logger.info(
        "Postprocess controls: variance_scale=%.4f diffusion_steps=%s",
        args.predict_variance_scale,
        args.predict_diffusion_steps,
    )
    logger.info("Model primary=%s fallback=%s", args.base_model_id, args.base_model_fallback_id)
    logger.info("Output dir: %s", output_dir)
    logger.info("HF cache dir: %s", cache_dir)
    logger.info(
        "HF load options: trust_remote_code=true local_files_only=%s force_download=%s",
        args.local_files_only,
        args.force_model_redownload,
    )


def main() -> int:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
    parser = build_parser()
    args = parser.parse_args()

    if not args.data_bucket.strip():
        parser.error("--data-bucket is required (or DATA_S3_BUCKET/DATA_BUCKET env)")

    train_main = _load_train_entrypoint()

    export_training_env(args)

    # Reuse the production training pipeline: train + upload version + promote manifest.
    try:
        train_main()
    except RuntimeError as exc:
        error_text = str(exc)
        should_retry = "Unable to load any Chronos base model candidate" in error_text
        if not should_retry:
            raise

        cache_dir = Path(os.environ.get("HF_CACHE_DIR", args.hf_cache_dir)).resolve()
        logger.warning(
            "Model load failed (%s). Resetting HF cache and retrying with forced redownload.",
            error_text,
        )
        _reset_hf_cache_dir(cache_dir)
        os.environ["HF_FORCE_DOWNLOAD"] = "1"
        os.environ["HF_LOCAL_FILES_ONLY"] = "0"
        train_main()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
