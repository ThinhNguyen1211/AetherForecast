from __future__ import annotations

from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any

from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingHyperParameters:
    output_dir: str
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


@dataclass
class ModelLoadOptions:
    trust_remote_code: bool = True
    local_files_only: bool = False
    force_download: bool = False


def _getenv_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_model_load_options() -> ModelLoadOptions:
    options = ModelLoadOptions(
        trust_remote_code=_getenv_bool("HF_TRUST_REMOTE_CODE", True),
        local_files_only=_getenv_bool("HF_LOCAL_FILES_ONLY", False),
        force_download=_getenv_bool("HF_FORCE_DOWNLOAD", False),
    )

    if options.force_download and options.local_files_only:
        logger.warning(
            "HF_FORCE_DOWNLOAD=1 conflicts with HF_LOCAL_FILES_ONLY=1. "
            "Disabling local_files_only for this run."
        )
        options.local_files_only = False

    return options


def _coerce_to_string_path(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    try:
        return os.fspath(value)
    except TypeError:
        try:
            return str(value)
        except Exception:
            return None


def _normalize_tokenizer_paths(tokenizer: Any) -> tuple[str | None, str]:
    tracked_keys = ["vocab_file", "sp_model_file", "sentencepiece_model_file", "tokenizer_file"]
    debug_parts: list[str] = []
    first_vocab_path: str | None = None

    for key in tracked_keys:
        if not hasattr(tokenizer, key):
            continue
        raw = getattr(tokenizer, key)
        coerced = _coerce_to_string_path(raw)
        if coerced is not None and not isinstance(raw, str):
            try:
                setattr(tokenizer, key, coerced)
            except Exception:
                pass

        current = getattr(tokenizer, key, None)
        current_as_string = _coerce_to_string_path(current)
        if first_vocab_path is None and current_as_string:
            first_vocab_path = current_as_string
        debug_parts.append(f"{key}={current_as_string!r} ({type(current).__name__})")

    init_kwargs = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init_kwargs, dict):
        for key in tracked_keys:
            if key not in init_kwargs:
                continue
            coerced = _coerce_to_string_path(init_kwargs.get(key))
            if coerced is not None:
                init_kwargs[key] = coerced
                if first_vocab_path is None:
                    first_vocab_path = coerced

    if not debug_parts:
        debug_parts.append("no_vocab_attributes")

    return first_vocab_path, ", ".join(debug_parts)


def _build_hf_kwargs(cache_dir: str, options: ModelLoadOptions) -> dict[str, Any]:
    return {
        "trust_remote_code": options.trust_remote_code,
        "cache_dir": cache_dir,
        "local_files_only": options.local_files_only,
        "force_download": options.force_download,
    }


def _load_base_model(
    model_id_or_path: str,
    cache_dir: str,
    fallback_model_id: str | None = None,
    load_options: ModelLoadOptions | None = None,
) -> tuple[Any, Any, bool]:
    options = load_options or ModelLoadOptions()
    candidates = [model_id_or_path]
    normalized_fallback = (fallback_model_id or "").strip()
    if normalized_fallback and normalized_fallback not in candidates:
        candidates.append(normalized_fallback)

    last_error: Exception | None = None

    for candidate in candidates:
        attempt_options = [options]
        if not options.force_download:
            attempt_options.append(
                ModelLoadOptions(
                    trust_remote_code=options.trust_remote_code,
                    local_files_only=False,
                    force_download=True,
                )
            )

        for attempt_index, current_options in enumerate(attempt_options, start=1):
            kwargs = _build_hf_kwargs(cache_dir, current_options)
            logger.info(
                "Loading candidate=%s attempt=%s cache_dir=%s trust_remote_code=%s local_files_only=%s force_download=%s",
                candidate,
                attempt_index,
                cache_dir,
                current_options.trust_remote_code,
                current_options.local_files_only,
                current_options.force_download,
            )

            try:
                tokenizer = AutoTokenizer.from_pretrained(candidate, use_fast=False, **kwargs)
                vocab_path, vocab_debug = _normalize_tokenizer_paths(tokenizer)
                logger.info(
                    "Tokenizer loaded candidate=%s class=%s vocab_file=%r debug=[%s]",
                    candidate,
                    tokenizer.__class__.__name__,
                    vocab_path,
                    vocab_debug,
                )

                if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                    tokenizer.pad_token = tokenizer.eos_token

                seq2seq_error: Exception | None = None
                try:
                    model = AutoModelForSeq2SeqLM.from_pretrained(candidate, **kwargs)
                    logger.info("Loaded seq2seq model for training from %s", candidate)
                    return model, tokenizer, True
                except Exception as exc:
                    seq2seq_error = exc
                    logger.warning(
                        "Seq2seq load failed for %s on attempt %s: %s",
                        candidate,
                        attempt_index,
                        exc,
                    )

                model = AutoModelForCausalLM.from_pretrained(candidate, **kwargs)
                logger.info("Loaded causal LM model for training from %s", candidate)
                return model, tokenizer, False
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "Failed loading candidate=%s attempt=%s: %s",
                    candidate,
                    attempt_index,
                    exc,
                )

    raise RuntimeError(
        "Unable to load any Chronos base model candidate. "
        f"primary={model_id_or_path}, fallback={normalized_fallback or 'none'}"
    ) from last_error


def _tokenize_seq2seq(dataset_dict: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def tokenize(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        labels = tokenizer(
            text_target=batch["target_text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return dataset_dict.map(tokenize, batched=True, remove_columns=dataset_dict["train"].column_names)


def _tokenize_causal(dataset_dict: DatasetDict, tokenizer, max_length: int) -> DatasetDict:
    def format_and_tokenize(batch):
        combined = [
            f"{inp}\nforecast={target}"
            for inp, target in zip(batch["input_text"], batch["target_text"], strict=False)
        ]
        encoded = tokenizer(
            combined,
            truncation=True,
            max_length=max_length,
            padding=False,
        )
        encoded["labels"] = encoded["input_ids"].copy()
        return encoded

    return dataset_dict.map(
        format_and_tokenize,
        batched=True,
        remove_columns=dataset_dict["train"].column_names,
    )


def build_trainer(
    model_id_or_path: str,
    cache_dir: str,
    dataset_dict: DatasetDict,
    hyper_params: TrainingHyperParameters,
    fallback_model_id: str | None = None,
    callbacks: list[Any] | None = None,
) -> tuple[Trainer, Any, Any]:
    load_options = _build_model_load_options()
    model, tokenizer, is_seq2seq = _load_base_model(
        model_id_or_path,
        cache_dir,
        fallback_model_id=fallback_model_id,
        load_options=load_options,
    )

    lora_task_type = TaskType.SEQ_2_SEQ_LM if is_seq2seq else TaskType.CAUSAL_LM
    lora_config = LoraConfig(
        task_type=lora_task_type,
        r=hyper_params.lora_r,
        lora_alpha=hyper_params.lora_alpha,
        lora_dropout=hyper_params.lora_dropout,
        target_modules="all-linear",
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    tokenized = (
        _tokenize_seq2seq(dataset_dict, tokenizer, hyper_params.max_length)
        if is_seq2seq
        else _tokenize_causal(dataset_dict, tokenizer, hyper_params.max_length)
    )

    data_collator = (
        DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
        if is_seq2seq
        else DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    output_dir = Path(hyper_params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_fp16 = torch.cuda.is_available()

    training_arguments = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        num_train_epochs=hyper_params.epochs,
        learning_rate=hyper_params.learning_rate,
        per_device_train_batch_size=hyper_params.batch_size,
        per_device_eval_batch_size=max(1, hyper_params.batch_size),
        gradient_accumulation_steps=hyper_params.gradient_accumulation_steps,
        warmup_ratio=hyper_params.warmup_ratio,
        weight_decay=hyper_params.weight_decay,
        logging_steps=hyper_params.logging_steps,
        save_steps=hyper_params.save_steps,
        eval_steps=hyper_params.eval_steps,
        save_total_limit=3,
        evaluation_strategy="steps",
        save_strategy="steps",
        report_to=[],
        bf16=False,
        fp16=use_fp16,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    return trainer, model, tokenizer
