from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import DatasetDict
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingHyperParameters:
    """Hyperparameters for the generic Seq2Seq LoRA trainer path."""

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
    predict_variance_scale: float = 1.18
    predict_diffusion_steps: int = 3


def _load_model_and_tokenizer(
    model_id_or_path: str,
    cache_dir: str,
    fallback_model_id: str | None,
    trust_remote_code: bool = True,
) -> tuple[Any, Any]:
    """Load a Seq2Seq model and tokenizer with fallback support."""
    candidates = [model_id_or_path]
    if fallback_model_id:
        candidates.append(fallback_model_id)

    last_error: Exception | None = None
    for candidate in candidates:
        try:
            logger.info("Loading model and tokenizer from %s", candidate)
            tokenizer = AutoTokenizer.from_pretrained(
                candidate,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                candidate,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
            )
            return model, tokenizer
        except Exception as exc:
            last_error = exc
            logger.warning("Failed to load %s: %s", candidate, exc)
            continue

    raise RuntimeError(
        f"Unable to load model from {candidates}. Last error: {last_error}"
    ) from last_error


def _apply_lora(model: Any, hyper_params: TrainingHyperParameters) -> Any:
    """Apply PEFT LoRA adapter to the model."""
    lora_config = LoraConfig(
        r=hyper_params.lora_r,
        lora_alpha=hyper_params.lora_alpha,
        lora_dropout=hyper_params.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
        target_modules=["q", "v", "k", "o"],
    )
    model = get_peft_model(model, lora_config)
    logger.info(
        "Applied LoRA config: r=%s alpha=%s dropout=%s",
        hyper_params.lora_r,
        hyper_params.lora_alpha,
        hyper_params.lora_dropout,
    )
    return model


def build_trainer(
    model_id_or_path: str,
    cache_dir: str,
    dataset_dict: DatasetDict,
    hyper_params: TrainingHyperParameters,
    fallback_model_id: str | None = None,
    callbacks: list[Any] | None = None,
) -> tuple[Trainer, Any, Any]:
    """Build a generic HuggingFace Trainer with LoRA for Seq2Seq models.

    Note: The Chronos-2 native training path in train.py does not use this
    helper; it calls chronos.base.BaseChronosPipeline.fit() directly. This
    implementation exists for fallback / non-Chronos model paths.
    """
    logger.warning(
        "Generic build_trainer path is invoked. Chronos-2 native training should bypass this."
    )

    model, tokenizer = _load_model_and_tokenizer(
        model_id_or_path,
        cache_dir,
        fallback_model_id,
    )
    model = _apply_lora(model, hyper_params)

    output_dir = Path(hyper_params.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=hyper_params.epochs,
        learning_rate=hyper_params.learning_rate,
        per_device_train_batch_size=hyper_params.batch_size,
        per_device_eval_batch_size=hyper_params.batch_size,
        gradient_accumulation_steps=hyper_params.gradient_accumulation_steps,
        warmup_ratio=hyper_params.warmup_ratio,
        weight_decay=hyper_params.weight_decay,
        save_steps=hyper_params.save_steps,
        eval_steps=hyper_params.eval_steps,
        logging_steps=hyper_params.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        report_to="none",
        remove_unused_columns=False,
    )

    # Minimal tokenization: this generic path is not optimized for time-series
    # forecasting. For Chronos-2, use the native fit() method.
    def _tokenize(example: dict[str, Any]) -> dict[str, Any]:
        text = " ".join(f"{v:.6f}" for v in example["values"])
        return tokenizer(
            text,
            truncation=True,
            max_length=hyper_params.max_length,
            padding="max_length",
        )

    tokenized = dataset_dict.map(_tokenize, batched=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        callbacks=callbacks or [],
    )

    return trainer, model, tokenizer
