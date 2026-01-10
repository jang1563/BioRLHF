"""
Supervised Fine-Tuning (SFT) module for BioRLHF.

This module provides functionality for fine-tuning language models on
biological instruction-following tasks using the TRL library.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb

from biorlhf.utils.model_utils import get_quantization_config, get_lora_config


@dataclass
class SFTTrainingConfig:
    """Configuration for SFT training."""

    # Model settings
    model_name: str = "mistralai/Mistral-7B-v0.3"
    dataset_path: str = "kmp_sft_dataset.json"
    output_dir: str = "./biorlhf_sft_model"

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    max_length: int = 1024
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01

    # LoRA settings
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05

    # Quantization
    use_4bit: bool = True

    # Logging
    logging_steps: int = 10
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 3

    # Weights & Biases
    wandb_project: str = "biorlhf"
    wandb_run_name: str = "sft_training"
    use_wandb: bool = True

    # Evaluation
    test_size: float = 0.1
    seed: int = 42


def run_sft_training(config: Optional[SFTTrainingConfig] = None) -> str:
    """
    Run SFT training with the given configuration.

    Args:
        config: Training configuration. If None, uses defaults.

    Returns:
        Path to the saved model.
    """
    if config is None:
        config = SFTTrainingConfig()

    print("=" * 60)
    print("BioRLHF SFT Training")
    print("=" * 60)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    print(f"Epochs: {config.num_epochs}")
    print("=" * 60)

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset("json", data_files=config.dataset_path)["train"]
    print(f"Dataset size: {len(dataset)} examples")

    # Split into train/eval
    dataset = dataset.train_test_split(test_size=config.test_size, seed=config.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Quantization config
    bnb_config = get_quantization_config() if config.use_4bit else None

    # Load model
    print(f"\nLoading model: {config.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Prepare model for training
    if config.use_4bit:
        model = prepare_model_for_kbit_training(model)

    # LoRA config
    print("\nConfiguring LoRA...")
    lora_config = get_lora_config(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # SFT Config
    print("\nConfiguring training...")
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.wandb_run_name,
        max_length=config.max_length,
        dataset_text_field="text",
        packing=False,
    )

    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)

    trainer.train()

    # Save final model
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    # Save LoRA adapters separately
    lora_output = os.path.join(config.output_dir, "lora_adapters")
    model.save_pretrained(lora_output)
    print(f"LoRA adapters saved to {lora_output}")

    if config.use_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)

    return config.output_dir
