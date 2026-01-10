"""
Direct Preference Optimization (DPO) module for BioRLHF.

This module provides functionality for aligning language models using
preference learning on biological reasoning tasks.
"""

import json
from dataclasses import dataclass
from typing import Optional
import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
import wandb

from biorlhf.utils.model_utils import get_quantization_config


@dataclass
class DPOTrainingConfig:
    """Configuration for DPO training."""

    # Model settings
    sft_model_path: str = "./biorlhf_sft_model"
    base_model: str = "mistralai/Mistral-7B-v0.3"
    dataset_path: str = "kmp_dpo_preferences.json"
    output_dir: str = "./biorlhf_dpo_model"

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    beta: float = 0.1  # DPO regularization parameter
    max_length: int = 1024
    max_prompt_length: int = 512
    warmup_ratio: float = 0.1

    # LoRA settings (typically smaller for DPO)
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05

    # Logging
    logging_steps: int = 5
    save_steps: int = 25
    eval_steps: int = 25
    save_total_limit: int = 2

    # Weights & Biases
    wandb_project: str = "biorlhf"
    wandb_run_name: str = "dpo_training"
    use_wandb: bool = True

    # Evaluation
    test_size: float = 0.1
    seed: int = 42


def run_dpo_training(config: Optional[DPOTrainingConfig] = None) -> str:
    """
    Run DPO training with the given configuration.

    Note: DPO can be fragile for domain-specific knowledge. Use conservative
    beta values (0.1-0.3) to avoid catastrophic forgetting.

    Args:
        config: Training configuration. If None, uses defaults.

    Returns:
        Path to the saved model.
    """
    if config is None:
        config = DPOTrainingConfig()

    print("=" * 60)
    print("BioRLHF DPO Training")
    print("=" * 60)
    print(f"SFT Model: {config.sft_model_path}")
    print(f"Base Model: {config.base_model}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    print(f"Beta: {config.beta}")
    print("=" * 60)

    # Initialize wandb
    if config.use_wandb:
        wandb.init(
            project=config.wandb_project,
            name=config.wandb_run_name,
            config=vars(config),
        )

    # Load preference dataset
    print("\nLoading preference dataset...")
    with open(config.dataset_path, "r") as f:
        raw_data = json.load(f)

    dataset = Dataset.from_list(raw_data)
    print(f"Preference pairs: {len(dataset)}")

    # Split
    dataset = dataset.train_test_split(test_size=config.test_size, seed=config.seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    # Quantization config
    print("\nUsing 4-bit quantization...")
    bnb_config = get_quantization_config()

    # Load base model
    print(f"\nLoading base model: {config.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    # Load SFT LoRA adapters
    print(f"\nLoading SFT adapters from: {config.sft_model_path}")
    model = PeftModel.from_pretrained(model, config.sft_model_path)
    model = model.merge_and_unload()  # Merge SFT adapters into base

    # Prepare for new LoRA training
    model = prepare_model_for_kbit_training(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO needs left padding

    # New LoRA config for DPO
    print("\nConfiguring LoRA for DPO...")
    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Reference model (frozen copy)
    print("\nLoading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    ref_model = PeftModel.from_pretrained(ref_model, config.sft_model_path)
    ref_model = ref_model.merge_and_unload()

    # DPO Config
    print("\nConfiguring DPO training...")
    dpo_config = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        beta=config.beta,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        eval_steps=config.eval_steps,
        eval_strategy="steps",
        save_total_limit=config.save_total_limit,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.wandb_run_name,
        max_length=config.max_length,
        max_prompt_length=config.max_prompt_length,
    )

    # Create DPO Trainer
    print("\nInitializing DPO trainer...")
    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    # Train
    print("\n" + "=" * 60)
    print("Starting DPO training...")
    print("=" * 60)

    trainer.train()

    # Save
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)

    if config.use_wandb:
        wandb.finish()

    print("\n" + "=" * 60)
    print("DPO Training complete!")
    print("=" * 60)

    return config.output_dir
