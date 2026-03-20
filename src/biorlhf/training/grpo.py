"""
Group Relative Policy Optimization (GRPO) training for BioGRPO.

Uses TRL's GRPOTrainer with composable biological verifiers as reward functions.
Supports configurable G values, verifier weights, and LoRA parameters.
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, PeftModel
from trl import GRPOTrainer, GRPOConfig

from biorlhf.verifiers.composer import make_grpo_reward_function
from biorlhf.data.grpo_dataset import build_grpo_dataset, get_dataset_stats


@dataclass
class BioGRPOConfig:
    """Configuration for BioGRPO training."""

    # Model settings
    model_name: str = "mistralai/Mistral-7B-v0.3"
    sft_model_path: Optional[str] = None
    output_dir: str = "./biogrpo_model"

    # GRPO hyperparameters
    num_generations: int = 8
    beta: float = 0.04
    num_iterations: int = 1
    scale_rewards: str = "group"
    loss_type: str = "grpo"

    # Training hyperparameters
    num_epochs: int = 1
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-6
    max_completion_length: int = 1024
    max_prompt_length: int = 512
    warmup_ratio: float = 0.1

    # LoRA settings
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05

    # Verifier configuration
    verifier_weights: Optional[Dict[str, float]] = None
    active_verifiers: Optional[List[str]] = None

    # Data
    pathway_db: str = "hallmark"
    hold_out_tissues: Optional[List[str]] = None
    seed: int = 42

    # Quantization
    use_4bit: bool = True

    # Logging
    wandb_project: str = "biogrpo"
    wandb_run_name: str = "grpo_v1"
    use_wandb: bool = True
    logging_steps: int = 5
    save_steps: int = 50
    eval_steps: int = 50
    save_total_limit: int = 3
    log_completions: bool = True

    # Memory optimization
    use_vllm: bool = False
    gradient_checkpointing: bool = True
    bf16: bool = True


def run_grpo_training(config: Optional[BioGRPOConfig] = None) -> str:
    """Run BioGRPO training.

    Pipeline:
      1. Build dataset from GeneLab + BioEval + SpaceOmicsBench
      2. Create composed reward function from verifier stack
      3. Load tokenizer and configure GRPOTrainer with LoRA
      4. Train and save model

    Args:
        config: Training configuration. Uses defaults if None.

    Returns:
        Path to the saved model directory.
    """
    if config is None:
        config = BioGRPOConfig()

    print("=" * 60)
    print("BioGRPO Training")
    print("=" * 60)
    print(f"  Model:           {config.model_name}")
    print(f"  SFT checkpoint:  {config.sft_model_path or 'None (from base)'}")
    print(f"  G (generations): {config.num_generations}")
    print(f"  Beta (KL):       {config.beta}")
    print(f"  Loss type:       {config.loss_type}")
    print(f"  Active verifiers:{config.active_verifiers or 'all (V1-V4)'}")
    print(f"  Verifier weights:{config.verifier_weights or 'default'}")
    print(f"  LoRA r/alpha:    {config.lora_r}/{config.lora_alpha}")
    print(f"  Learning rate:   {config.learning_rate}")
    print(f"  QLoRA 4-bit:     {config.use_4bit}")
    print(f"  Output:          {config.output_dir}")
    print("=" * 60)

    # Initialize wandb
    if config.use_wandb:
        try:
            import wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config={k: v for k, v in vars(config).items() if not k.startswith("_")},
            )
        except ImportError:
            print("Warning: wandb not installed, disabling logging")
            config.use_wandb = False

    # 1. Build dataset
    print("\n[1/5] Building GRPO dataset...")
    train_dataset, eval_dataset = build_grpo_dataset(
        db=config.pathway_db,
        seed=config.seed,
        hold_out_tissues=config.hold_out_tissues,
    )
    train_stats = get_dataset_stats(train_dataset)
    eval_stats = get_dataset_stats(eval_dataset)
    print(f"  Train: {train_stats['total']} samples")
    print(f"    By source: {train_stats['by_source']}")
    print(f"    By type:   {train_stats['by_question_type']}")
    print(f"  Eval:  {eval_stats['total']} samples")

    # 2. Create reward function
    print("\n[2/5] Initializing verifier stack...")
    reward_func = make_grpo_reward_function(
        weights=config.verifier_weights,
        active_verifiers=config.active_verifiers,
    )
    print(f"  Active: {config.active_verifiers or ['V1', 'V2', 'V3', 'V4']}")

    # 3. Load tokenizer (always from base model; adapter dirs lack config.json)
    print("\n[3/5] Loading tokenizer...")
    tokenizer_source = config.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"  Tokenizer: {tokenizer.__class__.__name__}, vocab={tokenizer.vocab_size}")

    # 4. Configure LoRA
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 5. Load model (merge SFT adapter if present)
    print("\n[4/5] Loading model...")

    # QLoRA quantization config
    bnb_config = None
    if config.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    # Check if sft_model_path is a LoRA adapter or a full model
    sft_is_adapter = (
        config.sft_model_path
        and os.path.isdir(config.sft_model_path)
        and os.path.exists(os.path.join(config.sft_model_path, "adapter_config.json"))
    )

    if sft_is_adapter:
        # Load base model, merge SFT adapter, then apply fresh LoRA for GRPO
        print(f"  Loading base model: {config.model_name}")
        base_model = AutoModelForCausalLM.from_pretrained(
            config.model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        print(f"  Loading SFT LoRA adapter: {config.sft_model_path}")
        model = PeftModel.from_pretrained(base_model, config.sft_model_path)
        print("  Merging SFT adapter into base model...")
        model = model.merge_and_unload()
        print("  SFT adapter merged successfully")
    else:
        # sft_model_path is a full model or use base model
        model_path = config.sft_model_path or config.model_name
        print(f"  Loading model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # 6. Configure GRPOTrainer
    print("\n[5/6] Configuring GRPOTrainer...")

    grpo_config = GRPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type="cosine",

        # GRPO-specific
        num_generations=config.num_generations,
        beta=config.beta,
        loss_type=config.loss_type,
        max_completion_length=config.max_completion_length,
        max_prompt_length=config.max_prompt_length,
        num_iterations=config.num_iterations,
        scale_rewards=config.scale_rewards,

        # Memory/compute
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=config.bf16,
        use_vllm=config.use_vllm,

        # Logging
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        report_to="wandb" if config.use_wandb else "none",
        run_name=config.wandb_run_name,

        # Evaluation
        eval_strategy="steps",
        eval_steps=config.eval_steps,
        log_completions=config.log_completions,
    )

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        reward_funcs=reward_func,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )

    # Train
    print("\n[6/6] Starting GRPO training...")
    print("=" * 60)

    trainer.train()

    # Save
    print(f"\nSaving model to {config.output_dir}")
    trainer.save_model(config.output_dir)

    if config.use_wandb:
        try:
            import wandb
            wandb.finish()
        except ImportError:
            pass

    print("\n" + "=" * 60)
    print("BioGRPO Training complete!")
    print("=" * 60)

    return config.output_dir
