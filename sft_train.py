#!/usr/bin/env python3
"""
BioRLHF SFT Training Script
Fine-tunes a language model on KMP biological reasoning tasks

Usage:
    python sft_train.py --model mistralai/Mistral-7B-v0.3 --epochs 3
"""

import argparse
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description='SFT Training for BioRLHF')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-7B-v0.3',
                       help='Base model to fine-tune')
    parser.add_argument('--dataset', type=str, default='kmp_sft_dataset.json',
                       help='Path to training dataset')
    parser.add_argument('--output_dir', type=str, default='./kmp_sft_model',
                       help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Per-device batch size')
    parser.add_argument('--grad_accum', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=2e-4,
                       help='Learning rate')
    parser.add_argument('--max_seq_length', type=int, default=2048,
                       help='Maximum sequence length')
    parser.add_argument('--lora_r', type=int, default=32,
                       help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=64,
                       help='LoRA alpha')
    parser.add_argument('--use_4bit', action='store_true', default=True,
                       help='Use 4-bit quantization')
    parser.add_argument('--wandb_project', type=str, default='biorlhf',
                       help='Weights & Biases project name')
    parser.add_argument('--wandb_run', type=str, default='kmp_sft_v1',
                       help='Weights & Biases run name')
    parser.add_argument('--no_wandb', action='store_true',
                       help='Disable Weights & Biases logging')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("BioRLHF SFT Training")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size} x {args.grad_accum} = {args.batch_size * args.grad_accum}")
    print("="*60)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run,
            config=vars(args)
        )
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset('json', data_files=args.dataset)['train']
    print(f"Dataset size: {len(dataset)} examples")
    
    # Split into train/eval
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Quantization config
    if args.use_4bit:
        print("\nUsing 4-bit quantization...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model
    print(f"\nLoading model: {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    
    # Prepare model for training
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    print("\nConfiguring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Training config
    training_args = SFTConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=50,
        eval_steps=50,
        eval_strategy="steps",
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=True,
        gradient_checkpointing=True,
        max_seq_length=args.max_seq_length,
        packing=False,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.wandb_run,
    )
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
    )
    
    # Train
    print("\n" + "="*60)
    print("Starting training...")
    print("="*60)
    
    trainer.train()
    
    # Save final model
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save LoRA adapters separately
    lora_output = os.path.join(args.output_dir, "lora_adapters")
    model.save_pretrained(lora_output)
    print(f"LoRA adapters saved to {lora_output}")
    
    if not args.no_wandb:
        wandb.finish()
    
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
