#!/usr/bin/env python3
"""
BioRLHF DPO Training Script
Direct Preference Optimization on biological reasoning

Usage:
    python dpo_train.py --sft_model ./kmp_sft_model_v2
"""

import argparse
import os
import torch
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, PeftModel, prepare_model_for_kbit_training
from trl import DPOTrainer, DPOConfig
import wandb
import json


def parse_args():
    parser = argparse.ArgumentParser(description='DPO Training for BioRLHF')
    parser.add_argument('--sft_model', type=str, default='./kmp_sft_model_v2',
                       help='Path to SFT fine-tuned model')
    parser.add_argument('--base_model', type=str, default='mistralai/Mistral-7B-v0.3',
                       help='Base model name')
    parser.add_argument('--dataset', type=str, default='kmp_dpo_preferences.json',
                       help='Path to preference dataset')
    parser.add_argument('--output_dir', type=str, default='./kmp_dpo_model',
                       help='Output directory')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=2,
                       help='Per-device batch size')
    parser.add_argument('--grad_accum', type=int, default=4,
                       help='Gradient accumulation steps')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1,
                       help='DPO beta parameter')
    parser.add_argument('--max_length', type=int, default=1024,
                       help='Maximum sequence length')
    parser.add_argument('--wandb_project', type=str, default='biorlhf')
    parser.add_argument('--wandb_run', type=str, default='kmp_dpo_v1')
    parser.add_argument('--no_wandb', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("="*60)
    print("BioRLHF DPO Training")
    print("="*60)
    print(f"SFT Model: {args.sft_model}")
    print(f"Base Model: {args.base_model}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {args.output_dir}")
    print(f"Beta: {args.beta}")
    print("="*60)
    
    # Initialize wandb
    if not args.no_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run, config=vars(args))
    
    # Load preference dataset
    print("\nLoading preference dataset...")
    with open(args.dataset, 'r') as f:
        raw_data = json.load(f)
    
    dataset = Dataset.from_list(raw_data)
    print(f"Preference pairs: {len(dataset)}")
    
    # Split
    dataset = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = dataset['train']
    eval_dataset = dataset['test']
    print(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
    
    # Quantization config
    print("\nUsing 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load base model
    print(f"\nLoading base model: {args.base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    
    # Load SFT LoRA adapters
    print(f"\nLoading SFT adapters from: {args.sft_model}")
    model = PeftModel.from_pretrained(model, args.sft_model)
    model = model.merge_and_unload()  # Merge SFT adapters into base
    
    # Prepare for new LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.sft_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # DPO needs left padding
    
    # New LoRA config for DPO
    print("\nConfiguring LoRA for DPO...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Reference model (frozen copy)
    print("\nLoading reference model...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    ref_model = PeftModel.from_pretrained(ref_model, args.sft_model)
    ref_model = ref_model.merge_and_unload()
    
    # DPO Config
    print("\nConfiguring DPO training...")
    dpo_config = DPOConfig(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        logging_steps=5,
        save_steps=25,
        eval_steps=25,
        eval_strategy="steps",
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        report_to="wandb" if not args.no_wandb else "none",
        run_name=args.wandb_run,
        max_length=args.max_length,
        max_prompt_length=512,
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
    print("\n" + "="*60)
    print("Starting DPO training...")
    print("="*60)
    
    trainer.train()
    
    # Save
    print(f"\nSaving model to {args.output_dir}")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    if not args.no_wandb:
        wandb.finish()
    
    print("\n" + "="*60)
    print("DPO Training complete!")
    print("="*60)


if __name__ == "__main__":
    main()
