#!/usr/bin/env python3
"""
BioRLHF SFT Training Example

This script demonstrates how to fine-tune a language model using
supervised fine-tuning (SFT) on biological reasoning tasks.

Requirements:
- CUDA-compatible GPU with 16GB+ VRAM (or use CPU with reduced batch size)
- PyTorch with CUDA support
- All BioRLHF dependencies installed

Usage:
    python train_sft.py [--config custom_config.json]
"""

import argparse
import json
from pathlib import Path

from biorlhf import SFTTrainingConfig, run_sft_training
from biorlhf.data.dataset import create_sft_dataset


def create_training_dataset(output_path: str = "training_dataset.json") -> str:
    """Create a training dataset if one doesn't exist."""
    path = Path(output_path)

    if path.exists():
        print(f"Using existing dataset: {output_path}")
        return output_path

    print(f"Creating new dataset: {output_path}")
    create_sft_dataset(
        output_path=output_path,
        include_calibration=True,
        include_chain_of_thought=True,
    )

    return output_path


def main():
    """Run SFT training."""
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for biological reasoning"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.3",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to training dataset (created if not provided)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./biorlhf_model",
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size per device",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="biorlhf",
        help="W&B project name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides other args)",
    )

    args = parser.parse_args()

    # Load config from file if provided
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = SFTTrainingConfig(**config_dict)
    else:
        # Create or use dataset
        dataset_path = args.dataset
        if dataset_path is None:
            dataset_path = create_training_dataset()

        # Build config from arguments
        config = SFTTrainingConfig(
            model_name=args.model,
            dataset_path=dataset_path,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
        )

    print("\nTraining Configuration:")
    print("-" * 40)
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    print("-" * 40)

    # Run training
    output_path = run_sft_training(config)

    print(f"\nModel saved to: {output_path}")
    print("\nTo evaluate the model, run:")
    print(f"  python evaluate_model.py --model {output_path}")


if __name__ == "__main__":
    main()
