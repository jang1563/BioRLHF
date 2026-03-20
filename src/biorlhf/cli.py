"""
Command-line interface for BioRLHF.

This module provides CLI entry points for training and evaluating models.
"""

import argparse
import json
import sys
from pathlib import Path

from biorlhf.training.sft import SFTTrainingConfig, run_sft_training
from biorlhf.evaluation.evaluate import evaluate_model as _evaluate_model
from biorlhf.training.grpo import BioGRPOConfig, run_grpo_training


def train():
    """CLI entry point for training models."""
    parser = argparse.ArgumentParser(
        description="Train a BioRLHF model using supervised fine-tuning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.3",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to training dataset JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./biorlhf_model",
        help="Output directory for trained model",
    )

    # Training hyperparameters
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
        "--max-length",
        type=int,
        default=1024,
        help="Maximum sequence length",
    )

    # LoRA settings
    parser.add_argument(
        "--lora-r",
        type=int,
        default=64,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=128,
        help="LoRA alpha",
    )

    # Other settings
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 4-bit quantization",
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
        "--wandb-run-name",
        type=str,
        default="sft_training",
        help="W&B run name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides other args)",
    )

    args = parser.parse_args()

    # Validate dataset path
    if not Path(args.dataset).exists():
        print(f"Error: Dataset not found at {args.dataset}", file=sys.stderr)
        sys.exit(1)

    # Load config from file if provided
    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = SFTTrainingConfig(**config_dict)
    else:
        config = SFTTrainingConfig(
            model_name=args.model,
            dataset_path=args.dataset,
            output_dir=args.output,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            max_length=args.max_length,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            use_4bit=not args.no_quantization,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )

    print("BioRLHF Training")
    print("=" * 50)
    print(f"Model: {config.model_name}")
    print(f"Dataset: {config.dataset_path}")
    print(f"Output: {config.output_dir}")
    print("=" * 50)

    try:
        output_path = run_sft_training(config)
        print(f"\nModel saved to: {output_path}")
    except Exception as e:
        print(f"Error during training: {e}", file=sys.stderr)
        sys.exit(1)


def evaluate():
    """CLI entry point for evaluating models."""
    parser = argparse.ArgumentParser(
        description="Evaluate a BioRLHF model on a test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory",
    )
    parser.add_argument(
        "--test-set",
        type=str,
        required=True,
        help="Path to test questions JSON file",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="mistralai/Mistral-7B-v0.3",
        help="Base model name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for detailed results JSON",
    )
    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Generation temperature (0 for greedy)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate",
    )

    args = parser.parse_args()

    # Validate paths
    if not Path(args.model).exists():
        print(f"Error: Model not found at {args.model}", file=sys.stderr)
        sys.exit(1)

    if not Path(args.test_set).exists():
        print(f"Error: Test set not found at {args.test_set}", file=sys.stderr)
        sys.exit(1)

    print("BioRLHF Evaluation")
    print("=" * 50)
    print(f"Model: {args.model}")
    print(f"Test Set: {args.test_set}")
    print("=" * 50)

    try:
        results = _evaluate_model(
            model_path=args.model,
            test_questions_path=args.test_set,
            base_model=args.base_model,
            use_4bit=not args.no_quantization,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
        )

        print("\nResults:")
        print("-" * 30)
        print(f"Overall Accuracy: {results.overall_accuracy:.1%}")
        print(f"Factual Accuracy: {results.factual_accuracy:.1%}")
        print(f"Reasoning Accuracy: {results.reasoning_accuracy:.1%}")
        print(f"Calibration Accuracy: {results.calibration_accuracy:.1%}")
        print(f"Total: {results.correct_answers}/{results.total_questions}")

        # Save detailed results if requested
        if args.output:
            output_data = {
                "model_path": args.model,
                "test_set": args.test_set,
                "metrics": {
                    "overall_accuracy": results.overall_accuracy,
                    "factual_accuracy": results.factual_accuracy,
                    "reasoning_accuracy": results.reasoning_accuracy,
                    "calibration_accuracy": results.calibration_accuracy,
                    "total_questions": results.total_questions,
                    "correct_answers": results.correct_answers,
                },
                "detailed_results": results.detailed_results,
            }

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2)

            print(f"\nDetailed results saved to: {args.output}")

    except Exception as e:
        print(f"Error during evaluation: {e}", file=sys.stderr)
        sys.exit(1)


def grpo_train():
    """CLI entry point for GRPO training with biological verifiers."""
    parser = argparse.ArgumentParser(
        description="Train a BioGRPO model with composable biological verifiers",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="mistralai/Mistral-7B-v0.3",
        help="Base model to fine-tune",
    )
    parser.add_argument(
        "--sft-model",
        type=str,
        default=None,
        help="Path to SFT checkpoint (recommended)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./biogrpo_model",
        help="Output directory",
    )
    parser.add_argument(
        "--num-generations",
        type=int,
        default=8,
        help="G value: number of completions per prompt",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.04,
        help="KL penalty coefficient",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-6,
        help="Learning rate",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=32,
        help="LoRA rank",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=64,
        help="LoRA alpha",
    )
    parser.add_argument(
        "--verifiers",
        type=str,
        nargs="+",
        default=None,
        help="Active verifiers (e.g., V1 V2 V3 V4). Default: all",
    )
    parser.add_argument(
        "--pathway-db",
        type=str,
        default="hallmark",
        choices=["hallmark", "kegg", "reactome", "mitocarta"],
        help="Pathway database for GeneLab questions",
    )
    parser.add_argument(
        "--no-wandb",
        action="store_true",
        help="Disable W&B logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="biogrpo",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default="grpo_v1",
        help="W&B run name",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file (overrides other args)",
    )

    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            config_dict = json.load(f)
        config = BioGRPOConfig(**config_dict)
    else:
        config = BioGRPOConfig(
            model_name=args.model,
            sft_model_path=args.sft_model,
            output_dir=args.output,
            num_generations=args.num_generations,
            beta=args.beta,
            learning_rate=args.learning_rate,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            active_verifiers=args.verifiers,
            pathway_db=args.pathway_db,
            use_wandb=not args.no_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
        )

    try:
        output_path = run_grpo_training(config)
        print(f"\nModel saved to: {output_path}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error during GRPO training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    print("Use 'biorlhf-train', 'biorlhf-evaluate', or 'biorlhf-grpo' commands after installation.")
