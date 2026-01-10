#!/usr/bin/env python3
"""
BioRLHF Model Evaluation Example

This script demonstrates how to evaluate a fine-tuned model on
biological reasoning tasks.

Usage:
    python evaluate_model.py --model ./biorlhf_model --test-set kmp_test_set.json
"""

import argparse
import json
from pathlib import Path

from biorlhf import evaluate_model


def main():
    """Run model evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a fine-tuned BioRLHF model"
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
        default="kmp_test_set.json",
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

    # Check if test set exists
    if not Path(args.test_set).exists():
        print(f"Error: Test set not found at {args.test_set}")
        print("\nYou can create a test set or use the default one from the data folder.")
        return

    print("=" * 60)
    print("BioRLHF Model Evaluation")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Base Model: {args.base_model}")
    print(f"Test Set: {args.test_set}")
    print(f"Quantization: {'Disabled' if args.no_quantization else '4-bit'}")
    print("=" * 60)

    # Run evaluation
    results = evaluate_model(
        model_path=args.model,
        test_questions_path=args.test_set,
        base_model=args.base_model,
        use_4bit=not args.no_quantization,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
    )

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"\nOverall Accuracy: {results.overall_accuracy:.1%} ({results.correct_answers}/{results.total_questions})")
    print(f"\nBy Category:")
    print(f"  Factual:     {results.factual_accuracy:.1%}")
    print(f"  Reasoning:   {results.reasoning_accuracy:.1%}")
    print(f"  Calibration: {results.calibration_accuracy:.1%}")

    # Show detailed results
    print("\n" + "-" * 60)
    print("Detailed Results:")
    print("-" * 60)

    for i, r in enumerate(results.detailed_results, 1):
        status = "CORRECT" if r["correct"] else "WRONG"
        print(f"\n{i}. [{r['category'].upper()}] {status}")
        print(f"   Q: {r['question'][:80]}...")
        print(f"   Expected: {r['expected'][:50]}..." if len(r["expected"]) > 50 else f"   Expected: {r['expected']}")
        print(f"   Response: {r['response'][:100]}..." if len(r["response"]) > 100 else f"   Response: {r['response']}")

    # Save detailed results if requested
    if args.output:
        output_data = {
            "model_path": args.model,
            "base_model": args.base_model,
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

    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
