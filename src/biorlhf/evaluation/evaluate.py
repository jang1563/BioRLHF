"""
Evaluation module for BioRLHF.

This module provides functionality for evaluating fine-tuned models on
biological reasoning tasks.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from biorlhf.utils.model_utils import get_quantization_config


@dataclass
class EvaluationResult:
    """Results from model evaluation."""

    overall_accuracy: float
    factual_accuracy: float
    reasoning_accuracy: float
    calibration_accuracy: float
    total_questions: int
    correct_answers: int
    detailed_results: List[Dict]


def evaluate_model(
    model_path: str,
    test_questions_path: str,
    base_model: str = "mistralai/Mistral-7B-v0.3",
    use_4bit: bool = True,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> EvaluationResult:
    """
    Evaluate a fine-tuned model on a test set.

    Args:
        model_path: Path to the fine-tuned model.
        test_questions_path: Path to JSON file with test questions.
        base_model: Base model name.
        use_4bit: Use 4-bit quantization.
        max_new_tokens: Maximum tokens to generate.
        temperature: Sampling temperature.

    Returns:
        EvaluationResult with accuracy metrics.
    """
    print(f"Loading model from {model_path}...")

    # Load quantization config
    bnb_config = get_quantization_config() if use_4bit else None

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    model = PeftModel.from_pretrained(model, model_path)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load test questions
    with open(test_questions_path, "r") as f:
        test_questions = json.load(f)

    print(f"Evaluating on {len(test_questions)} questions...")

    # Evaluate
    results = []
    category_correct = {"factual": 0, "reasoning": 0, "calibration": 0}
    category_total = {"factual": 0, "reasoning": 0, "calibration": 0}

    for q in test_questions:
        prompt = f"### Instruction:\n{q['question']}\n\n### Response:\n"

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()

        # Check correctness
        is_correct = _check_answer(response, q.get("expected_answer", ""), q.get("keywords", []))

        category = q.get("category", "factual")
        category_total[category] += 1
        if is_correct:
            category_correct[category] += 1

        results.append({
            "question": q["question"],
            "expected": q.get("expected_answer", ""),
            "response": response,
            "correct": is_correct,
            "category": category,
        })

    # Compute metrics
    total_correct = sum(category_correct.values())
    total_questions = sum(category_total.values())

    return EvaluationResult(
        overall_accuracy=total_correct / total_questions if total_questions > 0 else 0.0,
        factual_accuracy=category_correct["factual"] / category_total["factual"] if category_total["factual"] > 0 else 0.0,
        reasoning_accuracy=category_correct["reasoning"] / category_total["reasoning"] if category_total["reasoning"] > 0 else 0.0,
        calibration_accuracy=category_correct["calibration"] / category_total["calibration"] if category_total["calibration"] > 0 else 0.0,
        total_questions=total_questions,
        correct_answers=total_correct,
        detailed_results=results,
    )


def _check_answer(response: str, expected: str, keywords: List[str]) -> bool:
    """
    Check if a response is correct based on expected answer and keywords.

    Args:
        response: Model's response.
        expected: Expected answer (can be partial).
        keywords: Keywords that should appear in correct response.

    Returns:
        True if answer is considered correct.
    """
    response_lower = response.lower()

    # Check for keywords
    if keywords:
        return all(kw.lower() in response_lower for kw in keywords)

    # Check for expected answer substring
    if expected:
        return expected.lower() in response_lower

    return False


def compute_metrics(results: List[Dict]) -> Dict[str, float]:
    """
    Compute evaluation metrics from detailed results.

    Args:
        results: List of evaluation results with 'correct' and 'category' keys.

    Returns:
        Dictionary of metric names to values.
    """
    categories = set(r.get("category", "factual") for r in results)

    metrics = {}
    total_correct = 0
    total = 0

    for category in categories:
        category_results = [r for r in results if r.get("category") == category]
        correct = sum(1 for r in category_results if r.get("correct", False))
        total_cat = len(category_results)

        metrics[f"{category}_accuracy"] = correct / total_cat if total_cat > 0 else 0.0
        metrics[f"{category}_total"] = total_cat
        metrics[f"{category}_correct"] = correct

        total_correct += correct
        total += total_cat

    metrics["overall_accuracy"] = total_correct / total if total > 0 else 0.0
    metrics["total_questions"] = total
    metrics["total_correct"] = total_correct

    return metrics


def compare_models(
    model_paths: List[str],
    test_questions_path: str,
    base_model: str = "mistralai/Mistral-7B-v0.3",
    output_path: Optional[str] = None,
) -> Dict[str, EvaluationResult]:
    """
    Compare multiple models on the same test set.

    Args:
        model_paths: List of paths to fine-tuned models.
        test_questions_path: Path to test questions JSON.
        base_model: Base model name.
        output_path: Optional path to save comparison results.

    Returns:
        Dictionary mapping model paths to their evaluation results.
    """
    results = {}

    for model_path in model_paths:
        print(f"\nEvaluating {model_path}...")
        result = evaluate_model(
            model_path=model_path,
            test_questions_path=test_questions_path,
            base_model=base_model,
        )
        results[model_path] = result

        print(f"  Overall: {result.overall_accuracy:.1%}")
        print(f"  Factual: {result.factual_accuracy:.1%}")
        print(f"  Reasoning: {result.reasoning_accuracy:.1%}")
        print(f"  Calibration: {result.calibration_accuracy:.1%}")

    # Save comparison
    if output_path:
        comparison_data = {
            path: {
                "overall_accuracy": r.overall_accuracy,
                "factual_accuracy": r.factual_accuracy,
                "reasoning_accuracy": r.reasoning_accuracy,
                "calibration_accuracy": r.calibration_accuracy,
            }
            for path, r in results.items()
        }

        with open(output_path, "w") as f:
            json.dump(comparison_data, f, indent=2)

        print(f"\nComparison saved to {output_path}")

    return results
