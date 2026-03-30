#!/usr/bin/env python3
"""
BioGRPO Post-Training Evaluation Script

Evaluates a GRPO-trained model against:
1. Held-out GeneLab questions (LOMO: Leave-One-Mission-Out)
2. Calibration metrics (ECE, Brier, overconfidence rate)
3. Per-verifier reward scores
4. Baseline comparison (SFT, DPO)

Usage:
    python scripts/evaluate_grpo.py \
        --model ./biogrpo_mve_model \
        --sft-baseline ./kmp_sft_model_final \
        --hold-out-tissues eye \
        --output results/grpo_mve_eval.json
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm

from biorlhf.data.grpo_dataset import build_grpo_dataset, get_dataset_stats
from biorlhf.verifiers.composer import VerifierComposer
from biorlhf.verifiers.uncertainty import _extract_confidence_simple, SimpleConfidence
from biorlhf.evaluation.calibration import compute_calibration_metrics


def load_model(
    model_path: str,
    base_model: str = "mistralai/Mistral-7B-v0.3",
    use_4bit: bool = True,
    sft_adapter: Optional[str] = None,
):
    """Load a fine-tuned model with LoRA adapters.

    For GRPO checkpoints trained on an SFT-merged base, pass sft_adapter
    to first merge the SFT adapter before applying the GRPO adapter.
    """
    print(f"  Base model: {base_model}")
    if sft_adapter:
        print(f"  SFT adapter (merge first): {sft_adapter}")
    print(f"  Adapter: {model_path}")

    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # If GRPO was trained on SFT-merged base, merge SFT first
    if sft_adapter:
        print("  Merging SFT adapter...")
        model = PeftModel.from_pretrained(model, sft_adapter)
        model = model.merge_and_unload()

    model = PeftModel.from_pretrained(model, model_path)

    # Always load tokenizer from base model (adapter dirs lack config.json)
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.1,
) -> str:
    """Generate a response from the model."""
    formatted = f"### Instruction:\n{prompt}\n\n### Response:\n"
    inputs = tokenizer(formatted, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=0.9,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    return response


def evaluate_with_verifiers(
    model,
    tokenizer,
    eval_dataset,
    composer: VerifierComposer,
    max_samples: Optional[int] = None,
) -> Dict:
    """Evaluate model using the verifier stack.

    Returns per-sample results and aggregated metrics.
    """
    results = []
    n = len(eval_dataset)
    if max_samples:
        n = min(n, max_samples)

    for i in tqdm(range(n), desc="Evaluating"):
        sample = eval_dataset[i]
        prompt = sample["prompt"]
        gt = sample["ground_truth"]
        qtype = sample["question_type"]
        applicable = sample["applicable_verifiers"]

        response = generate_response(model, tokenizer, prompt)

        reward = composer.compute_reward(
            prompt=prompt,
            completion=response,
            ground_truth=gt,
            question_type=qtype,
            applicable_verifiers=applicable,
        )

        # Extract confidence for calibration (match V4's extraction method)
        try:
            from bioeval.scoring.calibration import extract_confidence
            conf_extraction = extract_confidence(response)
            conf = SimpleConfidence(
                stated=conf_extraction.stated_confidence or "medium",
                numeric=conf_extraction.confidence_score,
                source="bioeval",
            )
        except ImportError:
            conf = _extract_confidence_simple(response)

        results.append({
            "prompt": prompt[:100],
            "response": response[:300],
            "total_reward": reward.total_reward,
            "verifier_scores": reward.verifier_scores,
            "question_type": qtype,
            "source": sample.get("source", "unknown"),
            "tissue": sample.get("tissue", "unknown"),
            "confidence": conf.numeric,
            "confidence_stated": conf.stated,
        })

    # Aggregate metrics
    total_rewards = [r["total_reward"] for r in results]
    per_verifier: Dict[str, List[float]] = {}
    for r in results:
        for v, s in r["verifier_scores"].items():
            per_verifier.setdefault(v, []).append(s)

    verifier_means = {v: sum(s) / len(s) for v, s in per_verifier.items()}

    # Per question type
    by_type: Dict[str, List[float]] = {}
    for r in results:
        by_type.setdefault(r["question_type"], []).append(r["total_reward"])
    type_means = {t: sum(s) / len(s) for t, s in by_type.items()}

    return {
        "n_samples": len(results),
        "mean_reward": sum(total_rewards) / len(total_rewards) if total_rewards else 0,
        "verifier_means": verifier_means,
        "by_question_type": type_means,
        "per_sample": results,
    }


def evaluate_calibration(results: List[Dict]) -> Dict:
    """Compute calibration metrics from evaluation results."""
    confidences = [r["confidence"] for r in results]

    # Correctness: reward > 0.5 considered "correct"
    correctnesses = [r["total_reward"] > 0.5 for r in results]

    metrics = compute_calibration_metrics(
        confidences=confidences,
        correctnesses=correctnesses,
    )

    return {
        "ece": metrics.ece,
        "mce": metrics.mce,
        "brier_score": metrics.brier_score,
        "overconfidence_rate": metrics.overconfidence_rate,
        "underconfidence_rate": metrics.underconfidence_rate,
        "mean_confidence": metrics.mean_confidence,
        "mean_accuracy": metrics.mean_accuracy,
        "n_samples": metrics.n_samples,
        "reliability_bins": metrics.reliability_bins,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a BioGRPO-trained model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to the GRPO-trained model (LoRA adapter directory)",
    )
    parser.add_argument(
        "--base-model", type=str, default="mistralai/Mistral-7B-v0.3",
        help="Base model name",
    )
    parser.add_argument(
        "--sft-baseline", type=str, default=None,
        help="Path to SFT baseline model for comparison",
    )
    parser.add_argument(
        "--hold-out-tissues", type=str, nargs="+", default=["eye"],
        help="Tissues held out for evaluation",
    )
    parser.add_argument(
        "--pathway-db", type=str, default="hallmark",
        help="Pathway database",
    )
    parser.add_argument(
        "--active-verifiers", type=str, nargs="+", default=None,
        help="Active verifiers (default: all)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Max samples to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path for results JSON",
    )
    parser.add_argument(
        "--no-4bit", action="store_true",
        help="Disable 4-bit quantization",
    )
    parser.add_argument(
        "--sft-adapter", type=str, default=None,
        help="Path to SFT LoRA adapter to merge before applying GRPO adapter (for GRPO checkpoints trained on SFT-merged base)",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("BioGRPO Evaluation")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Base:           {args.base_model}")
    print(f"  Hold-out:       {args.hold_out_tissues}")
    print(f"  SFT baseline:   {args.sft_baseline or 'None'}")
    print(f"  Time:           {datetime.now().isoformat()}")
    print("=" * 60)

    # Build eval dataset
    print("\n[1/4] Building evaluation dataset...")
    _, eval_dataset = build_grpo_dataset(
        db=args.pathway_db,
        hold_out_tissues=args.hold_out_tissues,
    )
    eval_stats = get_dataset_stats(eval_dataset)
    print(f"  Eval samples: {eval_stats['total']}")
    print(f"  By source: {eval_stats['by_source']}")
    print(f"  By type: {eval_stats['by_question_type']}")

    # Create verifier composer
    composer = VerifierComposer(active_verifiers=args.active_verifiers)

    # Evaluate GRPO model
    print(f"\n[2/4] Evaluating GRPO model: {args.model}")
    model, tokenizer = load_model(
        args.model, args.base_model, use_4bit=not args.no_4bit,
        sft_adapter=args.sft_adapter,
    )
    grpo_results = evaluate_with_verifiers(
        model, tokenizer, eval_dataset, composer,
        max_samples=args.max_samples,
    )
    grpo_calibration = evaluate_calibration(grpo_results["per_sample"])

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

    # Evaluate baseline if provided
    baseline_results = None
    baseline_calibration = None
    if args.sft_baseline:
        print(f"\n[3/4] Evaluating SFT baseline: {args.sft_baseline}")
        baseline_model, baseline_tokenizer = load_model(
            args.sft_baseline, args.base_model, use_4bit=not args.no_4bit,
        )
        baseline_results = evaluate_with_verifiers(
            baseline_model, baseline_tokenizer, eval_dataset, composer,
            max_samples=args.max_samples,
        )
        baseline_calibration = evaluate_calibration(baseline_results["per_sample"])
        del baseline_model
        torch.cuda.empty_cache()
    else:
        print("\n[3/4] Skipping baseline (not provided)")

    # Print summary
    print("\n[4/4] Results Summary")
    print("=" * 60)
    print(f"GRPO Model: {args.model}")
    print(f"  Mean reward:     {grpo_results['mean_reward']:.3f}")
    print(f"  Per verifier:    {grpo_results['verifier_means']}")
    print(f"  ECE:             {grpo_calibration['ece']:.3f}")
    print(f"  Brier:           {grpo_calibration['brier_score']:.3f}")
    print(f"  Overconfidence:  {grpo_calibration['overconfidence_rate']:.3f}")
    print(f"  By type:         {grpo_results['by_question_type']}")

    comparison = {}
    if baseline_results:
        print(f"\nSFT Baseline: {args.sft_baseline}")
        print(f"  Mean reward:     {baseline_results['mean_reward']:.3f}")
        print(f"  ECE:             {baseline_calibration['ece']:.3f}")
        print(f"  Brier:           {baseline_calibration['brier_score']:.3f}")

        delta_reward = grpo_results["mean_reward"] - baseline_results["mean_reward"]
        delta_ece = grpo_calibration["ece"] - baseline_calibration["ece"]
        print(f"\n  Delta reward:    {delta_reward:+.3f}")
        print(f"  Delta ECE:       {delta_ece:+.3f} (negative = better)")

        comparison = {
            "sft_mean_reward": baseline_results["mean_reward"],
            "sft_ece": baseline_calibration["ece"],
            "delta_reward": delta_reward,
            "delta_ece": delta_ece,
        }

    # Success criteria
    criteria = {
        "reward_above_05": grpo_results["mean_reward"] > 0.5,
        "ece_below_015": grpo_calibration["ece"] < 0.15,
    }
    if baseline_results:
        criteria["reward_above_baseline"] = delta_reward > 0
    criteria["overall_pass"] = all(criteria.values())

    print(f"\nSuccess criteria: {criteria}")

    # Save results
    output_path = args.output or f"results/grpo_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "model_path": args.model,
        "base_model": args.base_model,
        "evaluation_date": datetime.now().isoformat(),
        "hold_out_tissues": args.hold_out_tissues,
        "eval_dataset_stats": eval_stats,
        "grpo": {
            "mean_reward": grpo_results["mean_reward"],
            "verifier_means": grpo_results["verifier_means"],
            "by_question_type": grpo_results["by_question_type"],
            "n_samples": grpo_results["n_samples"],
        },
        "calibration": grpo_calibration,
        "baseline_comparison": comparison,
        "success_criteria": criteria,
        "per_sample": grpo_results["per_sample"],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
