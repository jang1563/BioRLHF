#!/usr/bin/env python3
"""
Evaluate Ecosystem-Improved Model on Failure Patterns

This script evaluates the fine-tuned model specifically on the patterns
it was trained to improve:
- Calibration (uncertainty expression)
- Adversarial resistance
- Protocol completeness
- Fact recall

Usage (on HPC with GPU):
    python scripts/evaluate_ecosystem_model.py --model ./ecosystem_improved_model

Requirements:
    - CUDA GPU
    - transformers, peft, bitsandbytes, torch
"""

import argparse
import json
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel


def load_model(model_path: str, base_model: str = "mistralai/Mistral-7B-v0.3", use_4bit: bool = True):
    """Load the fine-tuned model with LoRA adapters."""
    print(f"Loading base model: {base_model}")

    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            device_map="auto",
            torch_dtype=torch.float16,
            trust_remote_code=True,
        )

    print(f"Loading LoRA adapters from: {model_path}")
    model = PeftModel.from_pretrained(model, model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Generate a response from the model."""
    formatted_prompt = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract just the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()

    return response


def evaluate_calibration(model, tokenizer, test_cases: list) -> dict:
    """Evaluate uncertainty calibration - does model admit when it doesn't know?"""
    print("\n" + "="*60)
    print("EVALUATING: Calibration (Uncertainty Expression)")
    print("="*60)

    results = []
    calibration_phrases = [
        "cannot determine",
        "not provided",
        "not included",
        "would need",
        "don't have",
        "doesn't include",
        "cannot say",
        "uncertain",
        "not specified",
        "additional information",
        "not in the data",
    ]

    for i, case in enumerate(test_cases):
        question = case["conversations"][0]["content"]
        expected = case["conversations"][1]["content"]

        print(f"\n[{i+1}] Testing: {question[:60]}...")
        response = generate_response(model, tokenizer, question)

        # Check if response expresses appropriate uncertainty
        expresses_uncertainty = any(phrase.lower() in response.lower() for phrase in calibration_phrases)

        result = {
            "id": case["id"],
            "question": question,
            "expected_behavior": "Express uncertainty about unavailable data",
            "response": response[:500],
            "expresses_uncertainty": expresses_uncertainty,
            "passed": expresses_uncertainty
        }
        results.append(result)

        status = "PASS" if expresses_uncertainty else "FAIL"
        print(f"    [{status}] Uncertainty expressed: {expresses_uncertainty}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    return {
        "category": "calibration",
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total > 0 else 0,
        "details": results
    }


def evaluate_adversarial(model, tokenizer, test_cases: list) -> dict:
    """Evaluate adversarial resistance - does model reject false claims?"""
    print("\n" + "="*60)
    print("EVALUATING: Adversarial Resistance")
    print("="*60)

    results = []
    resistance_phrases = [
        "not supported",
        "contradicted",
        "incorrect",
        "need to correct",
        "not true",
        "actually shows",
        "data shows",
        "opposite",
        "challenge that premise",
        "misrepresent",
    ]

    for i, case in enumerate(test_cases):
        question = case["conversations"][0]["content"]
        expected = case["conversations"][1]["content"]

        print(f"\n[{i+1}] Testing adversarial: {question[:60]}...")
        response = generate_response(model, tokenizer, question)

        # Check if response resists the false claim
        resists_claim = any(phrase.lower() in response.lower() for phrase in resistance_phrases)

        result = {
            "id": case["id"],
            "question": question,
            "expected_behavior": "Reject false premise with evidence",
            "response": response[:500],
            "resists_false_claim": resists_claim,
            "passed": resists_claim
        }
        results.append(result)

        status = "PASS" if resists_claim else "FAIL"
        print(f"    [{status}] Resisted false claim: {resists_claim}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    return {
        "category": "adversarial_resistance",
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total > 0 else 0,
        "details": results
    }


def evaluate_completeness(model, tokenizer, test_cases: list) -> dict:
    """Evaluate protocol completeness - does model detect all missing steps?"""
    print("\n" + "="*60)
    print("EVALUATING: Protocol Completeness")
    print("="*60)

    results = []

    # Key missing steps that should be detected
    key_steps = {
        "comp_001": ["dnase", "reverse transcription", "rt", "cdna"],
        "comp_002": ["transfer", "blot", "membrane transfer"]
    }

    for i, case in enumerate(test_cases):
        question = case["conversations"][0]["content"]
        expected = case["conversations"][1]["content"]
        case_id = case["id"]

        print(f"\n[{i+1}] Testing completeness: {case_id}...")
        response = generate_response(model, tokenizer, question, max_new_tokens=800)

        # Check if key missing steps are detected
        expected_steps = key_steps.get(case_id, [])
        response_lower = response.lower()
        detected = [step for step in expected_steps if step in response_lower]
        detection_rate = len(detected) / len(expected_steps) if expected_steps else 0

        result = {
            "id": case_id,
            "question": question[:100],
            "expected_steps": expected_steps,
            "detected_steps": detected,
            "response": response[:600],
            "detection_rate": detection_rate,
            "passed": detection_rate >= 0.5  # Pass if at least half detected
        }
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"    [{status}] Detected {len(detected)}/{len(expected_steps)} key steps")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    return {
        "category": "protocol_completeness",
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total > 0 else 0,
        "details": results
    }


def evaluate_fact_recall(model, tokenizer, test_cases: list) -> dict:
    """Evaluate fact recall - does model remember key trained facts?"""
    print("\n" + "="*60)
    print("EVALUATING: Fact Recall")
    print("="*60)

    results = []

    # Key facts and values that should be recalled
    key_facts = {
        "fact_001": ["52%", "52 percent"],
        "fact_002": ["52%", "52 percent"],
        "fact_003": ["52%", "8%"],
        "fact_004": ["-1.60", "-1.6", "suppressed", "suppression"],
        "fact_005": ["liver", "-1.60", "-1.6", "opposite"]
    }

    for i, case in enumerate(test_cases):
        question = case["conversations"][0]["content"]
        expected = case["conversations"][1]["content"]
        case_id = case["id"]

        print(f"\n[{i+1}] Testing fact recall: {case_id}...")
        response = generate_response(model, tokenizer, question)

        # Check if key facts are present
        expected_facts = key_facts.get(case_id, [])
        response_lower = response.lower()
        recalled = [fact for fact in expected_facts if fact.lower() in response_lower]
        recall_rate = len(recalled) / len(expected_facts) if expected_facts else 0

        result = {
            "id": case_id,
            "question": question,
            "expected_facts": expected_facts,
            "recalled_facts": recalled,
            "response": response[:400],
            "recall_rate": recall_rate,
            "passed": recall_rate >= 0.5  # Pass if at least half recalled
        }
        results.append(result)

        status = "PASS" if result["passed"] else "FAIL"
        print(f"    [{status}] Recalled {len(recalled)}/{len(expected_facts)} key facts")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)

    return {
        "category": "fact_recall",
        "passed": passed,
        "total": total,
        "accuracy": passed / total if total > 0 else 0,
        "details": results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate ecosystem-improved model")
    parser.add_argument("--model", type=str, default="./ecosystem_improved_model",
                        help="Path to the fine-tuned model")
    parser.add_argument("--base-model", type=str, default="mistralai/Mistral-7B-v0.3",
                        help="Base model name")
    parser.add_argument("--test-data", type=str, default="data/ecosystem_failures_training.json",
                        help="Path to test data JSON")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for results JSON")
    parser.add_argument("--no-4bit", action="store_true",
                        help="Disable 4-bit quantization")

    args = parser.parse_args()

    print("="*60)
    print("BioRLHF Ecosystem Model Evaluation")
    print("="*60)
    print(f"Model:      {args.model}")
    print(f"Base:       {args.base_model}")
    print(f"Test data:  {args.test_data}")
    print(f"Time:       {datetime.now().isoformat()}")
    print("="*60)

    # Load test data
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    # Load model
    model, tokenizer = load_model(args.model, args.base_model, use_4bit=not args.no_4bit)
    print("\nModel loaded successfully!\n")

    # Run evaluations
    results = {}

    # 1. Calibration
    if test_data.get("calibration_examples"):
        results["calibration"] = evaluate_calibration(
            model, tokenizer, test_data["calibration_examples"]
        )

    # 2. Adversarial resistance
    if test_data.get("adversarial_resistance_examples"):
        results["adversarial"] = evaluate_adversarial(
            model, tokenizer, test_data["adversarial_resistance_examples"]
        )

    # 3. Protocol completeness
    if test_data.get("completeness_examples"):
        results["completeness"] = evaluate_completeness(
            model, tokenizer, test_data["completeness_examples"]
        )

    # 4. Fact recall
    if test_data.get("fact_drilling_examples"):
        results["fact_recall"] = evaluate_fact_recall(
            model, tokenizer, test_data["fact_drilling_examples"]
        )

    # Summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)

    total_passed = 0
    total_tests = 0

    for category, data in results.items():
        print(f"\n{category.upper()}:")
        print(f"  Passed: {data['passed']}/{data['total']} ({data['accuracy']:.1%})")
        total_passed += data['passed']
        total_tests += data['total']

    overall_accuracy = total_passed / total_tests if total_tests > 0 else 0

    print("\n" + "-"*60)
    print(f"OVERALL: {total_passed}/{total_tests} ({overall_accuracy:.1%})")
    print("-"*60)

    # Save results
    output_path = args.output or f"ecosystem_eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    output_data = {
        "model_path": args.model,
        "base_model": args.base_model,
        "evaluation_date": datetime.now().isoformat(),
        "overall_accuracy": overall_accuracy,
        "total_passed": total_passed,
        "total_tests": total_tests,
        "results": results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print("\n" + "="*60)
    print("Evaluation complete!")
    print("="*60)


if __name__ == "__main__":
    main()
