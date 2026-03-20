#!/usr/bin/env python3
"""
Merge BioRLHF training data with ecosystem failure examples.

This script:
1. Loads existing kmp_sft_final.json training data
2. Loads ecosystem_failures_training.json (failure-based examples)
3. Converts failure examples to the same format
4. Outputs combined_training.json

Usage:
    python scripts/merge_training_data.py
"""

import json
from pathlib import Path
from datetime import datetime


def load_json(filepath: str) -> dict | list:
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def save_json(data: list, filepath: str):
    """Save JSON file."""
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"Saved {len(data)} examples to {filepath}")


def convert_conversation_to_text(conversation: list) -> str:
    """
    Convert conversation format to text format.

    Input: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    Output: "### Instruction:\n...\n\n### Response:\n..."
    """
    instruction = ""
    response = ""

    for turn in conversation:
        if turn["role"] == "user":
            instruction = turn["content"]
        elif turn["role"] == "assistant":
            response = turn["content"]

    return f"### Instruction:\n{instruction}\n\n### Response:\n{response}"


def extract_examples_from_failures(failure_data: dict) -> list:
    """
    Extract and convert all examples from failure training data.
    """
    examples = []

    # Process calibration examples
    for ex in failure_data.get("calibration_examples", []):
        text = convert_conversation_to_text(ex["conversations"])
        examples.append({
            "text": text,
            "source": f"ecosystem_failures:{ex['type']}",
            "id": ex["id"]
        })

    # Process adversarial resistance examples
    for ex in failure_data.get("adversarial_resistance_examples", []):
        text = convert_conversation_to_text(ex["conversations"])
        examples.append({
            "text": text,
            "source": f"ecosystem_failures:{ex['type']}",
            "id": ex["id"]
        })

    # Process completeness examples
    for ex in failure_data.get("completeness_examples", []):
        text = convert_conversation_to_text(ex["conversations"])
        examples.append({
            "text": text,
            "source": f"ecosystem_failures:{ex['type']}",
            "id": ex["id"]
        })

    # Process fact drilling examples
    for ex in failure_data.get("fact_drilling_examples", []):
        text = convert_conversation_to_text(ex["conversations"])
        examples.append({
            "text": text,
            "source": f"ecosystem_failures:{ex['type']}",
            "id": ex["id"]
        })

    return examples


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    existing_path = data_dir / "kmp_sft_final.json"
    failures_path = data_dir / "ecosystem_failures_training.json"
    output_path = data_dir / "combined_training.json"

    print("=" * 60)
    print("BioRLHF Training Data Merger")
    print("=" * 60)

    # Load existing data
    print(f"\n📂 Loading existing data from {existing_path}")
    existing_data = load_json(existing_path)
    print(f"   Found {len(existing_data)} existing examples")

    # Load failure-based examples
    print(f"\n📂 Loading failure examples from {failures_path}")
    failure_data = load_json(failures_path)

    # Convert failure examples
    print("\n🔄 Converting failure examples to training format...")
    new_examples = extract_examples_from_failures(failure_data)
    print(f"   Converted {len(new_examples)} examples")

    # Show breakdown
    print("\n📊 New examples by type:")
    type_counts = {}
    for ex in new_examples:
        source_type = ex["source"].split(":")[1] if ":" in ex["source"] else ex["source"]
        type_counts[source_type] = type_counts.get(source_type, 0) + 1
    for t, c in sorted(type_counts.items()):
        print(f"   - {t}: {c}")

    # Combine data
    print("\n🔀 Merging datasets...")

    # Add source field to existing data if not present
    for ex in existing_data:
        if "source" not in ex:
            ex["source"] = "kmp_sft_original"

    # Combine
    combined = existing_data + new_examples
    print(f"   Total examples: {len(combined)}")

    # Save combined data
    print(f"\n💾 Saving to {output_path}")
    save_json(combined, output_path)

    # Summary
    print("\n" + "=" * 60)
    print("✅ MERGE COMPLETE")
    print("=" * 60)
    print(f"   Original examples: {len(existing_data)}")
    print(f"   New examples:      {len(new_examples)}")
    print(f"   Total combined:    {len(combined)}")
    print(f"\n   Output: {output_path}")
    print("\nNext step: Run training with combined data:")
    print("   python sft_train_v2.py --dataset data/combined_training.json")


if __name__ == "__main__":
    main()
