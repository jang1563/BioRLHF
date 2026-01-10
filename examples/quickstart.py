#!/usr/bin/env python3
"""
BioRLHF Quickstart Example

This script demonstrates the basic workflow for using BioRLHF:
1. Loading ground truth biological data
2. Creating an SFT dataset
3. Exploring the generated examples

Note: This example does not require a GPU and is safe to run locally.
"""

import json
import tempfile
from pathlib import Path

# Import ground truth data
from biorlhf.data.ground_truth import (
    STRESSOR_EFFECTS,
    KMP_EFFECTS,
    TISSUE_TYPES,
    OXPHOS_PATTERNS,
)

# Import dataset creation utilities
from biorlhf.data.dataset import create_sft_dataset


def explore_ground_truth():
    """Explore the ground truth experimental data."""
    print("=" * 60)
    print("BioRLHF Ground Truth Data Explorer")
    print("=" * 60)

    print("\n1. STRESSOR EFFECTS (DEG counts by tissue)")
    print("-" * 40)
    for tissue, effects in STRESSOR_EFFECTS.items():
        print(f"\n{tissue}:")
        print(f"  Hindlimb Unloading (HU): {effects['HU']:,} DEGs")
        print(f"  Ionizing Radiation (IR): {effects['IR']:,} DEGs")
        print(f"  Combined HU+IR: {effects['HU_IR']:,} DEGs")

    print("\n\n2. KMP EFFECTS UNDER DIFFERENT CONDITIONS")
    print("-" * 40)
    for tissue, effects in KMP_EFFECTS.items():
        print(f"\n{tissue}:")
        print(f"  Baseline: {effects['baseline']:,} DEGs")
        print(f"  Under HU: {effects['in_HU']:,} DEGs")
        print(f"  Under IR: {effects['in_IR']:,} DEGs")
        print(f"  Under HU+IR: {effects['in_HU_IR']:,} DEGs")

    print("\n\n3. TISSUE CLASSIFICATIONS")
    print("-" * 40)
    for tissue, ttype in TISSUE_TYPES.items():
        print(f"  {tissue}: {ttype}")

    print("\n\n4. OXPHOS PATHWAY PATTERNS")
    print("-" * 40)
    for tissue, data in OXPHOS_PATTERNS.items():
        print(f"\n{tissue}:")
        print(f"  Stress NES: {data['stress_NES']}")
        print(f"  KMP NES: {data['KMP_NES']}")
        print(f"  Pattern: {data['pattern']}")


def create_example_dataset():
    """Create and explore an example SFT dataset."""
    print("\n\n" + "=" * 60)
    print("Creating Example SFT Dataset")
    print("=" * 60)

    # Create a temporary directory for the output
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "example_dataset.json"

        # Create the dataset
        examples = create_sft_dataset(
            output_path=output_path,
            include_calibration=True,
            include_chain_of_thought=True,
        )

        print(f"\nDataset created with {len(examples)} examples")
        print(f"Saved to: {output_path}")

        # Show example categories
        print("\n\nSample Examples by Category:")
        print("-" * 40)

        # Show a few examples
        for i, ex in enumerate(examples[:3]):
            print(f"\n--- Example {i+1} ---")
            text = ex["text"]
            # Truncate long outputs for display
            if len(text) > 500:
                text = text[:500] + "..."
            print(text)


def main():
    """Run the quickstart demonstration."""
    print("\n" + "=" * 60)
    print("Welcome to BioRLHF!")
    print("=" * 60)
    print("""
This quickstart demonstrates the BioRLHF framework for fine-tuning
LLMs on biological reasoning tasks.

Key features:
- Ground truth data from KMP 2x2x2 factorial transcriptomic study
- Automated SFT dataset generation
- Support for factual, reasoning, and calibration examples
""")

    # Run demonstrations
    explore_ground_truth()
    create_example_dataset()

    print("\n\n" + "=" * 60)
    print("Next Steps")
    print("=" * 60)
    print("""
To train a model, see the full training examples:
- examples/train_sft.py - Supervised fine-tuning
- examples/evaluate_model.py - Model evaluation

For GPU training, ensure you have:
- CUDA-compatible GPU
- torch with CUDA support
- Sufficient VRAM (16GB+ recommended)
""")


if __name__ == "__main__":
    main()
