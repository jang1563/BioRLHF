"""
Dataset creation and loading utilities for BioRLHF.

This module provides functions to create instruction-tuning datasets from
biological experimental data and load existing datasets.
"""

import json
from pathlib import Path
from typing import List, Dict, Optional, Union
from datasets import Dataset as HFDataset, load_dataset as hf_load_dataset

from biorlhf.data.ground_truth import (
    STRESSOR_EFFECTS,
    KMP_EFFECTS,
    INTERACTIONS,
    TISSUE_TYPES,
    OXPHOS_PATTERNS,
)


def load_dataset(
    path: Union[str, Path],
    split: Optional[str] = None,
    test_size: float = 0.1,
    seed: int = 42,
) -> Union[HFDataset, Dict[str, HFDataset]]:
    """
    Load a BioRLHF dataset from a JSON file.

    Args:
        path: Path to the JSON dataset file.
        split: If specified, return only this split ('train' or 'test').
        test_size: Fraction of data to use for testing.
        seed: Random seed for reproducible splits.

    Returns:
        HuggingFace Dataset or dict of train/test splits.
    """
    dataset = hf_load_dataset("json", data_files=str(path))["train"]

    if test_size > 0:
        splits = dataset.train_test_split(test_size=test_size, seed=seed)
        if split:
            return splits[split]
        return splits

    return dataset


def create_sft_dataset(
    output_path: Union[str, Path] = "kmp_sft_dataset.json",
    include_calibration: bool = True,
    include_chain_of_thought: bool = True,
) -> List[Dict[str, str]]:
    """
    Create an SFT dataset from ground truth biological data.

    Args:
        output_path: Path to save the generated dataset.
        include_calibration: Include uncertainty calibration examples.
        include_chain_of_thought: Include chain-of-thought reasoning examples.

    Returns:
        List of formatted training examples.
    """
    all_examples = []

    # Generate factual examples
    all_examples.extend(_generate_factual_examples())

    # Generate comparison examples
    all_examples.extend(_generate_comparison_examples())

    # Generate interaction prediction examples
    all_examples.extend(_generate_interaction_examples())

    # Generate design critique examples
    all_examples.extend(_generate_design_critique_examples())

    # Generate mechanistic reasoning examples
    if include_chain_of_thought:
        all_examples.extend(_generate_mechanistic_examples())

    # Generate calibration examples
    if include_calibration:
        all_examples.extend(_generate_calibration_examples())

    # Format for instruction tuning
    formatted = []
    for ex in all_examples:
        if ex.get("input"):
            text = (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Input:\n{ex['input']}\n\n"
                f"### Response:\n{ex['output']}"
            )
        else:
            text = (
                f"### Instruction:\n{ex['instruction']}\n\n"
                f"### Response:\n{ex['output']}"
            )
        formatted.append({"text": text})

    # Save to file
    with open(output_path, "w") as f:
        json.dump(formatted, f, indent=2)

    print(f"Created SFT dataset with {len(formatted)} examples at {output_path}")
    return formatted


def _generate_factual_examples() -> List[Dict]:
    """Generate factual Q&A examples about experimental results."""
    examples = []

    for tissue, effects in STRESSOR_EFFECTS.items():
        if tissue == "Soleus":
            hu_desc = "This reflects the severe muscle atrophy that occurs during simulated microgravity."
        else:
            hu_desc = f"This indicates {tissue.lower()} sensitivity to mechanical unloading."

        examples.append({
            "instruction": f"How many differentially expressed genes does hindlimb unloading cause in {tissue.lower()}?",
            "input": "",
            "output": f"Hindlimb unloading (HU) causes {effects['HU']:,} differentially expressed genes (padj < 0.05) in {tissue.lower()}. {hu_desc}"
        })

        if tissue == "Hippocampus":
            ir_desc = "This high number reflects the brain's particular sensitivity to radiation-induced damage."
        elif effects["IR"] < 100:
            ir_desc = "This minimal response indicates low radiation sensitivity in this tissue."
        else:
            ir_desc = "This moderate response indicates radiation sensitivity in this tissue."

        examples.append({
            "instruction": f"How many DEGs does ionizing radiation induce in {tissue.lower()}?",
            "input": "",
            "output": f"Ionizing radiation (IR) induces {effects['IR']:,} DEGs in {tissue.lower()}. {ir_desc}"
        })

    for tissue, effects in KMP_EFFECTS.items():
        pattern = "stress-activated" if effects["in_HU_IR"] > effects["baseline"] else "stress-blocked"
        examples.append({
            "instruction": f"How does kaempferol (KMP) affect gene expression in {tissue.lower()} under combined stress?",
            "input": "",
            "output": f"In {tissue.lower()}, KMP produces {effects['in_HU_IR']:,} DEGs under combined HU+IR stress, compared to {effects['baseline']:,} DEGs at baseline. This {pattern} pattern indicates KMP's pharmacodynamics are highly context-dependent."
        })

    for tissue, ttype in TISSUE_TYPES.items():
        examples.append({
            "instruction": f"What type of KMP response pattern does {tissue.lower()} show?",
            "input": "",
            "output": f"{tissue} is classified as {ttype}."
        })

    return examples


def _generate_comparison_examples() -> List[Dict]:
    """Generate tissue comparison examples."""
    examples = []

    examples.append({
        "instruction": "Which tissue is most sensitive to hindlimb unloading?",
        "input": "",
        "output": f"Soleus muscle is most sensitive to hindlimb unloading with {STRESSOR_EFFECTS['Soleus']['HU']:,} DEGs, followed by Liver ({STRESSOR_EFFECTS['Liver']['HU']:,}), Hippocampus ({STRESSOR_EFFECTS['Hippocampus']['HU']:,}), and Heart ({STRESSOR_EFFECTS['Heart']['HU']})."
    })

    examples.append({
        "instruction": "Which tissue is most sensitive to ionizing radiation?",
        "input": "",
        "output": f"Hippocampus is most sensitive to ionizing radiation with {STRESSOR_EFFECTS['Hippocampus']['IR']:,} DEGs, dramatically higher than other tissues."
    })

    examples.append({
        "instruction": "Compare KMP's transcriptional effects across all four tissues under combined stress.",
        "input": "",
        "output": f"KMP shows dramatically different effects across tissues under HU+IR stress: Heart ({KMP_EFFECTS['Heart']['in_HU_IR']:,} DEGs, stress-activated), Soleus ({KMP_EFFECTS['Soleus']['in_HU_IR']} DEGs, stress-activated), Hippocampus ({KMP_EFFECTS['Hippocampus']['in_HU_IR']} DEGs, stress-blocked), and Liver ({KMP_EFFECTS['Liver']['in_HU_IR']} DEGs, stress-blocked)."
    })

    return examples


def _generate_interaction_examples() -> List[Dict]:
    """Generate interaction prediction examples."""
    examples = []

    for tissue in ["Heart", "Hippocampus", "Liver", "Soleus"]:
        hu = STRESSOR_EFFECTS[tissue]["HU"]
        ir = STRESSOR_EFFECTS[tissue]["IR"]
        combined = STRESSOR_EFFECTS[tissue]["HU_IR"]

        examples.append({
            "instruction": f"Given the main effects of HU and IR separately in {tissue.lower()}, predict the combined HU+IR effect.",
            "input": f"{tissue} main effects: HU alone: {hu:,} DEGs, IR alone: {ir:,} DEGs. Predict the combined HU+IR effect.",
            "output": f"If additive, expect up to {hu + ir:,} DEGs. Actual result: {combined:,} DEGs. The combined effect reflects tissue-specific integration of stress responses."
        })

    return examples


def _generate_design_critique_examples() -> List[Dict]:
    """Generate experimental design critique examples."""
    examples = []

    examples.append({
        "instruction": "A researcher proposes testing KMP efficacy by comparing KMP-treated vs vehicle-treated mice under normal conditions. Critique this design.",
        "input": "Proposed: Compare KMP_vs_Ctrl under normal conditions.",
        "output": f"This design would miss KMP's primary mechanism. Heart: {KMP_EFFECTS['Heart']['baseline']} DEGs at baseline vs {KMP_EFFECTS['Heart']['in_HU_IR']:,} under stress. Soleus: {KMP_EFFECTS['Soleus']['baseline']} vs {KMP_EFFECTS['Soleus']['in_HU_IR']} DEGs. Recommendation: Use factorial design including stress conditions."
    })

    examples.append({
        "instruction": "Evaluate whether results from heart can predict liver response to KMP under stress.",
        "input": "Heart shows KMP rescues OXPHOS under stress. Can we extrapolate to liver?",
        "output": f"Direct extrapolation would be INCORRECT. Heart is Type A, Liver is Type C. OXPHOS patterns are opposite: Heart stress suppresses then KMP rescues; Liver stress activates then KMP suppresses. Each tissue requires independent testing."
    })

    return examples


def _generate_mechanistic_examples() -> List[Dict]:
    """Generate chain-of-thought mechanistic reasoning examples."""
    examples = []

    examples.append({
        "instruction": "Explain why hippocampus shows strong KMP effect at baseline (4,110 DEGs) but minimal effect under combined stress (140 DEGs).",
        "input": "",
        "output": "This 'baseline-active, stress-blocked' pattern suggests: (1) Receptor saturation - stress signals already engage KMP targets, (2) Pathway competition - IR-induced DNA damage response overrides KMP effects, (3) Altered cellular state - radiation changes transcriptional machinery. Implications: may need pre-treatment before stress."
    })

    examples.append({
        "instruction": f"The KMP x HU interaction in soleus produces {INTERACTIONS['Soleus']['KMP_x_HU']:,} DEGs. Explain this super-additive interaction.",
        "input": "",
        "output": "This massive interaction indicates: (1) Mechanosensitive activation - unloading creates binding sites for KMP-responsive factors, (2) Stress-primed responsiveness - HU upregulates KMP targets normally not expressed, (3) Mitochondrial crosstalk - KMP effects on mitochondria only manifest when organelles are stressed. Key insight: KMP specifically protects stressed tissue."
    })

    examples.append({
        "instruction": "Why does liver show OXPHOS suppression by KMP when heart and soleus show OXPHOS rescue?",
        "input": "",
        "output": "Different metabolic contexts: Heart/Soleus (RESCUE) - stress suppresses OXPHOS, insufficient ATP, KMP restores function. Liver (SUPPRESSION) - stress activates compensatory metabolism, over-activation causes damage, KMP provides 'hepatoprotective braking'. 'Rescue' is tissue-specific: restore suppressed function vs prevent harmful over-activation."
    })

    return examples


def _generate_calibration_examples() -> List[Dict]:
    """Generate uncertainty calibration examples."""
    examples = []

    examples.append({
        "instruction": "What is the effect of KMP on cardiac contractility?",
        "input": "",
        "output": "I cannot directly answer this. The data shows transcriptomic changes (2,110 DEGs, OXPHOS rescue) but NOT functional contractility. OXPHOS rescue suggests improved ATP availability, which might improve contractility, but transcription-to-function is not guaranteed. Confidence: low for functional claims."
    })

    examples.append({
        "instruction": "Will KMP prevent muscle atrophy in human astronauts?",
        "input": "",
        "output": "I cannot confidently answer this. Mouse data supports: stress-activated protection, OXPHOS rescue, KMP x HU interaction. Mouse data cannot tell us: human PK, dose requirements, true microgravity effects, long-duration effects. Strong mechanistic rationale but insufficient for human predictions without additional studies."
    })

    examples.append({
        "instruction": "Which gene is most important for KMP's protective effect?",
        "input": "",
        "output": "Cannot be definitively answered from transcriptomic data. Problems: correlation is not causation, pathway redundancy, no functional validation, tissue-specificity. Better questions: 'Which genes show largest effects?' (answerable), 'Which pathways are enriched?' (answerable), 'Which gene is necessary?' (requires CRISPR studies)."
    })

    examples.append({
        "instruction": "Based on this data, what dose of KMP should be used for spaceflight countermeasures?",
        "input": "",
        "output": "This data CANNOT inform dosing. Missing: dose-response relationship (only one dose tested), human PK, therapeutic window, bioavailability differences. Mouse-to-human scaling is unreliable (10-100x errors possible). Needed: mouse dose-response, PK modeling, human Phase I studies. Do not extrapolate dosing from this study."
    })

    return examples
