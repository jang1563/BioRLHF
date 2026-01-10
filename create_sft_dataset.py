#!/usr/bin/env python3
"""
BioRLHF SFT Dataset Generator
Creates instruction-tuning dataset from KMP 2x2x2 factorial mouse data

Usage:
    python create_sft_dataset.py --output kmp_sft_dataset.json
"""

import json
import argparse
from typing import List, Dict

# =============================================================================
# GROUND TRUTH DATA (from KMP_Analysis_Tables.xlsx)
# =============================================================================

STRESSOR_EFFECTS = {
    'Heart': {'HU': 165, 'IR': 33, 'HU_IR': 910},
    'Hippocampus': {'HU': 1555, 'IR': 5477, 'HU_IR': 5510},
    'Liver': {'HU': 4110, 'IR': 1273, 'HU_IR': 6213},
    'Soleus': {'HU': 6425, 'IR': 67, 'HU_IR': 6830},
}

KMP_EFFECTS = {
    'Heart': {'baseline': 112, 'in_HU': 2, 'in_IR': 2, 'in_HU_IR': 2110},
    'Hippocampus': {'baseline': 4110, 'in_HU': 1, 'in_IR': 243, 'in_HU_IR': 140},
    'Liver': {'baseline': 309, 'in_HU': 17, 'in_IR': 389, 'in_HU_IR': 3},
    'Soleus': {'baseline': 0, 'in_HU': 1, 'in_IR': 52, 'in_HU_IR': 491},
}

INTERACTIONS = {
    'Heart': {'HU_x_IR': 244, 'KMP_x_HU': 479, 'KMP_x_IR': 29},
    'Hippocampus': {'HU_x_IR': 93, 'KMP_x_HU': 36, 'KMP_x_IR': 1221},
    'Liver': {'HU_x_IR': 3210, 'KMP_x_HU': 3369, 'KMP_x_IR': 247},
    'Soleus': {'HU_x_IR': 211, 'KMP_x_HU': 8484, 'KMP_x_IR': 484},
}

TISSUE_TYPES = {
    'Heart': 'Type A (stress-activated)',
    'Soleus': 'Type A (stress-activated)', 
    'Hippocampus': 'Type B (baseline-active)',
    'Liver': 'Type C (stress-blocked)',
}

OXPHOS_PATTERNS = {
    'Heart': {'stress_NES': -2.302, 'KMP_NES': 3.691, 'pattern': 'RESCUE'},
    'Hippocampus': {'stress_NES': 0.931, 'KMP_NES': 1.585, 'pattern': 'KMP Only'},
    'Liver': {'stress_NES': 3.596, 'KMP_NES': -1.6, 'pattern': 'SUPPRESSION'},
    'Soleus': {'stress_NES': -2.997, 'KMP_NES': 2.46, 'pattern': 'RESCUE'},
}


def generate_factual_examples() -> List[Dict]:
    examples = []
    
    for tissue, effects in STRESSOR_EFFECTS.items():
        if tissue == 'Soleus':
            hu_desc = "This reflects the severe muscle atrophy that occurs during simulated microgravity."
        else:
            hu_desc = f"This indicates {tissue.lower()} sensitivity to mechanical unloading."
        
        examples.append({
            "instruction": f"How many differentially expressed genes does hindlimb unloading cause in {tissue.lower()}?",
            "input": "",
            "output": f"Hindlimb unloading (HU) causes {effects['HU']:,} differentially expressed genes (padj < 0.05) in {tissue.lower()}. {hu_desc}"
        })
        
        if tissue == 'Hippocampus':
            ir_desc = "This high number reflects the brain's particular sensitivity to radiation-induced damage."
        elif effects['IR'] < 100:
            ir_desc = "This minimal response indicates low radiation sensitivity in this tissue."
        else:
            ir_desc = "This moderate response indicates radiation sensitivity in this tissue."
        
        examples.append({
            "instruction": f"How many DEGs does ionizing radiation induce in {tissue.lower()}?",
            "input": "",
            "output": f"Ionizing radiation (IR) induces {effects['IR']:,} DEGs in {tissue.lower()}. {ir_desc}"
        })
        
        examples.append({
            "instruction": f"What is the combined effect of HU and IR stress on {tissue.lower()} transcriptome?",
            "input": "",
            "output": f"Combined HU+IR stress produces {effects['HU_IR']:,} DEGs in {tissue.lower()}."
        })
    
    for tissue, effects in KMP_EFFECTS.items():
        pattern = "stress-activated" if effects['in_HU_IR'] > effects['baseline'] else "stress-blocked"
        examples.append({
            "instruction": f"How does kaempferol (KMP) affect gene expression in {tissue.lower()} under combined stress?",
            "input": "",
            "output": f"In {tissue.lower()}, KMP produces {effects['in_HU_IR']:,} DEGs under combined HU+IR stress, compared to {effects['baseline']:,} DEGs at baseline. This {pattern} pattern indicates KMP's pharmacodynamics are highly context-dependent."
        })
    
    for tissue, ints in INTERACTIONS.items():
        examples.append({
            "instruction": f"What is the KMP x HU interaction effect in {tissue.lower()}?",
            "input": "",
            "output": f"The KMP x HU interaction in {tissue.lower()} produces {ints['KMP_x_HU']:,} DEGs, indicating non-additive pharmacodynamics."
        })
    
    for tissue, pattern in OXPHOS_PATTERNS.items():
        examples.append({
            "instruction": f"What is the OXPHOS pathway pattern in {tissue.lower()} under stress and KMP treatment?",
            "input": "",
            "output": f"In {tissue.lower()}, combined stress produces OXPHOS NES = {pattern['stress_NES']:.2f}, while KMP treatment shifts this to NES = {pattern['KMP_NES']:.2f}. This represents a {pattern['pattern']} pattern."
        })
    
    for tissue, ttype in TISSUE_TYPES.items():
        examples.append({
            "instruction": f"What type of KMP response pattern does {tissue.lower()} show?",
            "input": "",
            "output": f"{tissue} is classified as {ttype}."
        })
    
    return examples


def generate_comparison_examples() -> List[Dict]:
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
    
    examples.append({
        "instruction": "Which tissue shows the largest KMP x HU interaction effect?",
        "input": "",
        "output": f"Soleus shows the largest KMP x HU interaction with {INTERACTIONS['Soleus']['KMP_x_HU']:,} DEGs, the largest interaction in the entire dataset."
    })
    
    examples.append({
        "instruction": "Classify tissues by their dominant stressor sensitivity.",
        "input": "",
        "output": f"HU-dominant: Heart ({STRESSOR_EFFECTS['Heart']['HU']} vs {STRESSOR_EFFECTS['Heart']['IR']}), Soleus ({STRESSOR_EFFECTS['Soleus']['HU']:,} vs {STRESSOR_EFFECTS['Soleus']['IR']}). IR-dominant: Hippocampus ({STRESSOR_EFFECTS['Hippocampus']['HU']:,} vs {STRESSOR_EFFECTS['Hippocampus']['IR']:,}). Both: Liver."
    })
    
    tissue_pairs = [('Heart', 'Soleus'), ('Heart', 'Liver'), ('Hippocampus', 'Liver')]
    for t1, t2 in tissue_pairs:
        examples.append({
            "instruction": f"Compare KMP context-dependency between {t1.lower()} and {t2.lower()}.",
            "input": "",
            "output": f"{t1} ({TISSUE_TYPES[t1]}): baseline {KMP_EFFECTS[t1]['baseline']} DEGs, stressed {KMP_EFFECTS[t1]['in_HU_IR']:,} DEGs. {t2} ({TISSUE_TYPES[t2]}): baseline {KMP_EFFECTS[t2]['baseline']} DEGs, stressed {KMP_EFFECTS[t2]['in_HU_IR']} DEGs."
        })
    
    return examples


def generate_interaction_examples() -> List[Dict]:
    examples = []
    
    for tissue in ['Heart', 'Hippocampus', 'Liver', 'Soleus']:
        hu = STRESSOR_EFFECTS[tissue]['HU']
        ir = STRESSOR_EFFECTS[tissue]['IR']
        combined = STRESSOR_EFFECTS[tissue]['HU_IR']
        
        examples.append({
            "instruction": f"Given the main effects of HU and IR separately in {tissue.lower()}, predict the combined HU+IR effect.",
            "input": f"{tissue} main effects: HU alone: {hu:,} DEGs, IR alone: {ir:,} DEGs. Predict the combined HU+IR effect.",
            "output": f"If additive, expect up to {hu + ir:,} DEGs. Actual result: {combined:,} DEGs. The combined effect reflects tissue-specific integration of stress responses."
        })
    
    for tissue in ['Heart', 'Soleus', 'Liver', 'Hippocampus']:
        baseline = KMP_EFFECTS[tissue]['baseline']
        stressed = KMP_EFFECTS[tissue]['in_HU_IR']
        ttype = TISSUE_TYPES[tissue]
        
        examples.append({
            "instruction": f"KMP shows {baseline} DEGs at baseline in {tissue.lower()}. Predict KMP effect under combined HU+IR stress.",
            "input": f"KMP at baseline in {tissue.lower()}: {baseline} DEGs. {tissue} stress response (HU+IR): {STRESSOR_EFFECTS[tissue]['HU_IR']:,} DEGs.",
            "output": f"Actual result: {stressed:,} DEGs ({ttype}). {'Stress activates KMP response.' if stressed > baseline else 'Stress blocks KMP response.'}"
        })
    
    examples.append({
        "instruction": "The KMP x HU interaction in heart produces 479 DEGs. Predict the magnitude in soleus.",
        "input": "Heart KMP x HU: 479 DEGs. Both are striated muscle. Soleus has larger HU response.",
        "output": f"Prediction: Larger than heart. Actual: {INTERACTIONS['Soleus']['KMP_x_HU']:,} DEGs, the largest interaction in the dataset."
    })
    
    examples.append({
        "instruction": "Given OXPHOS RESCUE in heart, predict liver OXPHOS response to KMP.",
        "input": f"Heart: Stress suppresses OXPHOS (NES={OXPHOS_PATTERNS['Heart']['stress_NES']}), KMP rescues (NES={OXPHOS_PATTERNS['Heart']['KMP_NES']}). Liver is Type C.",
        "output": f"Prediction: Opposite pattern. Actual: Liver stress NES={OXPHOS_PATTERNS['Liver']['stress_NES']:.2f}, KMP NES={OXPHOS_PATTERNS['Liver']['KMP_NES']:.2f}. SUPPRESSION pattern (hepatoprotective braking)."
    })
    
    return examples


def generate_design_critique_examples() -> List[Dict]:
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
    
    examples.append({
        "instruction": "A study finds KMP reduces oxidative stress markers in cell culture. How confident can we be this translates to in vivo protection?",
        "input": "",
        "output": "Confidence: LOW to MODERATE. Cell culture shows mechanism plausibility but cannot capture context-dependency (in vivo effects range 0-2,110 DEGs), tissue specificity (opposite OXPHOS patterns), or interaction effects (KMP x HU = 8,484 DEGs in soleus)."
    })
    
    examples.append({
        "instruction": "Critique a study using padj < 0.05 AND |log2FC| > 1 threshold for DEG detection.",
        "input": "Study reports 'no effect' of KMP in heart under stress using stringent thresholds.",
        "output": f"Overly stringent thresholds miss meaningful effects. With padj < 0.05 only: KMP produces {KMP_EFFECTS['Heart']['in_HU_IR']:,} DEGs in heart. Many coordinated small changes drive pathway-level effects. Use padj-only for discovery, complement with pathway analysis."
    })
    
    examples.append({
        "instruction": "A grant proposal plans to test KMP in astronauts based solely on the mouse data. Evaluate the translational gap.",
        "input": "",
        "output": "SUBSTANTIAL gaps remain. Mouse data provides: proof of concept, tissue patterns, OXPHOS mechanism. Missing: human PK, true microgravity effects (HU is analog), long-duration effects, safety profile. Needed: human organoid studies, dose-finding, analog studies before human trials."
    })
    
    return examples


def generate_mechanistic_examples() -> List[Dict]:
    examples = []
    
    examples.append({
        "instruction": "Explain why hippocampus shows strong KMP effect at baseline (4,110 DEGs) but minimal effect under combined stress (140 DEGs).",
        "input": "",
        "output": "This 'baseline-active, stress-blocked' pattern suggests: (1) Receptor saturation - stress signals already engage KMP targets, (2) Pathway competition - IR-induced DNA damage response overrides KMP effects, (3) Altered cellular state - radiation changes transcriptional machinery. Implications: may need pre-treatment before stress."
    })
    
    examples.append({
        "instruction": "The KMP x HU interaction in soleus produces 8,484 DEGs. Explain this super-additive interaction.",
        "input": "",
        "output": "This massive interaction indicates: (1) Mechanosensitive activation - unloading creates binding sites for KMP-responsive factors, (2) Stress-primed responsiveness - HU upregulates KMP targets normally not expressed, (3) Mitochondrial crosstalk - KMP effects on mitochondria only manifest when organelles are stressed. Key insight: KMP specifically protects stressed tissue."
    })
    
    examples.append({
        "instruction": "Why does liver show OXPHOS suppression by KMP when heart and soleus show OXPHOS rescue?",
        "input": "",
        "output": "Different metabolic contexts: Heart/Soleus (RESCUE) - stress suppresses OXPHOS, insufficient ATP, KMP restores function. Liver (SUPPRESSION) - stress activates compensatory metabolism, over-activation causes damage, KMP provides 'hepatoprotective braking'. 'Rescue' is tissue-specific: restore suppressed function vs prevent harmful over-activation."
    })
    
    examples.append({
        "instruction": "Explain the tissue-type classification system (Type A/B/C).",
        "input": "",
        "output": f"Type A (stress-activated): Heart, Soleus - minimal baseline effect, strong under stress. Type B (baseline-active): Hippocampus - strong baseline ({KMP_EFFECTS['Hippocampus']['baseline']:,} DEGs), minimal under stress ({KMP_EFFECTS['Hippocampus']['in_HU_IR']}). Type C (stress-blocked): Liver - moderate baseline ({KMP_EFFECTS['Liver']['baseline']}), nearly zero under stress ({KMP_EFFECTS['Liver']['in_HU_IR']}). Clinical implications: tissue-specific dosing, timing, and endpoints needed."
    })
    
    return examples


def generate_calibration_examples() -> List[Dict]:
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
    
    examples.append({
        "instruction": "Is the KMP x HU interaction (8,484 DEGs) statistically significant?",
        "input": "",
        "output": "Yes, with context. Each gene has padj < 0.05 for the interaction term. However: (1) exact number is threshold-dependent, (2) effect sizes vary across genes later. Appropriate statement: 'approximately 8,500 significantly changed genes (padj < 0.05), indicating substantial non-additive pharmacological effect.'"
    })
    
    return examples


def compile_sft_dataset(output_file: str = 'kmp_sft_dataset.json'):
    all_examples = []
    
    print("Generating factual examples...")
    all_examples.extend(generate_factual_examples())
    
    print("Generating comparison examples...")
    all_examples.extend(generate_comparison_examples())
    
    print("Generating interaction prediction examples...")
    all_examples.extend(generate_interaction_examples())
    
    print("Generating design critique examples...")
    all_examples.extend(generate_design_critique_examples())
    
    print("Generating mechanistic reasoning examples...")
    all_examples.extend(generate_mechanistic_examples())
    
    print("Generating calibration examples...")
    all_examples.extend(generate_calibration_examples())
    
    formatted = []
    for ex in all_examples:
        if ex.get('input'):
            text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        else:
            text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
        formatted.append({"text": text})
    
    with open(output_file, 'w') as f:
        json.dump(formatted, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"SFT Dataset Summary")
    print(f"{'='*60}")
    print(f"Total examples: {len(formatted)}")
    print(f"Output file: {output_file}")
    
    return formatted


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='kmp_sft_dataset.json')
    args = parser.parse_args()
    compile_sft_dataset(args.output)
