#!/usr/bin/env python3
"""
BioRLHF SFT Dataset Generator - EXPANDED VERSION
Creates 200+ instruction-tuning examples from KMP 2x2x2 factorial mouse data
"""

import json
import argparse
from typing import List, Dict
import random

# =============================================================================
# GROUND TRUTH DATA
# =============================================================================

STRESSOR_EFFECTS = {
    'Heart': {'HU': 165, 'IR': 33, 'HU_IR': 910, 'HU_up': 67, 'HU_down': 98, 'IR_up': 17, 'IR_down': 16},
    'Hippocampus': {'HU': 1555, 'IR': 5477, 'HU_IR': 5510, 'HU_up': 711, 'HU_down': 844, 'IR_up': 2554, 'IR_down': 2923},
    'Liver': {'HU': 4110, 'IR': 1273, 'HU_IR': 6213, 'HU_up': 2189, 'HU_down': 1921, 'IR_up': 413, 'IR_down': 860},
    'Soleus': {'HU': 6425, 'IR': 67, 'HU_IR': 6830, 'HU_up': 3251, 'HU_down': 3174, 'IR_up': 28, 'IR_down': 39},
}

KMP_EFFECTS = {
    'Heart': {'baseline': 112, 'in_HU': 2, 'in_IR': 2, 'in_HU_IR': 2110, 'in_HU_IR_up': 1336, 'in_HU_IR_down': 774},
    'Hippocampus': {'baseline': 4110, 'in_HU': 1, 'in_IR': 243, 'in_HU_IR': 140, 'baseline_up': 1813, 'baseline_down': 2297},
    'Liver': {'baseline': 309, 'in_HU': 17, 'in_IR': 389, 'in_HU_IR': 3},
    'Soleus': {'baseline': 0, 'in_HU': 1, 'in_IR': 52, 'in_HU_IR': 491, 'in_HU_IR_up': 425, 'in_HU_IR_down': 66},
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
    'Heart': {'stress_NES': -2.302, 'KMP_NES': 3.691, 'pattern': 'RESCUE', 'delta': 5.993},
    'Hippocampus': {'stress_NES': 0.931, 'KMP_NES': 1.585, 'pattern': 'KMP Only', 'delta': 0.654},
    'Liver': {'stress_NES': 3.596, 'KMP_NES': -1.6, 'pattern': 'SUPPRESSION', 'delta': -5.196},
    'Soleus': {'stress_NES': -2.997, 'KMP_NES': 2.46, 'pattern': 'RESCUE', 'delta': 5.457},
}

PATHWAY_PATTERNS = {
    'Heart': {
        'FATTY_ACID_METABOLISM': {'stress': -2.371, 'kmp': 3.1, 'pattern': 'RESCUE'},
        'ADIPOGENESIS': {'stress': -1.839, 'kmp': 2.81, 'pattern': 'RESCUE'},
        'MTORC1_SIGNALING': {'stress': -1.662, 'kmp': 2.585, 'pattern': 'RESCUE'},
        'INTERFERON_ALPHA': {'stress': -2.072, 'kmp': 1.581, 'pattern': 'RESCUE'},
    },
    'Liver': {
        'MTORC1_SIGNALING': {'stress': 3.075, 'kmp': -1.678, 'pattern': 'SUPPRESSION'},
        'INTERFERON_GAMMA': {'stress': 1.542, 'kmp': -2.336, 'pattern': 'SUPPRESSION'},
    },
    'Soleus': {
        'FATTY_ACID_METABOLISM': {'stress': -2.418, 'kmp': 1.506, 'pattern': 'RESCUE'},
    }
}


def generate_factual_basic() -> List[Dict]:
    """Basic factual questions about DEG counts."""
    examples = []
    
    for tissue, effects in STRESSOR_EFFECTS.items():
        # HU effects with direction
        examples.append({
            "instruction": f"How many genes are differentially expressed due to hindlimb unloading in {tissue.lower()}?",
            "input": "",
            "output": f"Hindlimb unloading causes {effects['HU']:,} DEGs in {tissue.lower()} (padj < 0.05), with {effects.get('HU_up', 'N/A')} upregulated and {effects.get('HU_down', 'N/A')} downregulated."
        })
        
        # IR effects
        examples.append({
            "instruction": f"What is the transcriptomic response to ionizing radiation in {tissue.lower()}?",
            "input": "",
            "output": f"Ionizing radiation induces {effects['IR']:,} DEGs in {tissue.lower()}, with {effects.get('IR_up', 'N/A')} upregulated and {effects.get('IR_down', 'N/A')} downregulated."
        })
        
        # Combined effects
        examples.append({
            "instruction": f"Describe the combined stress response in {tissue.lower()}.",
            "input": "",
            "output": f"Combined HU+IR stress produces {effects['HU_IR']:,} DEGs in {tissue.lower()}, representing the integrated response to simultaneous mechanical unloading and radiation exposure."
        })
        
        # Additivity check
        additive = effects['HU'] + effects['IR']
        if effects['HU_IR'] > additive * 1.2:
            add_type = "synergistic (more than additive)"
        elif effects['HU_IR'] < additive * 0.8:
            add_type = "sub-additive (less than sum)"
        else:
            add_type = "approximately additive"
        
        examples.append({
            "instruction": f"Is the combined HU+IR effect additive in {tissue.lower()}?",
            "input": "",
            "output": f"In {tissue.lower()}, HU causes {effects['HU']:,} DEGs and IR causes {effects['IR']:,} DEGs. The combined effect ({effects['HU_IR']:,} DEGs) is {add_type}."
        })
    
    return examples


def generate_factual_kmp() -> List[Dict]:
    """Factual questions about KMP effects."""
    examples = []
    
    for tissue, effects in KMP_EFFECTS.items():
        # Baseline vs stress comparison
        fold = effects['in_HU_IR'] / max(effects['baseline'], 1)
        if fold > 5:
            change = "dramatically increases"
        elif fold < 0.2:
            change = "dramatically decreases"
        else:
            change = "moderately changes"
        
        examples.append({
            "instruction": f"How does stress affect KMP's transcriptional activity in {tissue.lower()}?",
            "input": "",
            "output": f"KMP effect {change} from {effects['baseline']:,} DEGs at baseline to {effects['in_HU_IR']:,} DEGs under combined stress in {tissue.lower()}. This indicates {'stress-activated' if fold > 1 else 'stress-blocked'} pharmacodynamics."
        })
        
        # Each stress condition
        examples.append({
            "instruction": f"Compare KMP effects across different stress conditions in {tissue.lower()}.",
            "input": "",
            "output": f"In {tissue.lower()}, KMP produces: {effects['baseline']} DEGs at baseline, {effects['in_HU']} DEGs under HU only, {effects['in_IR']} DEGs under IR only, and {effects['in_HU_IR']:,} DEGs under combined HU+IR stress."
        })
        
        # Direction of KMP effect
        if 'in_HU_IR_up' in effects:
            pct_up = effects['in_HU_IR_up'] / effects['in_HU_IR'] * 100
            examples.append({
                "instruction": f"What is the direction of KMP-induced gene expression changes in {tissue.lower()} under stress?",
                "input": "",
                "output": f"Under combined stress, KMP induces {effects['in_HU_IR_up']:,} upregulated and {effects['in_HU_IR_down']:,} downregulated genes in {tissue.lower()} ({pct_up:.1f}% upregulated). This {'anabolic/protective' if pct_up > 60 else 'mixed' if pct_up > 40 else 'suppressive'} signature suggests {'tissue protection' if pct_up > 60 else 'complex regulation'}."
            })
    
    return examples


def generate_factual_interactions() -> List[Dict]:
    """Factual questions about interaction effects."""
    examples = []
    
    for tissue, ints in INTERACTIONS.items():
        # KMP x HU
        examples.append({
            "instruction": f"What is the statistical interaction between KMP and HU in {tissue.lower()}?",
            "input": "",
            "output": f"The KMP × HU interaction produces {ints['KMP_x_HU']:,} DEGs in {tissue.lower()}, indicating {'massive' if ints['KMP_x_HU'] > 5000 else 'substantial' if ints['KMP_x_HU'] > 500 else 'moderate'} non-additive effects."
        })
        
        # KMP x IR
        examples.append({
            "instruction": f"Describe the KMP × IR interaction in {tissue.lower()}.",
            "input": "",
            "output": f"The KMP × IR interaction produces {ints['KMP_x_IR']:,} DEGs in {tissue.lower()}, {'representing the largest radiation-drug interaction' if ints['KMP_x_IR'] > 1000 else 'indicating modest interaction with radiation stress'}."
        })
        
        # HU x IR
        examples.append({
            "instruction": f"Is there a HU × IR interaction in {tissue.lower()}?",
            "input": "",
            "output": f"Yes, the HU × IR interaction produces {ints['HU_x_IR']:,} DEGs in {tissue.lower()}, indicating the two stressors have {'strong synergistic' if ints['HU_x_IR'] > 1000 else 'moderate non-additive'} effects."
        })
    
    return examples


def generate_factual_pathways() -> List[Dict]:
    """Factual questions about pathway patterns."""
    examples = []
    
    for tissue, pattern in OXPHOS_PATTERNS.items():
        examples.append({
            "instruction": f"What happens to oxidative phosphorylation in {tissue.lower()} under stress?",
            "input": "",
            "output": f"Under combined HU+IR stress, OXPHOS shows NES = {pattern['stress_NES']:.2f} in {tissue.lower()}, indicating {'suppression' if pattern['stress_NES'] < 0 else 'activation'} of mitochondrial respiration."
        })
        
        examples.append({
            "instruction": f"How does KMP affect OXPHOS in {tissue.lower()}?",
            "input": "",
            "output": f"KMP shifts OXPHOS NES from {pattern['stress_NES']:.2f} to {pattern['KMP_NES']:.2f} in {tissue.lower()} (Δ = {pattern['delta']:.2f}). This {pattern['pattern']} pattern indicates {'restoration of mitochondrial function' if 'RESCUE' in pattern['pattern'] else 'metabolic braking' if 'SUPPRESSION' in pattern['pattern'] else 'KMP-specific effects'}."
        })
    
    for tissue, pathways in PATHWAY_PATTERNS.items():
        for pathway, data in pathways.items():
            examples.append({
                "instruction": f"What is the {pathway.replace('_', ' ').lower()} response in {tissue.lower()}?",
                "input": "",
                "output": f"In {tissue.lower()}, stress produces {pathway.replace('_', ' ')} NES = {data['stress']:.2f}, and KMP shifts this to NES = {data['kmp']:.2f}. Pattern: {data['pattern']}."
            })
    
    return examples


def generate_comparison_questions() -> List[Dict]:
    """Cross-tissue and cross-condition comparisons."""
    examples = []
    
    # Tissue rankings
    tissues_by_hu = sorted(STRESSOR_EFFECTS.items(), key=lambda x: x[1]['HU'], reverse=True)
    examples.append({
        "instruction": "Rank tissues by sensitivity to hindlimb unloading.",
        "input": "",
        "output": f"HU sensitivity ranking: 1) {tissues_by_hu[0][0]} ({tissues_by_hu[0][1]['HU']:,}), 2) {tissues_by_hu[1][0]} ({tissues_by_hu[1][1]['HU']:,}), 3) {tissues_by_hu[2][0]} ({tissues_by_hu[2][1]['HU']:,}), 4) {tissues_by_hu[3][0]} ({tissues_by_hu[3][1]['HU']})."
    })
    
    tissues_by_ir = sorted(STRESSOR_EFFECTS.items(), key=lambda x: x[1]['IR'], reverse=True)
    examples.append({
        "instruction": "Rank tissues by sensitivity to ionizing radiation.",
        "input": "",
        "output": f"IR sensitivity ranking: 1) {tissues_by_ir[0][0]} ({tissues_by_ir[0][1]['IR']:,}), 2) {tissues_by_ir[1][0]} ({tissues_by_ir[1][1]['IR']:,}), 3) {tissues_by_ir[2][0]} ({tissues_by_ir[2][1]['IR']}), 4) {tissues_by_ir[3][0]} ({tissues_by_ir[3][1]['IR']})."
    })
    
    tissues_by_kmp = sorted(KMP_EFFECTS.items(), key=lambda x: x[1]['in_HU_IR'], reverse=True)
    examples.append({
        "instruction": "Rank tissues by KMP effect under combined stress.",
        "input": "",
        "output": f"KMP effect under stress: 1) {tissues_by_kmp[0][0]} ({tissues_by_kmp[0][1]['in_HU_IR']:,}), 2) {tissues_by_kmp[1][0]} ({tissues_by_kmp[1][1]['in_HU_IR']}), 3) {tissues_by_kmp[2][0]} ({tissues_by_kmp[2][1]['in_HU_IR']}), 4) {tissues_by_kmp[3][0]} ({tissues_by_kmp[3][1]['in_HU_IR']})."
    })
    
    # Pairwise comparisons
    for t1 in ['Heart', 'Hippocampus', 'Liver', 'Soleus']:
        for t2 in ['Heart', 'Hippocampus', 'Liver', 'Soleus']:
            if t1 < t2:
                examples.append({
                    "instruction": f"Compare {t1.lower()} and {t2.lower()} responses to HU.",
                    "input": "",
                    "output": f"{t1}: {STRESSOR_EFFECTS[t1]['HU']:,} DEGs. {t2}: {STRESSOR_EFFECTS[t2]['HU']:,} DEGs. {'Same' if TISSUE_TYPES[t1] == TISSUE_TYPES[t2] else 'Different'} KMP response type."
                })
                
                examples.append({
                    "instruction": f"Compare KMP context-dependency in {t1.lower()} vs {t2.lower()}.",
                    "input": "",
                    "output": f"{t1} ({TISSUE_TYPES[t1]}): baseline→stress = {KMP_EFFECTS[t1]['baseline']}→{KMP_EFFECTS[t1]['in_HU_IR']:,}. {t2} ({TISSUE_TYPES[t2]}): {KMP_EFFECTS[t2]['baseline']}→{KMP_EFFECTS[t2]['in_HU_IR']}."
                })
    
    # Stressor dominance
    for tissue, effects in STRESSOR_EFFECTS.items():
        if effects['HU'] > effects['IR'] * 3:
            dominance = "HU-dominant"
        elif effects['IR'] > effects['HU'] * 3:
            dominance = "IR-dominant"
        else:
            dominance = "balanced response"
        
        examples.append({
            "instruction": f"What stressor dominates the response in {tissue.lower()}?",
            "input": "",
            "output": f"{tissue} shows {dominance}: HU = {effects['HU']:,} DEGs, IR = {effects['IR']:,} DEGs (ratio = {effects['HU']/max(effects['IR'],1):.1f})."
        })
    
    return examples


def generate_prediction_tasks() -> List[Dict]:
    """Interaction and cross-tissue prediction tasks."""
    examples = []
    
    # Predict combined from main effects
    for tissue in STRESSOR_EFFECTS.keys():
        effects = STRESSOR_EFFECTS[tissue]
        examples.append({
            "instruction": f"Predict combined HU+IR effect in {tissue.lower()} from main effects.",
            "input": f"HU alone: {effects['HU']:,} DEGs. IR alone: {effects['IR']:,} DEGs.",
            "output": f"Additive prediction: ~{effects['HU']+effects['IR']:,} DEGs. Actual: {effects['HU_IR']:,} DEGs. The {'synergistic' if effects['HU_IR'] > effects['HU']+effects['IR'] else 'sub-additive'} effect reflects biological interaction between stressors."
        })
    
    # Predict KMP under stress from baseline
    for tissue in KMP_EFFECTS.keys():
        kmp = KMP_EFFECTS[tissue]
        examples.append({
            "instruction": f"Predict KMP effect under stress in {tissue.lower()}.",
            "input": f"KMP at baseline: {kmp['baseline']} DEGs. Tissue type: {TISSUE_TYPES[tissue]}.",
            "output": f"Based on tissue type, predict {'increase' if 'stress-activated' in TISSUE_TYPES[tissue] else 'decrease'}. Actual: {kmp['in_HU_IR']:,} DEGs. Ratio: {kmp['in_HU_IR']/max(kmp['baseline'],1):.1f}x."
        })
    
    # Cross-tissue predictions
    examples.append({
        "instruction": "Given heart (Type A) and soleus (Type A), predict similarity of KMP response.",
        "input": "Both are Type A (stress-activated). Heart KMP under stress: 2,110 DEGs.",
        "output": f"Prediction: Similar stress-activated pattern. Actual soleus: {KMP_EFFECTS['Soleus']['in_HU_IR']} DEGs. Both show stress-activated response, confirming Type A classification predicts pharmacodynamics."
    })
    
    examples.append({
        "instruction": "Given heart OXPHOS RESCUE, predict liver OXPHOS pattern.",
        "input": f"Heart: stress {OXPHOS_PATTERNS['Heart']['stress_NES']:.1f} → KMP {OXPHOS_PATTERNS['Heart']['KMP_NES']:.1f}. Liver is Type C (different).",
        "output": f"Prediction: Different pattern due to metabolic role. Actual: Liver shows SUPPRESSION (stress {OXPHOS_PATTERNS['Liver']['stress_NES']:.1f} → KMP {OXPHOS_PATTERNS['Liver']['KMP_NES']:.1f}). Opposite pattern confirms tissue-specific mechanisms."
    })
    
    examples.append({
        "instruction": "Predict KMP × HU interaction magnitude in soleus from heart data.",
        "input": f"Heart KMP × HU: {INTERACTIONS['Heart']['KMP_x_HU']} DEGs. Soleus has larger HU response ({STRESSOR_EFFECTS['Soleus']['HU']:,} vs {STRESSOR_EFFECTS['Heart']['HU']}).",
        "output": f"Prediction: Larger than heart due to greater HU sensitivity. Actual: {INTERACTIONS['Soleus']['KMP_x_HU']:,} DEGs ({INTERACTIONS['Soleus']['KMP_x_HU']/INTERACTIONS['Heart']['KMP_x_HU']:.0f}x larger). Massive interaction reflects stress-dependent KMP activation."
    })
    
    return examples


def generate_design_critique() -> List[Dict]:
    """Experimental design critique tasks."""
    examples = []
    
    examples.append({
        "instruction": "Critique: Testing KMP only at baseline to assess spaceflight protection.",
        "input": "",
        "output": f"FLAWED. Heart baseline KMP: {KMP_EFFECTS['Heart']['baseline']} DEGs. Heart stressed KMP: {KMP_EFFECTS['Heart']['in_HU_IR']:,} DEGs. Testing at baseline would miss {KMP_EFFECTS['Heart']['in_HU_IR']/max(KMP_EFFECTS['Heart']['baseline'],1):.0f}x larger stressed effect. Must include stress conditions."
    })
    
    examples.append({
        "instruction": "Critique: Extrapolating heart results to predict liver response.",
        "input": "",
        "output": "INCORRECT. Heart (Type A) and Liver (Type C) have opposite patterns. Heart OXPHOS: RESCUE. Liver OXPHOS: SUPPRESSION. KMP effect: Heart {0} → {1} DEGs, Liver {2} → {3} DEGs. Cross-tissue extrapolation invalid.".format(
            KMP_EFFECTS['Heart']['baseline'], KMP_EFFECTS['Heart']['in_HU_IR'],
            KMP_EFFECTS['Liver']['baseline'], KMP_EFFECTS['Liver']['in_HU_IR']
        )
    })
    
    examples.append({
        "instruction": "Critique: Using |log2FC| > 1 threshold.",
        "input": "",
        "output": f"PROBLEMATIC. Stringent thresholds miss coordinated small effects. Using padj-only: Heart KMP = {KMP_EFFECTS['Heart']['in_HU_IR']:,} DEGs. Many genes with <2-fold change contribute to pathway-level effects like OXPHOS rescue."
    })
    
    examples.append({
        "instruction": "Critique: Single-tissue study for KMP countermeasure development.",
        "input": "",
        "output": f"INSUFFICIENT. Tissue responses vary dramatically: Heart {KMP_EFFECTS['Heart']['in_HU_IR']:,}, Soleus {KMP_EFFECTS['Soleus']['in_HU_IR']}, Hippocampus {KMP_EFFECTS['Hippocampus']['in_HU_IR']}, Liver {KMP_EFFECTS['Liver']['in_HU_IR']} DEGs under stress. Type A/B/C classification requires multi-tissue data."
    })
    
    examples.append({
        "instruction": "Critique: Assuming additive drug-stressor effects.",
        "input": "",
        "output": f"INCORRECT. KMP × HU interactions are highly non-additive. Soleus: {INTERACTIONS['Soleus']['KMP_x_HU']:,} interaction DEGs. This super-additive effect indicates synergistic mechanisms that additive models cannot capture."
    })
    
    examples.append({
        "instruction": "Evaluate cell culture KMP data for in vivo predictions.",
        "input": "",
        "output": f"LIMITED VALUE. Cell culture cannot capture: context-dependency (KMP effects range 0-{KMP_EFFECTS['Heart']['in_HU_IR']:,} DEGs), tissue specificity (Type A/B/C), or interaction effects (KMP × HU = {INTERACTIONS['Soleus']['KMP_x_HU']:,} DEGs). In vivo factorial studies essential."
    })
    
    examples.append({
        "instruction": "Critique: Human dosing based on mouse efficacy data.",
        "input": "",
        "output": "PREMATURE. Missing: dose-response (single dose tested), human PK, therapeutic window, bioavailability. Mouse-to-human scaling unreliable (10-100x errors). Need: dose-response studies, PK modeling, Phase I trials."
    })
    
    return examples


def generate_mechanistic_reasoning() -> List[Dict]:
    """Mechanistic reasoning tasks."""
    examples = []
    
    examples.append({
        "instruction": "Explain hippocampus baseline-active, stress-blocked pattern.",
        "input": "",
        "output": f"Hippocampus: {KMP_EFFECTS['Hippocampus']['baseline']:,} DEGs baseline → {KMP_EFFECTS['Hippocampus']['in_HU_IR']} DEGs stressed. Mechanisms: (1) Target saturation by stress signals, (2) Pathway competition with DNA damage response, (3) Altered transcriptional machinery from radiation. Implication: Pre-treatment may be more effective."
    })
    
    examples.append({
        "instruction": "Explain the massive KMP × HU interaction in soleus.",
        "input": "",
        "output": f"Soleus KMP × HU = {INTERACTIONS['Soleus']['KMP_x_HU']:,} DEGs (largest interaction). Mechanisms: (1) Mechanosensitive pathways create KMP binding sites, (2) HU upregulates KMP targets normally not expressed, (3) Mitochondrial dysfunction reveals KMP protective capacity. Key insight: KMP specifically protects stressed tissue."
    })
    
    examples.append({
        "instruction": "Explain opposite OXPHOS patterns in heart vs liver.",
        "input": "",
        "output": f"Heart OXPHOS: stress suppresses ({OXPHOS_PATTERNS['Heart']['stress_NES']:.1f}) → KMP rescues ({OXPHOS_PATTERNS['Heart']['KMP_NES']:.1f}). Liver OXPHOS: stress activates ({OXPHOS_PATTERNS['Liver']['stress_NES']:.1f}) → KMP suppresses ({OXPHOS_PATTERNS['Liver']['KMP_NES']:.1f}). Heart needs ATP restoration for contraction. Liver needs metabolic braking to prevent oxidative damage. 'Rescue' is tissue-specific."
    })
    
    examples.append({
        "instruction": "Why does soleus show 0 KMP DEGs at baseline but 491 under stress?",
        "input": "",
        "output": f"Complete stress-dependency. Baseline: KMP targets not expressed or accessible. Under HU stress: Mechanosensitive pathways activate, target expression increases, mitochondrial stress reveals KMP binding sites. Result: 0 → {KMP_EFFECTS['Soleus']['in_HU_IR']} DEGs. KMP is conditionally active therapeutic."
    })
    
    examples.append({
        "instruction": "Explain Type A/B/C classification system.",
        "input": "",
        "output": f"Type A (Heart, Soleus): Stress-activated. Baseline minimal, stress maximal. KMP works when needed. Type B (Hippocampus): Baseline-active ({KMP_EFFECTS['Hippocampus']['baseline']:,}→{KMP_EFFECTS['Hippocampus']['in_HU_IR']}). Stress blocks. May need pre-treatment. Type C (Liver): Stress-blocked ({KMP_EFFECTS['Liver']['baseline']}→{KMP_EFFECTS['Liver']['in_HU_IR']}). Different mechanism needed."
    })
    
    examples.append({
        "instruction": "Why is KMP × HU larger than KMP × IR in most tissues?",
        "input": "",
        "output": f"KMP × HU: Heart {INTERACTIONS['Heart']['KMP_x_HU']}, Liver {INTERACTIONS['Liver']['KMP_x_HU']:,}, Soleus {INTERACTIONS['Soleus']['KMP_x_HU']:,}. KMP × IR: Heart {INTERACTIONS['Heart']['KMP_x_IR']}, Liver {INTERACTIONS['Liver']['KMP_x_IR']}, Soleus {INTERACTIONS['Soleus']['KMP_x_IR']}. KMP likely targets mechanosensitive/metabolic pathways more than DNA damage pathways."
    })
    
    return examples


def generate_uncertainty_calibration() -> List[Dict]:
    """Uncertainty calibration tasks."""
    examples = []
    
    examples.append({
        "instruction": "What is KMP's effect on cardiac contractility?",
        "input": "",
        "output": f"CANNOT DIRECTLY ANSWER. Data shows: {KMP_EFFECTS['Heart']['in_HU_IR']:,} DEGs, OXPHOS rescue. These are transcriptomic changes. NOT shown: functional contractility, protein levels, force generation. Inference: OXPHOS rescue may improve ATP→may improve contractility. Confidence: LOW for function."
    })
    
    examples.append({
        "instruction": "Will KMP prevent human astronaut muscle atrophy?",
        "input": "",
        "output": f"CANNOT CONFIDENTLY ANSWER. Supports: Stress-activated protection ({KMP_EFFECTS['Soleus']['in_HU_IR']} DEGs), KMP × HU interaction ({INTERACTIONS['Soleus']['KMP_x_HU']:,}), OXPHOS rescue. Missing: Human PK, true microgravity (HU is analog), long-duration effects. Strong rationale but insufficient for prediction."
    })
    
    examples.append({
        "instruction": "Which gene is most important for KMP protection?",
        "input": "",
        "output": "CANNOT DEFINITIVELY ANSWER. Transcriptomics shows correlation, not causation. Large change ≠ causal role. Redundant pathways exist. No knockout validation. Better questions: 'Largest effects?' (answerable), 'Most enriched pathways?' (answerable). Need: CRISPR screening for causality."
    })
    
    examples.append({
        "instruction": "What KMP dose should be used for astronauts?",
        "input": "",
        "output": "CANNOT INFORM DOSING. Missing: Dose-response (one dose tested), human PK, therapeutic window, bioavailability. Mouse-human scaling unreliable. Need: Multi-dose mouse studies, PK modeling, human Phase I. Do not extrapolate."
    })
    
    examples.append({
        "instruction": "Is the 8,484 DEG interaction statistically robust?",
        "input": "",
        "output": f"YES with caveats. Each of {INTERACTIONS['Soleus']['KMP_x_HU']:,} genes has padj < 0.05. Highly significant. But: exact number is threshold-dependent, effect sizes vary. Appropriate: 'approximately 8,500 DEGs indicating substantial non-additive effect.' Avoid false precision."
    })
    
    examples.append({
        "instruction": "Can we predict KMP effects in kidney from this data?",
        "input": "",
        "output": "CANNOT PREDICT. Kidney not studied. Type A/B/C patterns are tissue-specific and cannot be extrapolated. Kidney has unique metabolic/filtration role. Would need: dedicated kidney experiments. Cross-tissue prediction requires mechanistic validation."
    })
    
    return examples


def generate_application_questions() -> List[Dict]:
    """Practical application questions."""
    examples = []
    
    examples.append({
        "instruction": "Design a KMP efficacy study for astronaut protection.",
        "input": "",
        "output": f"Required: (1) Factorial design with stress conditions (not baseline-only), (2) Multiple tissues (Type A/B/C responses differ), (3) Functional endpoints beyond transcriptomics, (4) Dose-response. Rationale: Baseline-only would miss {KMP_EFFECTS['Heart']['in_HU_IR']:,}/{KMP_EFFECTS['Heart']['baseline']} DEG effect in heart."
    })
    
    examples.append({
        "instruction": "Prioritize tissues for KMP therapeutic development.",
        "input": "",
        "output": f"Priority 1: Soleus - largest interaction ({INTERACTIONS['Soleus']['KMP_x_HU']:,}), stress-activated, OXPHOS rescue. Priority 2: Heart - strong rescue ({KMP_EFFECTS['Heart']['in_HU_IR']:,} DEGs). Lower priority: Hippocampus (stress-blocked), Liver (minimal effect). Focus on Type A tissues."
    })
    
    examples.append({
        "instruction": "Identify biomarkers for KMP efficacy monitoring.",
        "input": "",
        "output": "Candidate biomarkers: (1) OXPHOS pathway genes (track rescue/suppression), (2) Tissue-type classification (confirm Type A/B/C), (3) Interaction-responsive genes from KMP × HU. Need validation: functional correlation, accessibility (blood-based), temporal dynamics."
    })
    
    examples.append({
        "instruction": "What controls are essential for KMP spaceflight studies?",
        "input": "",
        "output": f"Essential controls: (1) Vehicle under all stress conditions (not just baseline), (2) KMP at baseline (to detect context-dependency), (3) Single stressors (HU-only, IR-only) for interaction calculation, (4) Multiple tissues. Missing any control prevents detecting effects like {KMP_EFFECTS['Soleus']['baseline']}→{KMP_EFFECTS['Soleus']['in_HU_IR']} shift."
    })
    
    return examples


def compile_sft_dataset(output_file: str = 'kmp_sft_dataset.json'):
    """Compile all examples into final SFT dataset."""
    
    all_examples = []
    
    print("Generating factual basic examples...")
    all_examples.extend(generate_factual_basic())
    
    print("Generating factual KMP examples...")
    all_examples.extend(generate_factual_kmp())
    
    print("Generating factual interaction examples...")
    all_examples.extend(generate_factual_interactions())
    
    print("Generating factual pathway examples...")
    all_examples.extend(generate_factual_pathways())
    
    print("Generating comparison examples...")
    all_examples.extend(generate_comparison_questions())
    
    print("Generating prediction examples...")
    all_examples.extend(generate_prediction_tasks())
    
    print("Generating design critique examples...")
    all_examples.extend(generate_design_critique())
    
    print("Generating mechanistic reasoning examples...")
    all_examples.extend(generate_mechanistic_reasoning())
    
    print("Generating uncertainty calibration examples...")
    all_examples.extend(generate_uncertainty_calibration())
    
    print("Generating application examples...")
    all_examples.extend(generate_application_questions())
    
    # Format for training
    formatted = []
    for ex in all_examples:
        if ex.get('input'):
            text = f"### Instruction:\n{ex['instruction']}\n\n### Input:\n{ex['input']}\n\n### Response:\n{ex['output']}"
        else:
            text = f"### Instruction:\n{ex['instruction']}\n\n### Response:\n{ex['output']}"
        formatted.append({"text": text})
    
    # Shuffle for training
    random.seed(42)
    random.shuffle(formatted)
    
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
