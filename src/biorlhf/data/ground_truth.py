"""
Ground truth data from KMP 2x2x2 factorial transcriptomic study.

This module contains the experimental data from the Kaempferol (KMP) countermeasure
study examining:
- 4 tissues: Heart, Hippocampus, Liver, Soleus
- 2 stressors: Hindlimb Unloading (HU), Ionizing Radiation (IR)
- 1 intervention: Kaempferol (KMP)
"""

from typing import Dict, Any

# DEG counts for stressor effects (padj < 0.05)
STRESSOR_EFFECTS: Dict[str, Dict[str, int]] = {
    "Heart": {"HU": 165, "IR": 33, "HU_IR": 910},
    "Hippocampus": {"HU": 1555, "IR": 5477, "HU_IR": 5510},
    "Liver": {"HU": 4110, "IR": 1273, "HU_IR": 6213},
    "Soleus": {"HU": 6425, "IR": 67, "HU_IR": 6830},
}

# KMP effects under different conditions
KMP_EFFECTS: Dict[str, Dict[str, int]] = {
    "Heart": {"baseline": 112, "in_HU": 2, "in_IR": 2, "in_HU_IR": 2110},
    "Hippocampus": {"baseline": 4110, "in_HU": 1, "in_IR": 243, "in_HU_IR": 140},
    "Liver": {"baseline": 309, "in_HU": 17, "in_IR": 389, "in_HU_IR": 3},
    "Soleus": {"baseline": 0, "in_HU": 1, "in_IR": 52, "in_HU_IR": 491},
}

# Interaction effects (non-additive pharmacodynamics)
INTERACTIONS: Dict[str, Dict[str, int]] = {
    "Heart": {"HU_x_IR": 244, "KMP_x_HU": 479, "KMP_x_IR": 29},
    "Hippocampus": {"HU_x_IR": 93, "KMP_x_HU": 36, "KMP_x_IR": 1221},
    "Liver": {"HU_x_IR": 3210, "KMP_x_HU": 3369, "KMP_x_IR": 247},
    "Soleus": {"HU_x_IR": 211, "KMP_x_HU": 8484, "KMP_x_IR": 484},
}

# Tissue classification by KMP response pattern
TISSUE_TYPES: Dict[str, str] = {
    "Heart": "Type A (stress-activated)",
    "Soleus": "Type A (stress-activated)",
    "Hippocampus": "Type B (baseline-active)",
    "Liver": "Type C (stress-blocked)",
}

# OXPHOS pathway patterns
OXPHOS_PATTERNS: Dict[str, Dict[str, Any]] = {
    "Heart": {"stress_NES": -2.302, "KMP_NES": 3.691, "pattern": "RESCUE"},
    "Hippocampus": {"stress_NES": 0.931, "KMP_NES": 1.585, "pattern": "KMP Only"},
    "Liver": {"stress_NES": 3.596, "KMP_NES": -1.6, "pattern": "SUPPRESSION"},
    "Soleus": {"stress_NES": -2.997, "KMP_NES": 2.46, "pattern": "RESCUE"},
}
