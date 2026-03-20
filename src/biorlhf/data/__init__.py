"""Data processing and dataset creation modules for BioRLHF."""

# ground_truth has no heavy dependencies, safe to import eagerly
from biorlhf.data.ground_truth import (
    STRESSOR_EFFECTS,
    KMP_EFFECTS,
    INTERACTIONS,
    TISSUE_TYPES,
    OXPHOS_PATTERNS,
)

__all__ = [
    "create_sft_dataset",
    "load_dataset",
    "STRESSOR_EFFECTS",
    "KMP_EFFECTS",
    "INTERACTIONS",
    "TISSUE_TYPES",
    "OXPHOS_PATTERNS",
]


def __getattr__(name):
    """Lazy imports for modules with heavy dependencies."""
    if name in ("create_sft_dataset", "load_dataset"):
        from biorlhf.data.dataset import create_sft_dataset, load_dataset
        return {"create_sft_dataset": create_sft_dataset, "load_dataset": load_dataset}[name]
    raise AttributeError(f"module 'biorlhf.data' has no attribute {name!r}")
