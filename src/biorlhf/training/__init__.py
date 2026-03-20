"""Training modules for BioRLHF."""

__all__ = [
    "SFTTrainingConfig",
    "run_sft_training",
    "DPOTrainingConfig",
    "run_dpo_training",
    "BioGRPOConfig",
    "run_grpo_training",
]


def __getattr__(name):
    """Lazy imports for torch-dependent modules."""
    if name in ("SFTTrainingConfig", "run_sft_training"):
        from biorlhf.training.sft import SFTTrainingConfig, run_sft_training
        return {"SFTTrainingConfig": SFTTrainingConfig, "run_sft_training": run_sft_training}[name]
    elif name in ("DPOTrainingConfig", "run_dpo_training"):
        from biorlhf.training.dpo import DPOTrainingConfig, run_dpo_training
        return {"DPOTrainingConfig": DPOTrainingConfig, "run_dpo_training": run_dpo_training}[name]
    elif name in ("BioGRPOConfig", "run_grpo_training"):
        from biorlhf.training.grpo import BioGRPOConfig, run_grpo_training
        return {"BioGRPOConfig": BioGRPOConfig, "run_grpo_training": run_grpo_training}[name]
    raise AttributeError(f"module 'biorlhf.training' has no attribute {name!r}")
