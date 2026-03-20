"""
BioRLHF: Biological Reinforcement Learning from Human Feedback

A framework for fine-tuning LLMs on biological reasoning tasks with emphasis on
factual accuracy, chain-of-thought reasoning, and uncertainty calibration.
"""

__version__ = "0.1.0"
__author__ = "JangKeun Kim"
__email__ = "jangkeun.kim@med.cornell.edu"

def __getattr__(name):
    """Lazy imports for torch-dependent modules."""
    if name == "SFTTrainingConfig":
        from biorlhf.training.sft import SFTTrainingConfig
        return SFTTrainingConfig
    elif name == "run_sft_training":
        from biorlhf.training.sft import run_sft_training
        return run_sft_training
    elif name == "DPOTrainingConfig":
        from biorlhf.training.dpo import DPOTrainingConfig
        return DPOTrainingConfig
    elif name == "run_dpo_training":
        from biorlhf.training.dpo import run_dpo_training
        return run_dpo_training
    elif name == "create_sft_dataset":
        from biorlhf.data.dataset import create_sft_dataset
        return create_sft_dataset
    elif name == "load_dataset":
        from biorlhf.data.dataset import load_dataset
        return load_dataset
    elif name == "evaluate_model":
        from biorlhf.evaluation.evaluate import evaluate_model
        return evaluate_model
    raise AttributeError(f"module 'biorlhf' has no attribute {name!r}")

__all__ = [
    "__version__",
    "SFTTrainingConfig",
    "run_sft_training",
    "DPOTrainingConfig",
    "run_dpo_training",
    "create_sft_dataset",
    "load_dataset",
    "evaluate_model",
]
