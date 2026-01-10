"""
BioRLHF: Biological Reinforcement Learning from Human Feedback

A framework for fine-tuning LLMs on biological reasoning tasks with emphasis on
factual accuracy, chain-of-thought reasoning, and uncertainty calibration.
"""

__version__ = "0.1.0"
__author__ = "JangKeun Kim"
__email__ = "jangkeun.kim@med.cornell.edu"

from biorlhf.training.sft import SFTTrainingConfig, run_sft_training
from biorlhf.training.dpo import DPOTrainingConfig, run_dpo_training
from biorlhf.data.dataset import create_sft_dataset, load_dataset
from biorlhf.evaluation.evaluate import evaluate_model

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
