"""Tests for the main BioRLHF package."""

import pytest


class TestPackageImports:
    """Test that package exports are available."""

    def test_version_available(self):
        """Test that version is accessible."""
        from biorlhf import __version__

        assert isinstance(__version__, str)
        assert __version__ == "0.1.0"

    def test_author_metadata(self):
        """Test that author metadata is available."""
        from biorlhf import __author__, __email__

        assert isinstance(__author__, str)
        assert isinstance(__email__, str)

    def test_sft_training_imports(self):
        """Test that SFT training components are importable."""
        from biorlhf import SFTTrainingConfig, run_sft_training

        assert SFTTrainingConfig is not None
        assert callable(run_sft_training)

    def test_dpo_training_imports(self):
        """Test that DPO training components are importable."""
        from biorlhf import DPOTrainingConfig, run_dpo_training

        assert DPOTrainingConfig is not None
        assert callable(run_dpo_training)

    def test_dataset_imports(self):
        """Test that dataset functions are importable."""
        from biorlhf import create_sft_dataset, load_dataset

        assert callable(create_sft_dataset)
        assert callable(load_dataset)

    def test_evaluation_imports(self):
        """Test that evaluation functions are importable."""
        from biorlhf import evaluate_model

        assert callable(evaluate_model)

    def test_all_exports(self):
        """Test that __all__ contains expected exports."""
        import biorlhf

        expected_exports = [
            "__version__",
            "SFTTrainingConfig",
            "run_sft_training",
            "DPOTrainingConfig",
            "run_dpo_training",
            "create_sft_dataset",
            "load_dataset",
            "evaluate_model",
        ]

        for export in expected_exports:
            assert export in biorlhf.__all__, f"{export} missing from __all__"


class TestSubmoduleImports:
    """Test that submodules are properly organized."""

    def test_training_submodule(self):
        """Test training submodule structure."""
        from biorlhf.training import SFTTrainingConfig, DPOTrainingConfig

        assert SFTTrainingConfig is not None
        assert DPOTrainingConfig is not None

    def test_data_submodule(self):
        """Test data submodule structure."""
        from biorlhf.data import ground_truth, dataset

        assert hasattr(ground_truth, "STRESSOR_EFFECTS")
        assert hasattr(dataset, "create_sft_dataset")

    def test_evaluation_submodule(self):
        """Test evaluation submodule structure."""
        from biorlhf.evaluation import evaluate

        assert hasattr(evaluate, "evaluate_model")

    def test_utils_submodule(self):
        """Test utils submodule structure."""
        from biorlhf.utils import model_utils

        assert model_utils is not None
