"""Tests for dataset creation and loading module."""

import json
import tempfile
from pathlib import Path

import pytest


class TestDatasetCreation:
    """Tests for dataset creation functions."""

    def test_generate_factual_examples_import(self):
        """Test that _generate_factual_examples can be imported and called."""
        from biorlhf.data.dataset import _generate_factual_examples

        examples = _generate_factual_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0

    def test_factual_examples_structure(self):
        """Test that factual examples have required fields."""
        from biorlhf.data.dataset import _generate_factual_examples

        examples = _generate_factual_examples()
        for ex in examples:
            assert "instruction" in ex
            assert "output" in ex
            # Input can be empty string but must exist
            assert "input" in ex

    def test_generate_comparison_examples(self):
        """Test comparison example generation."""
        from biorlhf.data.dataset import _generate_comparison_examples

        examples = _generate_comparison_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0

        # Check for specific comparison questions
        instructions = [ex["instruction"] for ex in examples]
        assert any("most sensitive" in instr.lower() for instr in instructions)

    def test_generate_interaction_examples(self):
        """Test interaction prediction example generation."""
        from biorlhf.data.dataset import _generate_interaction_examples

        examples = _generate_interaction_examples()
        assert isinstance(examples, list)
        # Should have one example per tissue
        assert len(examples) == 4

    def test_generate_design_critique_examples(self):
        """Test experimental design critique example generation."""
        from biorlhf.data.dataset import _generate_design_critique_examples

        examples = _generate_design_critique_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0

    def test_generate_mechanistic_examples(self):
        """Test mechanistic reasoning example generation."""
        from biorlhf.data.dataset import _generate_mechanistic_examples

        examples = _generate_mechanistic_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0

    def test_generate_calibration_examples(self):
        """Test uncertainty calibration example generation."""
        from biorlhf.data.dataset import _generate_calibration_examples

        examples = _generate_calibration_examples()
        assert isinstance(examples, list)
        assert len(examples) > 0

        # Calibration examples should express uncertainty
        for ex in examples:
            output = ex["output"].lower()
            uncertainty_markers = ["cannot", "insufficient", "confidence", "needed", "missing"]
            has_uncertainty = any(marker in output for marker in uncertainty_markers)
            assert has_uncertainty, f"Calibration example should express uncertainty: {ex['output'][:100]}"


class TestCreateSFTDataset:
    """Tests for the main create_sft_dataset function."""

    def test_creates_dataset_file(self):
        """Test that create_sft_dataset creates a JSON file."""
        from biorlhf.data.dataset import create_sft_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dataset.json"
            result = create_sft_dataset(output_path=output_path)

            assert output_path.exists()
            assert isinstance(result, list)
            assert len(result) > 0

    def test_dataset_format(self):
        """Test that created dataset has correct format."""
        from biorlhf.data.dataset import create_sft_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dataset.json"
            result = create_sft_dataset(output_path=output_path)

            # Each example should have "text" field
            for ex in result:
                assert "text" in ex
                text = ex["text"]
                # Should have instruction format
                assert "### Instruction:" in text
                assert "### Response:" in text

    def test_dataset_json_valid(self):
        """Test that output file is valid JSON."""
        from biorlhf.data.dataset import create_sft_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dataset.json"
            create_sft_dataset(output_path=output_path)

            with open(output_path) as f:
                data = json.load(f)

            assert isinstance(data, list)

    def test_exclude_calibration(self):
        """Test that calibration examples can be excluded."""
        from biorlhf.data.dataset import create_sft_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            path_with = Path(tmpdir) / "with_cal.json"
            path_without = Path(tmpdir) / "without_cal.json"

            result_with = create_sft_dataset(output_path=path_with, include_calibration=True)
            result_without = create_sft_dataset(output_path=path_without, include_calibration=False)

            # Dataset with calibration should be larger
            assert len(result_with) > len(result_without)

    def test_exclude_chain_of_thought(self):
        """Test that chain-of-thought examples can be excluded."""
        from biorlhf.data.dataset import create_sft_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            path_with = Path(tmpdir) / "with_cot.json"
            path_without = Path(tmpdir) / "without_cot.json"

            result_with = create_sft_dataset(output_path=path_with, include_chain_of_thought=True)
            result_without = create_sft_dataset(output_path=path_without, include_chain_of_thought=False)

            # Dataset with CoT should be larger
            assert len(result_with) > len(result_without)


class TestLoadDataset:
    """Tests for the load_dataset function."""

    def test_load_dataset_basic(self):
        """Test basic dataset loading."""
        from biorlhf.data.dataset import create_sft_dataset, load_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dataset.json"
            create_sft_dataset(output_path=output_path)

            # Load the dataset
            dataset = load_dataset(output_path, test_size=0)

            assert hasattr(dataset, "__len__")
            assert len(dataset) > 0

    def test_load_dataset_with_split(self):
        """Test dataset loading with train/test split."""
        from biorlhf.data.dataset import create_sft_dataset, load_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dataset.json"
            create_sft_dataset(output_path=output_path)

            # Load with split
            splits = load_dataset(output_path, test_size=0.2)

            assert "train" in splits
            assert "test" in splits
            assert len(splits["train"]) > len(splits["test"])

    def test_load_specific_split(self):
        """Test loading a specific split."""
        from biorlhf.data.dataset import create_sft_dataset, load_dataset

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_dataset.json"
            create_sft_dataset(output_path=output_path)

            # Load only train split
            train_dataset = load_dataset(output_path, split="train", test_size=0.2)

            # Should not be a dict, should be a Dataset
            assert not isinstance(train_dataset, dict)
            assert hasattr(train_dataset, "__len__")
