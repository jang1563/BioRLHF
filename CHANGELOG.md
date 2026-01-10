# Changelog

All notable changes to BioRLHF will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- GitHub Actions CI workflow for automated testing
- Pre-commit hooks configuration
- Unit tests for ground truth data and dataset creation
- Example scripts (quickstart, train_sft, evaluate_model)
- CONTRIBUTING.md guidelines
- CHANGELOG.md

### Changed
- Updated README with additional badges (CI status, Ruff, PRs welcome)

## [0.1.0] - 2025-01-09

### Added
- Initial release of BioRLHF framework
- SFT (Supervised Fine-Tuning) training pipeline
- DPO (Direct Preference Optimization) training pipeline
- Ground truth biological data from KMP 2x2x2 factorial study
- Automated SFT dataset generation with multiple example types:
  - Factual Q&A examples
  - Chain-of-thought reasoning examples
  - Uncertainty calibration examples
  - Interaction prediction examples
  - Experimental design critique examples
- Model evaluation with accuracy metrics:
  - Overall accuracy
  - Factual accuracy
  - Reasoning accuracy
  - Calibration accuracy
- Support for 4-bit quantization (QLoRA)
- LoRA adapter training
- Weights & Biases integration for experiment tracking
- HPC support with SLURM job scripts

### Training Results
- Achieved 90% overall accuracy on biological reasoning tasks
- 100% calibration accuracy (appropriate uncertainty expression)
- Successfully trained on 363 examples
- Model comparison study: Mistral-7B (90%) > Qwen2.5-7B (40%) > Phi-2 (25%)

### Data
- `kmp_sft_final.json`: 363 training examples
- `kmp_test_set.json`: 20-question evaluation set
- `kmp_dpo_preferences.json`: Preference pairs for DPO training

### Dependencies
- PyTorch >= 2.0.0
- Transformers >= 4.36.0
- TRL >= 0.7.0
- PEFT >= 0.6.0
- BitsAndBytes >= 0.41.0

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.1.0 | 2025-01-09 | Initial release with SFT/DPO pipelines |

[Unreleased]: https://github.com/jang1563/BioRLHF/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/jang1563/BioRLHF/releases/tag/v0.1.0
