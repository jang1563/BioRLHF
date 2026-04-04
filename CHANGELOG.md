# Changelog

All notable changes to BioRLHF will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2026-04-04

### Added
- Phase 4 GRPO ablation experiments (legacy vs match_v1 V4 modes)
- V4 verifier V1-aware calibration mode (`_score_calibration_aware`)
- Environment variable `BIORLHF_V4_DEFAULT_MODE` for V4 scoring mode
- Ablation training configs for Phase 4
- HuggingFace model uploads: SFT and GRPO adapters

### Changed
- V4 verifier weight increased to 0.45 (dominant) for calibration pressure
- Updated README with multi-phase GRPO results and HF model links
- Sanitized HPC paths in all scripts — now use env vars (`BIORLHF_SCRATCH`, `BIORLHF_CONDA_SH`)
- Added `generate_response` to public API exports
- Fixed `source ~/.bashrc` in setup script (doesn't work non-interactively)

### Training Results
- **Legacy ablation**: Reward 0.566, ECE 0.183 (best calibration with full verifiers)
- **MatchV1 ablation**: Reward 0.523, ECE 0.242
- Legacy mode outperforms match_v1 on all metrics

## [0.2.0] - 2026-03-22

### Added
- **GRPO training pipeline** with verifier-based reward models
  - `GRPOConfig` and `run_grpo_training` for Group Relative Policy Optimization
  - CLI command `biorlhf-grpo --config <path>` for GRPO training
- **Verifier system (V1-V4)** for multi-dimensional reward scoring
  - V1 (Factual): Exact match scoring for DEG counts, tissue names, directions
  - V2 (Pathway): Pathway/gene set enrichment validation (Hallmark, KEGG)
  - V3 (Consistency): Internal logical consistency checking
  - V4 (Uncertainty): Calibration and epistemic humility scoring
  - `RewardComposer` for weighted multi-reward composition
- **GRPO dataset module** (`grpo_dataset.py`) for prompt-based training data with hold-out tissues
- **GeneLab data loader** (`genelabloader.py`) for NES conservation questions
- **Calibration evaluation** (`calibration.py`) with Expected Calibration Error (ECE) scoring
- **Question generator** (`question_generator.py`) for automated biological question creation
- GRPO training configs: `grpo_mve.json` (MVE) and `grpo_full_v2.json` (full multi-reward)
- SLURM job scripts for GRPO training on HPC clusters
- Hold-out tissue evaluation (eye, thymus) for generalization testing

### Changed
- Bumped version to 0.2.0
- Updated README with GRPO architecture, verifier system, and latest results
- V1 factual verifier: reduced negation window from 30 to 12 characters to prevent cross-clause false negation
- V1/V4 verifiers: smoothed reward scoring for GRPO (continuous instead of binary)
- Updated HPC training guide with GRPO workflow and SLURM configurations
- Updated dependencies: TRL >= 0.14.0 (GRPO support), PEFT >= 0.6.0
- Lazy imports in `evaluation/__init__.py` to avoid torch dependency at import time

### Training Results
- **MVE experiment** (G=4, V1+V4): Reward improved from 0.547 (SFT) to 0.650 (+19%), ECE reduced from 0.258 to 0.078 (-70%)
- **Full v2 experiment** (G=16, V1-V4): Multi-reward training with zero-variance batch fraction <5% (vs 50% in MVE)

### Fixed
- LoRA adapter loading: properly load base model first, then merge SFT adapter
- Tokenizer loading from adapter directories in Transformers 4.57+
- TRL GRPOConfig: `scale_rewards` as string type, explicit `loss_type="grpo"`
- Batch size compatibility: both `per_device_eval_batch_size` and `generation_batch_size` divisible by `num_generations`
- BioEval ground truth serialization for dict-type answers

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
- GitHub Actions CI workflow for automated testing
- Pre-commit hooks configuration
- Unit tests for ground truth data and dataset creation
- Example scripts (quickstart, train_sft, evaluate_model)
- CONTRIBUTING.md guidelines

### Training Results
- Achieved 90% overall accuracy on biological reasoning tasks
- 100% calibration accuracy (appropriate uncertainty expression)
- Successfully trained on 363 examples
- Model comparison study: Mistral-7B (90%) > Qwen2.5-7B (40%) > Phi-2 (25%)

### Data
- `kmp_sft_final.json`: 363 training examples
- `kmp_test_set.json`: 20-question evaluation set
- `kmp_dpo_preferences.json`: Preference pairs for DPO training

---

## Version History Summary

| Version | Date | Highlights |
|---------|------|------------|
| 0.3.0 | 2026-04-04 | Phase 4 ablation, HF models, sanitized HPC paths |
| 0.2.0 | 2026-03-22 | GRPO pipeline, V1-V4 verifiers, multi-reward training |
| 0.1.0 | 2025-01-09 | Initial release with SFT/DPO pipelines |

[0.3.0]: https://github.com/jang1563/BioRLHF/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/jang1563/BioRLHF/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/jang1563/BioRLHF/releases/tag/v0.1.0
