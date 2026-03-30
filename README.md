# BioRLHF

[![CI](https://github.com/jang1563/BioRLHF/actions/workflows/ci.yml/badge.svg)](https://github.com/jang1563/BioRLHF/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Biological Reinforcement Learning from Human Feedback** — A framework for fine-tuning LLMs on biological reasoning tasks using SFT, DPO, and GRPO with verifier-based reward models for factual accuracy, calibrated uncertainty, and chain-of-thought reasoning.

## Highlights

- **Three-stage training pipeline**: SFT → DPO → GRPO with verifier-based rewards
- **Multi-reward GRPO**: Four composable verifiers (factual, pathway, consistency, uncertainty) with configurable weights
- **+19% reward improvement** over SFT baseline using GRPO (0.650 vs 0.547)
- **-70% calibration error**: ECE reduced from 0.258 to 0.078 after GRPO
- **90% accuracy** on domain-specific biological reasoning tasks (SFT stage)
- **Learns from 363 examples** — efficient domain adaptation from spaceflight transcriptomics data

## Key Results

### GRPO Training (Phase 3)

| Metric | SFT Baseline | After GRPO | Improvement |
|--------|-------------|------------|-------------|
| Avg Reward | 0.547 | 0.650 | +19% |
| ECE (Calibration Error) | 0.258 | 0.078 | -70% |

**GRPO Configuration (Full v2):**
- 16 generations per prompt (G=16) for robust advantage estimation
- Multi-reward: V1 (factual, 0.35) + V2 (pathway, 0.30) + V3 (consistency, 0.15) + V4 (uncertainty, 0.20)
- KL penalty beta=0.02, 2 iterations per batch, group-normalized rewards

### Model Comparison (SFT, 20-question evaluation)

| Model | Overall | Factual | Reasoning | Calibration |
|-------|---------|---------|-----------|-------------|
| **Mistral-7B** | **90.0%** | 80.0% | 100.0% | 100.0% |
| Qwen2.5-7B | 40.0% | 30.0% | 80.0% | 20.0% |
| Phi-2 | 25.0% | 20.0% | 60.0% | 0.0% |

### SFT Training Progression

| Version | Accuracy | Key Improvement |
|---------|----------|-----------------|
| v1 (Base SFT) | ~20% | Format learned, facts wrong |
| v2 (Expanded) | ~60% | More examples helped |
| v3 (Fact Drilling) | ~80% | Repetition fixed key facts |
| v4 (Advanced) | ~85% | Chain-of-thought, calibration |
| **Final** | **90%** | Targeted drilling for remaining errors |

## Installation

### From Source

```bash
git clone https://github.com/jang1563/BioRLHF.git
cd BioRLHF
pip install -e .
```

### With Development Dependencies

```bash
pip install -e ".[dev]"
```

### GPU Requirements

- NVIDIA GPU with 48GB+ VRAM recommended (A40 or A100)
- 24GB+ VRAM sufficient for SFT/DPO with 4-bit quantization
- CUDA 12.1+ recommended

## Quick Start

### SFT Training

```python
from biorlhf import SFTTrainingConfig, run_sft_training

config = SFTTrainingConfig(
    model_name="mistralai/Mistral-7B-v0.3",
    dataset_path="data/kmp_sft_final.json",
    output_dir="./my_sft_model",
    num_epochs=10,
    learning_rate=1e-4,
)

model_path = run_sft_training(config)
```

### GRPO Training with Verifiers

```bash
# Using the CLI
biorlhf-grpo --config configs/grpo_full_v2.json
```

```python
# Or programmatically
from biorlhf.training.grpo import GRPOConfig, run_grpo_training

config = GRPOConfig.from_json("configs/grpo_full_v2.json")
run_grpo_training(config)
```

### Creating a Dataset

```python
from biorlhf.data import create_sft_dataset

dataset = create_sft_dataset(
    output_path="my_dataset.json",
    include_calibration=True,
    include_chain_of_thought=True,
)

print(f"Created {len(dataset)} training examples")
```

### Evaluating a Model

```python
from biorlhf import evaluate_model

result = evaluate_model(
    model_path="./my_sft_model",
    test_questions_path="data/kmp_test_set.json",
)

print(f"Overall Accuracy: {result.overall_accuracy:.1%}")
print(f"Factual: {result.factual_accuracy:.1%}")
print(f"Reasoning: {result.reasoning_accuracy:.1%}")
print(f"Calibration: {result.calibration_accuracy:.1%}")
```

### Running Inference

```python
from biorlhf.utils import load_model_for_inference, generate_response

model, tokenizer = load_model_for_inference(
    model_path="./my_sft_model",
    base_model="mistralai/Mistral-7B-v0.3",
)

prompt = "### Instruction:\nWhich tissue is most sensitive to ionizing radiation?\n\n### Response:\n"
response = generate_response(model, tokenizer, prompt)
print(response)
```

## Architecture

### Three-Stage Training Pipeline

```
Stage 1: SFT                    Stage 2: DPO                Stage 3: GRPO
(Supervised Fine-Tuning)        (Direct Preference          (Group Relative Policy
                                 Optimization)               Optimization)

Mistral-7B-v0.3                 SFT model                   SFT model (merged)
      |                              |                            |
   LoRA (r=64, alpha=128)       Preference pairs            Generate G=16 completions
      |                              |                            |
   363 training examples         Ranked responses           Score with V1-V4 verifiers
      |                              |                            |
   10 epochs, lr=1e-4            beta=0.1                   Multi-reward composition
      |                              |                            |
   SFT Adapter                  DPO Model                   GRPO Model
```

### Verifier-Based Reward System (V1-V4)

| Verifier | Name | Weight | What It Scores |
|----------|------|--------|----------------|
| **V1** | Factual | 0.35 | Exact match of biological facts (DEG counts, tissue names, directions) |
| **V2** | Pathway | 0.30 | Correct pathway/gene set enrichment references (Hallmark, KEGG) |
| **V3** | Consistency | 0.15 | Internal logical consistency within the response |
| **V4** | Uncertainty | 0.20 | Appropriate confidence calibration and epistemic humility |

The verifiers are composable via `RewardComposer` and can be individually weighted:

```python
from biorlhf.verifiers import RewardComposer

composer = RewardComposer(
    active_verifiers=["V1", "V2", "V3", "V4"],
    weights={"V1": 0.35, "V2": 0.30, "V3": 0.15, "V4": 0.20},
)

reward = composer.score(question, response, ground_truth)
```

## Dataset

Training data is derived from a 2x2x2 factorial transcriptomic study:

- **Drug**: Kaempferol (KMP) vs Control
- **Stressor 1**: Hindlimb Unloading (HU) — simulates microgravity
- **Stressor 2**: Ionizing Radiation (IR) — simulates space radiation
- **Tissues**: Heart, Hippocampus, Liver, Soleus (+ Eye, Thymus for GRPO hold-out)

### Training Example Types

| Type | Count | Purpose |
|------|-------|---------|
| Factual Q&A | ~150 | Specific facts (DEG counts, tissue types) |
| Chain-of-Thought | ~50 | Step-by-step reasoning |
| Calibration | ~30 | Uncertainty expression |
| Multi-hop Reasoning | ~30 | Integrating multiple facts |
| Error Correction | ~20 | Learning from mistakes |

### Ground Truth Data

```python
from biorlhf.data import (
    STRESSOR_EFFECTS,
    KMP_EFFECTS,
    INTERACTIONS,
    TISSUE_TYPES,
    OXPHOS_PATTERNS,
)

# Example: Get DEG counts for stressors
print(STRESSOR_EFFECTS["Hippocampus"])
# {'HU': 1555, 'IR': 5477, 'HU_IR': 5510}
```

## Project Structure

```
BioRLHF/
├── src/biorlhf/              # Main package
│   ├── training/             # SFT, DPO, and GRPO trainers
│   ├── data/                 # Dataset creation & ground truth
│   ├── evaluation/           # Model evaluation & calibration
│   ├── verifiers/            # V1-V4 reward verifiers
│   │   ├── factual.py        #   V1: Factual accuracy scoring
│   │   ├── pathway.py        #   V2: Pathway enrichment scoring
│   │   ├── consistency.py    #   V3: Logical consistency scoring
│   │   ├── uncertainty.py    #   V4: Calibration/uncertainty scoring
│   │   └── composer.py       #   Multi-reward composition
│   ├── utils/                # Model loading, inference helpers
│   └── cli.py                # Command-line interface
├── configs/                  # Training configurations
│   ├── grpo_mve.json         #   Minimum viable experiment
│   └── grpo_full_v2.json     #   Full multi-reward training
├── data/                     # Training datasets
│   ├── kmp_sft_final.json    #   363 SFT training examples
│   └── kmp_test_set.json     #   20-question evaluation set
├── examples/                 # Usage examples
├── scripts/                  # SLURM job scripts & HPC guide
├── tests/                    # Unit tests
└── docs/                     # Documentation
```

## Scientific Contributions

### 1. Verifier-Based GRPO Improves Calibration

- GRPO with V1-V4 verifiers reduced calibration error (ECE) by 70%
- Multi-reward composition outperforms single-reward training
- G=16 generations dramatically reduces zero-variance batches (from 50% to <5%)

### 2. Fact Drilling Works for SFT

- Initial training: 20% accuracy on key facts
- After targeted repetition: 100% accuracy on drilled facts
- LLMs need explicit reinforcement of specific domain facts

### 3. Calibration is Learnable

- Trained on "I cannot determine X from this data" examples
- Mistral achieved 100% calibration accuracy at SFT stage
- GRPO further improved calibration via the V4 uncertainty verifier

### 4. DPO is Fragile for Domain Knowledge

- Aggressive DPO (beta=0.05) destroyed learned knowledge
- Model hallucinated unrelated content
- Preference learning needs careful tuning in specialized domains

### 5. Architecture Matters More Than Size

- Mistral-7B >> Qwen2.5-7B despite similar parameter counts
- Phi-2 (2.7B) insufficient for complex biological reasoning
- Model selection is critical for domain fine-tuning

## Key Learnings for AI Safety

1. **Honesty is trainable** — Models can learn appropriate epistemic humility
2. **Domain grounding matters** — Anchoring to experimental truth prevents hallucination
3. **Multi-reward > single reward** — Decomposing correctness into verifiable dimensions improves learning signal
4. **Preference learning is fragile** — DPO can catastrophically forget domain knowledge
5. **Evaluation drives improvement** — Systematic testing reveals specific failure modes

## Related Projects

- **[SpaceOmicsBench](https://github.com/jang1563/SpaceOmicsBench)** — 115-question benchmark for LLMs on spaceflight biomedical data

## Citation

If you use BioRLHF in your research, please cite:

```bibtex
@software{biorlhf2026,
  author = {Kim, JangKeun},
  title = {BioRLHF: Biological Reinforcement Learning from Human Feedback},
  year = {2026},
  url = {https://github.com/jang1563/BioRLHF},
  note = {Fine-tuning LLMs for biological reasoning with verifier-based GRPO}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Developed by JangKeun Kim, Weill Cornell Medicine*
