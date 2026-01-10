# BioRLHF

[![CI](https://github.com/jang1563/BioRLHF/actions/workflows/ci.yml/badge.svg)](https://github.com/jang1563/BioRLHF/actions/workflows/ci.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**Biological Reinforcement Learning from Human Feedback** — A framework for fine-tuning LLMs on biological reasoning tasks with emphasis on factual accuracy, chain-of-thought reasoning, and uncertainty calibration.

## Highlights

- **90% accuracy** on domain-specific biological reasoning tasks
- **100% calibration accuracy** — model knows what it doesn't know
- **Learns from 363 examples** — efficient domain adaptation
- **Supports SFT and DPO** training pipelines

## Key Results

### Model Comparison (20-question evaluation)

| Model | Overall | Factual | Reasoning | Calibration |
|-------|---------|---------|-----------|-------------|
| **Mistral-7B** | **90.0%** | 80.0% | 100.0% | 100.0% |
| Qwen2.5-7B | 40.0% | 30.0% | 80.0% | 20.0% |
| Phi-2 | 25.0% | 20.0% | 60.0% | 0.0% |

### Training Progression

| Version | Accuracy | Key Improvement |
|---------|----------|-----------------|
| v1 (Base SFT) | ~20% | Format learned, facts wrong |
| v2 (Expanded) | ~60% | More examples helped |
| v3 (Fact Drilling) | ~80% | Repetition fixed key facts |
| v4 (Advanced) | ~85% | Chain-of-thought, calibration |
| **Final** | **90%** | Targeted drilling for remaining errors |

## Installation

### From PyPI (coming soon)

```bash
pip install BioRLHF
```

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

- NVIDIA GPU with 24GB+ VRAM (for 7B models with 4-bit quantization)
- CUDA 11.8+ recommended

## Quick Start

### Training a Model

```python
from biorlhf import SFTTrainingConfig, run_sft_training

# Configure training
config = SFTTrainingConfig(
    model_name="mistralai/Mistral-7B-v0.3",
    dataset_path="data/kmp_sft_final.json",
    output_dir="./my_biorlhf_model",
    num_epochs=10,
    learning_rate=1e-4,
)

# Run training
model_path = run_sft_training(config)
```

### Creating a Dataset

```python
from biorlhf.data import create_sft_dataset

# Generate dataset from ground truth biological data
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
    model_path="./my_biorlhf_model",
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
    model_path="./my_biorlhf_model",
    base_model="mistralai/Mistral-7B-v0.3",
)

prompt = "### Instruction:\nWhich tissue is most sensitive to ionizing radiation?\n\n### Response:\n"
response = generate_response(model, tokenizer, prompt)
print(response)
```

## Dataset

Training data is derived from a 2×2×2 factorial transcriptomic study:

- **Drug**: Kaempferol (KMP) vs Control
- **Stressor 1**: Hindlimb Unloading (HU) — simulates microgravity
- **Stressor 2**: Ionizing Radiation (IR) — simulates space radiation
- **Tissues**: Heart, Hippocampus, Liver, Soleus

### Training Example Types

| Type | Count | Purpose |
|------|-------|---------|
| Factual Q&A | ~150 | Specific facts (DEG counts, tissue types) |
| Chain-of-Thought | ~50 | Step-by-step reasoning |
| Calibration | ~30 | Uncertainty expression |
| Multi-hop Reasoning | ~30 | Integrating multiple facts |
| Error Correction | ~20 | Learning from mistakes |

### Ground Truth Data

Access the biological ground truth data directly:

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
├── src/biorlhf/           # Main package
│   ├── training/          # SFT and DPO trainers
│   ├── data/              # Dataset creation utilities
│   ├── evaluation/        # Model evaluation
│   └── utils/             # Helper functions
├── data/                  # Training datasets
│   ├── kmp_sft_final.json
│   └── kmp_test_set.json
├── examples/              # Usage examples
├── scripts/               # Training scripts
├── tests/                 # Unit tests
└── docs/                  # Documentation
```

## Scientific Contributions

### 1. Fact Drilling Works
- Initial training: 20% accuracy on key facts
- After targeted repetition: 100% accuracy on drilled facts
- **Insight**: LLMs need explicit reinforcement of specific facts

### 2. Calibration is Learnable
- Trained on "I cannot determine X from this data" examples
- Mistral achieved 100% calibration accuracy
- **Insight**: Uncertainty expression can be taught, not just prompted

### 3. DPO is Fragile for Domain Knowledge
- Aggressive DPO (β=0.05) destroyed learned knowledge
- Model hallucinated unrelated content
- **Insight**: Preference learning needs careful calibration in specialized domains

### 4. Architecture Matters More Than Size
- Mistral-7B >> Qwen2.5-7B despite similar parameter counts
- Phi-2 (2.7B) insufficient for complex biological reasoning
- **Insight**: Model selection is critical for domain fine-tuning

## Key Learnings for AI Safety

1. **Honesty is trainable** — Models can learn appropriate epistemic humility
2. **Domain grounding matters** — Anchoring to experimental truth prevents hallucination
3. **Preference learning is fragile** — DPO can catastrophically forget domain knowledge
4. **Evaluation drives improvement** — Systematic testing reveals specific failure modes

## Related Projects

- **[SpaceOmicsBench](https://github.com/jang1563/SpaceOmicsBench)** — 115-question benchmark for LLMs on spaceflight biomedical data
- **CAMELOT** — Adversarial robustness benchmark for biological reasoning

## Citation

If you use BioRLHF in your research, please cite:

```bibtex
@software{biorlhf2026,
  author = {Kim, JangKeun},
  title = {BioRLHF: Biological Reinforcement Learning from Human Feedback},
  year = {2026},
  url = {https://github.com/jang1563/BioRLHF}
}
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

*Developed by JangKeun Kim, Weill Cornell Medicine*
