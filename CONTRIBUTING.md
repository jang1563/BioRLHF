# Contributing to BioRLHF

Thank you for your interest in contributing to BioRLHF! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Style Guidelines](#style-guidelines)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributors of all backgrounds and experience levels.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/YOUR_USERNAME/BioRLHF.git
   cd BioRLHF
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/ORIGINAL_OWNER/BioRLHF.git
   ```

## Development Setup

### Prerequisites

- Python 3.9 or higher
- CUDA-compatible GPU (recommended for training)
- Git

### Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the package in development mode with all dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

3. Install pre-commit hooks:
   ```bash
   pre-commit install
   ```

### Verify Installation

```bash
# Run tests
pytest

# Check code formatting
black --check src/ tests/
ruff check src/ tests/
```

## Making Changes

### Branch Naming

Create a descriptive branch for your changes:

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation updates
- `refactor/description` - Code refactoring

Example:
```bash
git checkout -b feature/add-new-evaluation-metric
```

### Commit Messages

Write clear, concise commit messages:

- Use the present tense ("Add feature" not "Added feature")
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit the first line to 72 characters
- Reference issues when applicable

Example:
```
Add calibration accuracy metric to evaluation module

- Implement uncertainty detection in model responses
- Add tests for calibration scoring
- Update documentation with new metric

Closes #42
```

## Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=biorlhf --cov-report=html

# Run specific test file
pytest tests/test_dataset.py

# Run tests matching a pattern
pytest -k "test_evaluation"
```

### Writing Tests

- Place tests in the `tests/` directory
- Mirror the source structure (e.g., `src/biorlhf/data/dataset.py` → `tests/test_dataset.py`)
- Use descriptive test names
- Include docstrings explaining what the test verifies

Example:
```python
def test_load_dataset_returns_expected_format():
    """Verify that load_dataset returns a HuggingFace Dataset object."""
    dataset = load_dataset("kmp_sft_final.json")
    assert isinstance(dataset, Dataset)
    assert "text" in dataset.column_names
```

## Submitting Changes

### Before Submitting

1. **Sync with upstream**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Run all checks**:
   ```bash
   # Format code
   black src/ tests/

   # Check linting
   ruff check src/ tests/

   # Run tests
   pytest
   ```

3. **Update documentation** if needed

### Pull Request Process

1. Push your branch to your fork:
   ```bash
   git push origin feature/your-feature
   ```

2. Open a Pull Request on GitHub

3. Fill in the PR template with:
   - Description of changes
   - Related issue numbers
   - Testing performed
   - Screenshots (if UI changes)

4. Wait for review and address feedback

### Review Checklist

- [ ] Code follows style guidelines
- [ ] Tests pass locally
- [ ] New code has appropriate test coverage
- [ ] Documentation is updated
- [ ] Commit messages are clear

## Style Guidelines

### Python Code Style

We use [Black](https://black.readthedocs.io/) for code formatting and [Ruff](https://docs.astral.sh/ruff/) for linting.

Key conventions:
- Line length: 88 characters (Black default)
- Use type hints where practical
- Write docstrings for public functions and classes
- Use meaningful variable names

### Docstring Format

Use Google-style docstrings:

```python
def evaluate_model(model_path: str, test_data: str) -> dict:
    """Evaluate a trained model on test data.

    Args:
        model_path: Path to the trained model directory.
        test_data: Path to the test dataset JSON file.

    Returns:
        Dictionary containing evaluation metrics including
        factual_accuracy, reasoning_accuracy, and calibration_score.

    Raises:
        FileNotFoundError: If model_path or test_data doesn't exist.

    Example:
        >>> results = evaluate_model("./model", "test.json")
        >>> print(results["factual_accuracy"])
        0.90
    """
```

### Import Order

Organize imports in this order:
1. Standard library
2. Third-party packages
3. Local imports

Example:
```python
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

from biorlhf.data import load_dataset
from biorlhf.utils import setup_quantization
```

## Questions?

If you have questions about contributing, feel free to:
- Open an issue for discussion
- Reach out to the maintainers

Thank you for contributing to BioRLHF!
