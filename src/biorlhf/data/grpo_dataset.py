"""
Unified GRPO dataset builder for BioGRPO.

Merges pathway questions from GeneLab, calibration tasks from BioEval,
and domain questions from SpaceOmicsBench into a single TRL-compatible
dataset with multi-dimensional ground truth.
"""

import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import json

from datasets import Dataset as HFDataset

from biorlhf.data.question_generator import generate_all_questions

# ── External data paths (configurable via env vars for HPC) ───────────────
BIOEVAL_DATA = Path(os.environ.get(
    "BIOEVAL_DATA",
    "/Users/jak4013/Dropbox/Bioinformatics/Claude/Evaluation_model/BioEval/data",
))
SPACEOMICS_DATA = Path(os.environ.get(
    "SPACEOMICS_DATA",
    "/Users/jak4013/Dropbox/Bioinformatics/Claude/SpaceOmicsBench/v3/evaluation/llm",
))


def load_bioeval_for_grpo() -> List[Dict]:
    """Load BioEval tasks that have verifiable ground truth.

    Selects:
    - calibration tasks (30) → V4 training
    - bioambiguity tasks (45) → V3 training
    - Other verifiable tasks → V2 training
    """
    samples: List[Dict] = []
    base_path = BIOEVAL_DATA / "bioeval_v060_base.jsonl"
    if not base_path.exists():
        return samples

    with open(base_path) as f:
        for line in f:
            task = json.loads(line)
            component = task.get("component", "")
            prompt = task.get("prompt", "")
            gt = task.get("ground_truth", "{}")

            # Ensure ground_truth is a JSON string
            gt_str = json.dumps(gt) if isinstance(gt, dict) else gt

            if component == "calibration":
                samples.append({
                    "prompt": prompt,
                    "ground_truth": gt_str,
                    "question_type": "calibration",
                    "applicable_verifiers": json.dumps(["V4"]),
                    "source": "bioeval",
                    "tissue": "general",
                    "difficulty": "medium",
                })
            elif component == "bioambiguity":
                samples.append({
                    "prompt": prompt,
                    "ground_truth": gt_str,
                    "question_type": "context_dependent",
                    "applicable_verifiers": json.dumps(["V3", "V4"]),
                    "source": "bioeval",
                    "tissue": "general",
                    "difficulty": "hard",
                })
            elif component in ("causalbio", "designcheck", "adversarial"):
                samples.append({
                    "prompt": prompt,
                    "ground_truth": gt_str,
                    "question_type": component,
                    "applicable_verifiers": json.dumps(["V2"]),
                    "source": "bioeval",
                    "tissue": "general",
                    "difficulty": "hard" if component == "adversarial" else "medium",
                })

    return samples


def load_spaceomics_for_grpo() -> List[Dict]:
    """Load SpaceOmicsBench v3 questions with ground truth."""
    samples: List[Dict] = []
    qbank_path = SPACEOMICS_DATA / "question_bank_v3.json"
    if not qbank_path.exists():
        return samples

    with open(qbank_path) as f:
        qbank = json.load(f)

    questions = qbank.get("questions", [])
    for q in questions:
        gt = {
            "key_facts": q.get("ground_truth_key_facts", []),
            "expected_reasoning": q.get("expected_reasoning", []),
        }

        verifiers = ["V2"]
        if q.get("requires_uncertainty_calibration", False):
            verifiers.append("V4")
            gt["expected_confidence"] = "medium"

        samples.append({
            "prompt": q["question"],
            "ground_truth": json.dumps(gt),
            "question_type": q.get("category", "factual"),
            "applicable_verifiers": json.dumps(verifiers),
            "source": "spaceomics",
            "tissue": "general",
            "difficulty": q.get("difficulty", "medium"),
        })

    return samples


def build_grpo_dataset(
    db: str = "hallmark",
    seed: int = 42,
    hold_out_tissues: Optional[List[str]] = None,
) -> Tuple[HFDataset, HFDataset]:
    """Build the full GRPO training dataset with train/eval split.

    Args:
        db: Pathway database to use for GeneLab questions.
        seed: Random seed for splitting.
        hold_out_tissues: If set, questions from these tissues go to eval.
            Otherwise uses random 10% split.

    Returns:
        (train_dataset, eval_dataset) as HuggingFace Datasets.

    Dataset columns (TRL-compatible):
        - prompt: str (required by GRPOTrainer)
        - ground_truth: str (JSON, forwarded to reward function)
        - question_type: str (forwarded to reward function)
        - applicable_verifiers: str (JSON list, forwarded to reward function)
        - source: str ("genelab", "bioeval", "spaceomics")
        - tissue: str (for LOMO splitting)
        - difficulty: str ("easy", "medium", "hard")
    """
    all_samples: List[Dict] = []

    # 1. GeneLab pathway questions
    genelab_qs = generate_all_questions(db)
    for q in genelab_qs:
        all_samples.append({
            "prompt": q.prompt,
            "ground_truth": json.dumps(q.ground_truth),
            "question_type": q.question_type,
            "applicable_verifiers": json.dumps(q.applicable_verifiers),
            "source": "genelab",
            "tissue": q.tissue,
            "difficulty": q.difficulty,
        })

    # 2. BioEval tasks
    all_samples.extend(load_bioeval_for_grpo())

    # 3. SpaceOmicsBench questions
    all_samples.extend(load_spaceomics_for_grpo())

    if not all_samples:
        raise ValueError("No training samples generated. Check data paths.")

    # Convert to HF Dataset
    full_dataset = HFDataset.from_list(all_samples)

    # Split strategy
    if hold_out_tissues:
        train_indices = []
        eval_indices = []
        for i, sample in enumerate(all_samples):
            if sample["tissue"] in hold_out_tissues:
                eval_indices.append(i)
            else:
                train_indices.append(i)
        if not eval_indices:
            # Fallback: random split if no matching tissues
            split = full_dataset.train_test_split(test_size=0.1, seed=seed)
            return split["train"], split["test"]
        train_dataset = full_dataset.select(train_indices)
        eval_dataset = full_dataset.select(eval_indices)
    else:
        split = full_dataset.train_test_split(test_size=0.1, seed=seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]

    return train_dataset, eval_dataset


def get_dataset_stats(dataset: HFDataset) -> Dict:
    """Return summary statistics for a GRPO dataset."""
    sources = {}
    types = {}
    tissues = {}
    difficulties = {}

    for sample in dataset:
        src = sample["source"]
        sources[src] = sources.get(src, 0) + 1
        qt = sample["question_type"]
        types[qt] = types.get(qt, 0) + 1
        t = sample["tissue"]
        tissues[t] = tissues.get(t, 0) + 1
        d = sample["difficulty"]
        difficulties[d] = difficulties.get(d, 0) + 1

    return {
        "total": len(dataset),
        "by_source": sources,
        "by_question_type": types,
        "by_tissue": tissues,
        "by_difficulty": difficulties,
    }
