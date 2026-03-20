"""
Pathway reasoning question generator for BioGRPO.

Generates verifiable QA pairs from GeneLab fGSEA pathway data.
Each question has structured ground truth for scoring by the verifier stack.
"""

from typing import List, Dict, Set
from dataclasses import dataclass, field

from biorlhf.data.genelabloader import (
    get_consensus_directions,
    get_disagreeing_pathways,
    get_pathway_directions,
    load_nes_conservation,
    TISSUE_MISSIONS,
)


@dataclass
class GRPOQuestion:
    """A question with verifiable ground truth for GRPO training."""
    prompt: str
    ground_truth: Dict
    tissue: str
    db: str
    question_type: str          # "direction", "comparison", "consistency", "uncertainty"
    applicable_verifiers: List[str]
    difficulty: str             # "easy", "medium", "hard"
    metadata: Dict = field(default_factory=dict)


def _clean_pathway_name(pathway: str) -> str:
    """HALLMARK_OXIDATIVE_PHOSPHORYLATION → Oxidative Phosphorylation"""
    for prefix in ("HALLMARK_", "KEGG_", "REACTOME_", "MITOCARTA_"):
        pathway = pathway.replace(prefix, "")
    return pathway.replace("_", " ").title()


# ── Question generators ────────────────────────────────────────────────────

def generate_direction_questions(
    tissue: str,
    db: str = "hallmark",
    padj_threshold: float = 0.05,
) -> List[GRPOQuestion]:
    """Generate V1-targetable questions about pathway direction."""
    consensus = get_consensus_directions(tissue, db, min_missions=2, padj_threshold=padj_threshold)
    questions: List[GRPOQuestion] = []

    for pathway, info in consensus.items():
        pw = _clean_pathway_name(pathway)
        n_agree = info["n_agree"]

        # Type 1: Direct direction question (easy/medium)
        questions.append(GRPOQuestion(
            prompt=(
                f"In mouse {tissue} tissue during spaceflight, is the "
                f"{pw} pathway upregulated or downregulated based on "
                f"gene set enrichment analysis? "
                f"Provide your confidence level."
            ),
            ground_truth={
                "pathway": pathway,
                "direction": info["direction"],
                "n_supporting_missions": n_agree,
                "expected_confidence": "high" if n_agree >= 3 else "medium",
            },
            tissue=tissue,
            db=db,
            question_type="direction",
            applicable_verifiers=["V1", "V4"],
            difficulty="easy" if n_agree >= 3 else "medium",
        ))

        # Type 2: Mechanistic reasoning question (medium/hard)
        direction_word = "activation" if info["direction"] == "UP" else "suppression"
        questions.append(GRPOQuestion(
            prompt=(
                f"Explain the biological significance of {pw} pathway "
                f"{direction_word} in mouse {tissue} under spaceflight conditions. "
                f"What mechanisms might drive this change? "
                f"State your confidence in the direction and magnitude."
            ),
            ground_truth={
                "pathway": pathway,
                "direction": info["direction"],
                "n_supporting_missions": n_agree,
                "requires_mechanism": True,
                "expected_confidence": "medium",
            },
            tissue=tissue,
            db=db,
            question_type="direction",
            applicable_verifiers=["V1", "V2", "V4"],
            difficulty="medium" if n_agree >= 3 else "hard",
        ))

    return questions


def generate_comparison_questions(
    db: str = "hallmark",
    padj_threshold: float = 0.05,
) -> List[GRPOQuestion]:
    """Generate cross-tissue comparison questions (V1 + V3 targetable)."""
    questions: List[GRPOQuestion] = []

    # Collect consensus directions across tissues (exclude skin subsites for cleaner Qs)
    tissue_dirs: Dict[str, Dict[str, Dict]] = {}
    comparison_tissues = ["liver", "gastrocnemius", "kidney", "thymus", "eye"]
    for tissue in comparison_tissues:
        consensus = get_consensus_directions(tissue, db, min_missions=2, padj_threshold=padj_threshold)
        if consensus:
            tissue_dirs[tissue] = consensus

    if len(tissue_dirs) < 2:
        return questions

    # Find pathways in 2+ tissues
    all_pathways: Set[str] = set()
    for dirs in tissue_dirs.values():
        all_pathways.update(dirs.keys())

    for pathway in sorted(all_pathways):
        tissues_with = {
            t: d[pathway]
            for t, d in tissue_dirs.items()
            if pathway in d
        }
        if len(tissues_with) < 2:
            continue

        pw = _clean_pathway_name(pathway)
        tissue_list = sorted(tissues_with.keys())
        directions_set = {info["direction"] for info in tissues_with.values()}
        is_consistent = len(directions_set) == 1

        questions.append(GRPOQuestion(
            prompt=(
                f"Compare the response of the {pw} pathway to spaceflight "
                f"across {', '.join(tissue_list)} tissues in mice. "
                f"Is the direction of change consistent or tissue-specific? "
                f"Explain the biological basis for any differences."
            ),
            ground_truth={
                "pathway": pathway,
                "tissue_directions": {
                    t: info["direction"] for t, info in tissues_with.items()
                },
                "is_consistent": is_consistent,
                "n_tissues": len(tissues_with),
            },
            tissue="multi",
            db=db,
            question_type="comparison",
            applicable_verifiers=["V1", "V3", "V4"],
            difficulty="hard",
        ))

    return questions


def generate_uncertainty_questions(
    tissue: str,
    db: str = "hallmark",
    padj_threshold: float = 0.05,
) -> List[GRPOQuestion]:
    """Generate questions where missions disagree → model should express uncertainty."""
    disagreeing = get_disagreeing_pathways(tissue, db, padj_threshold)
    questions: List[GRPOQuestion] = []

    for pathway, info in disagreeing.items():
        pw = _clean_pathway_name(pathway)
        questions.append(GRPOQuestion(
            prompt=(
                f"Is the {pw} pathway consistently activated or suppressed "
                f"in mouse {tissue} across different spaceflight missions? "
                f"How confident are you in the direction of change?"
            ),
            ground_truth={
                "pathway": pathway,
                "missions_up": info["missions_up"],
                "missions_down": info["missions_down"],
                "missions_ns": info["missions_ns"],
                "correct_behavior": "context_dependent",
                "expected_confidence": "low",
            },
            tissue=tissue,
            db=db,
            question_type="uncertainty",
            applicable_verifiers=["V1", "V4"],
            difficulty="hard",
        ))

    return questions


def generate_conservation_questions(
    db: str = "hallmark",
) -> List[GRPOQuestion]:
    """Generate questions about NES conservation across missions."""
    conservation = load_nes_conservation(db)
    if not conservation:
        return []

    questions: List[GRPOQuestion] = []
    data = conservation.get("data", conservation)

    for tissue, info in data.items():
        if not isinstance(info, dict):
            continue
        mean_r = info.get("nes_mean_r")
        if mean_r is None:
            continue

        if mean_r > 0.5:
            conservation_level = "highly conserved"
            expected_conf = "high"
        elif mean_r > 0.2:
            conservation_level = "moderately conserved"
            expected_conf = "medium"
        else:
            conservation_level = "poorly conserved"
            expected_conf = "medium"

        questions.append(GRPOQuestion(
            prompt=(
                f"How conserved are pathway-level responses to spaceflight "
                f"across different missions in mouse {tissue}? "
                f"Are the enrichment patterns reproducible?"
            ),
            ground_truth={
                "tissue": tissue,
                "nes_mean_r": mean_r,
                "conservation_level": conservation_level,
                "expected_confidence": expected_conf,
                "key_facts": [
                    f"Mean pairwise NES correlation across missions is {mean_r:.3f}",
                    f"Pathway responses in {tissue} are {conservation_level}",
                ],
            },
            tissue=tissue,
            db=db,
            question_type="direction",
            applicable_verifiers=["V2", "V4"],
            difficulty="medium",
        ))

    return questions


def generate_all_questions(db: str = "hallmark") -> List[GRPOQuestion]:
    """Generate the full question set from GeneLab data."""
    all_q: List[GRPOQuestion] = []

    for tissue in TISSUE_MISSIONS:
        all_q.extend(generate_direction_questions(tissue, db))
        all_q.extend(generate_uncertainty_questions(tissue, db))

    all_q.extend(generate_comparison_questions(db))
    all_q.extend(generate_conservation_questions(db))

    return all_q
