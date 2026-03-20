"""
V1: Pathway Direction Verifier.

Extracts directional claims about biological pathways from model responses
and compares them against fGSEA NES direction ground truth.

Scoring:
  1.0 — correct direction claimed
  0.5 — mixed/contradictory claims
  0.3 — no directional claim extracted
  0.0 — wrong direction claimed
"""

import re
from typing import Dict, List, Tuple

from biorlhf.verifiers.base import BaseVerifier, VerifierResult

# ── Direction indicator patterns ──────────────────────────────────────────
UP_INDICATORS = [
    r"\bupregulat\w*\b",
    r"\bactivat\w*\b",
    r"\bincreas\w*\b",
    r"\belevat\w*\b",
    r"\benhanc\w*\b",
    r"\binduced?\b",
    r"\bhigher\b",
    r"\boverexpress\w*\b",
    r"\benrich\w*\b",
    r"\bpositive\s+NES\b",
    r"\bNES\s*[>=]\s*0\b",
    r"\bupstream\s+activat\w*\b",
    r"\bstimulat\w*\b",
    r"\bpromot\w*\b",
]

DOWN_INDICATORS = [
    r"\bdownregulat\w*\b",
    r"\bsuppress\w*\b",
    r"\bdecreas\w*\b",
    r"\breduced?\b",
    r"\binhibit\w*\b",
    r"\brepress\w*\b",
    r"\blower\w*\b",
    r"\bunderexpress\w*\b",
    r"\bdepress\w*\b",
    r"\bnegative\s+NES\b",
    r"\bNES\s*<\s*0\b",
    r"\bdiminish\w*\b",
    r"\battenuati\w*\b",
    r"\bimpair\w*\b",
]

# Negation patterns that flip direction
NEGATION_PATTERNS = [
    r"\bnot\s+",
    r"\bno\s+",
    r"\bneither\b",
    r"\bwithout\s+",
    r"\bfail\w*\s+to\b",
    r"\bdoes\s+not\b",
    r"\bdid\s+not\b",
    r"\bisn'?t\b",
    r"\bwasn'?t\b",
    r"\baren'?t\b",
]

# ── Pathway name abbreviations ────────────────────────────────────────────
PATHWAY_ABBREVIATIONS: Dict[str, List[str]] = {
    "oxidative phosphorylation": ["oxphos", "oxidative phosphorylation", "ox phos"],
    "tnfa signaling via nfkb": ["tnf-alpha", "nfkb", "nf-kb", "nf-κb", "tnfα"],
    "mtorc1 signaling": ["mtor", "mtorc1"],
    "pi3k akt mtor signaling": ["pi3k", "akt", "mtor", "pi3k/akt"],
    "interferon gamma response": ["ifn-gamma", "ifn-γ", "interferon gamma", "ifnγ"],
    "interferon alpha response": ["ifn-alpha", "ifn-α", "interferon alpha", "ifnα"],
    "adipogenesis": ["adipogenesis", "adipogenic"],
    "myogenesis": ["myogenesis", "myogenic"],
    "epithelial mesenchymal transition": ["emt", "epithelial-mesenchymal"],
    "unfolded protein response": ["upr", "unfolded protein"],
    "reactive oxygen species pathway": ["ros", "reactive oxygen"],
    "fatty acid metabolism": ["fatty acid", "fat metabolism", "lipid metabolism"],
    "glycolysis": ["glycolysis", "glycolytic"],
    "dna repair": ["dna repair", "dna damage response"],
    "apoptosis": ["apoptosis", "apoptotic", "programmed cell death"],
    "inflammatory response": ["inflammatory", "inflammation"],
    "hypoxia": ["hypoxia", "hypoxic"],
    "angiogenesis": ["angiogenesis", "angiogenic"],
    "p53 pathway": ["p53", "tp53"],
    "wnt beta catenin signaling": ["wnt", "beta-catenin", "β-catenin"],
}


def _generate_pathway_variants(pathway_name: str) -> List[str]:
    """Generate matching variants for a pathway name.

    E.g. HALLMARK_OXIDATIVE_PHOSPHORYLATION →
         ["HALLMARK_OXIDATIVE_PHOSPHORYLATION",
          "oxidative phosphorylation",
          "oxidative phosphorylation pathway",
          "oxphos"]
    """
    variants = [pathway_name]

    clean = pathway_name
    for prefix in ("HALLMARK_", "KEGG_", "REACTOME_", "MITOCARTA_"):
        clean = clean.replace(prefix, "")
    human = clean.replace("_", " ").lower()

    variants.append(human)
    variants.append(human + " pathway")

    # Add known abbreviations
    for key, abbrevs in PATHWAY_ABBREVIATIONS.items():
        if key in human:
            variants.extend(abbrevs)

    return variants


def _extract_sentences_with_term(text: str, term: str) -> List[str]:
    """Extract sentences containing a term."""
    sentences = re.split(r"[.!?\n]+", text)
    return [
        s.strip()
        for s in sentences
        if term.lower() in s.lower() and len(s.strip()) > 10
    ]


def _has_negation_before(text: str, match_start: int, window: int = 12) -> bool:
    """Check if a negation word appears shortly before a match position.

    Window of ~12 chars catches "not " + up to ~8 chars of whitespace/adverbs,
    without reaching across clause boundaries like "not X but rather Y".
    """
    start = max(0, match_start - window)
    preceding = text[start:match_start].lower()
    return any(re.search(p, preceding) for p in NEGATION_PATTERNS)


def extract_direction_claims(
    text: str,
    pathway_name: str,
) -> List[Tuple[str, str]]:
    """Extract directional claims about a specific pathway from text.

    Returns list of (pathway_variant, direction) tuples.
    Direction is "UP", "DOWN", or "AMBIGUOUS".
    """
    text_lower = text.lower()
    pathway_variants = _generate_pathway_variants(pathway_name)

    claims: List[Tuple[str, str]] = []
    for variant in pathway_variants:
        if variant.lower() not in text_lower:
            continue

        sentences = _extract_sentences_with_term(text, variant)
        for sentence in sentences:
            sent_lower = sentence.lower()
            up_count = 0
            down_count = 0

            for pattern in UP_INDICATORS:
                for match in re.finditer(pattern, sent_lower):
                    if _has_negation_before(sent_lower, match.start()):
                        down_count += 1  # Negated up = down
                    else:
                        up_count += 1

            for pattern in DOWN_INDICATORS:
                for match in re.finditer(pattern, sent_lower):
                    if _has_negation_before(sent_lower, match.start()):
                        up_count += 1  # Negated down = up
                    else:
                        down_count += 1

            if up_count > down_count:
                claims.append((variant, "UP"))
            elif down_count > up_count:
                claims.append((variant, "DOWN"))
            elif up_count > 0:
                claims.append((variant, "AMBIGUOUS"))

    return claims


class PathwayDirectionVerifier(BaseVerifier):
    """V1: Verifies pathway direction claims against fGSEA NES data."""

    name = "V1"

    def score(
        self,
        prompt: str,
        completion: str,
        ground_truth: Dict,
        question_type: str,
    ) -> VerifierResult:
        if "pathway" not in ground_truth or "direction" not in ground_truth:
            return VerifierResult(
                score=0.5,
                verifier_name=self.name,
                details={"reason": "no_pathway_in_gt"},
                applicable=False,
            )

        expected_dir = ground_truth["direction"]
        pathway = ground_truth["pathway"]

        # For comparison questions, check all tissue directions
        if "tissue_directions" in ground_truth and question_type == "comparison":
            return self._score_comparison(completion, ground_truth)

        claims = extract_direction_claims(completion, pathway)

        if not claims:
            return VerifierResult(
                score=0.3,
                verifier_name=self.name,
                details={
                    "reason": "no_claim_extracted",
                    "pathway": pathway,
                    "expected": expected_dir,
                },
            )

        matching = [c for c in claims if c[1] == expected_dir]
        contradicting = [
            c for c in claims if c[1] != expected_dir and c[1] != "AMBIGUOUS"
        ]

        if matching and not contradicting:
            score = 1.0
        elif matching and contradicting:
            score = 0.5
        elif contradicting:
            score = 0.0
        else:
            score = 0.3  # Only ambiguous claims

        return VerifierResult(
            score=score,
            verifier_name=self.name,
            details={
                "pathway": pathway,
                "expected": expected_dir,
                "claims": [(v, d) for v, d in claims],
                "n_matching": len(matching),
                "n_contradicting": len(contradicting),
            },
        )

    def _score_comparison(
        self, completion: str, ground_truth: Dict
    ) -> VerifierResult:
        """Score cross-tissue comparison: check direction per tissue."""
        tissue_dirs = ground_truth.get("tissue_directions", {})
        pathway = ground_truth.get("pathway", "")

        if not tissue_dirs:
            return VerifierResult(
                score=0.5, verifier_name=self.name,
                details={"reason": "no_tissue_directions"}, applicable=False,
            )

        correct = 0
        checked = 0
        details_per_tissue = {}

        for tissue, expected_dir in tissue_dirs.items():
            # Look for tissue-specific claims in the response
            tissue_sentences = _extract_sentences_with_term(completion, tissue)
            if not tissue_sentences:
                continue

            tissue_text = " ".join(tissue_sentences)
            claims = extract_direction_claims(tissue_text, pathway)
            checked += 1

            if any(c[1] == expected_dir for c in claims):
                correct += 1
                details_per_tissue[tissue] = "correct"
            elif claims:
                details_per_tissue[tissue] = "wrong"
            else:
                details_per_tissue[tissue] = "no_claim"

        score = correct / checked if checked > 0 else 0.3

        return VerifierResult(
            score=score,
            verifier_name=self.name,
            details={
                "pathway": pathway,
                "tissues_checked": checked,
                "tissues_correct": correct,
                "per_tissue": details_per_tissue,
            },
        )
