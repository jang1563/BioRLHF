"""
V2: Biological Fact Verifier.

Scores model responses based on overlap with known correct facts
from curated knowledge bases (SpaceOmicsBench, BioEval, GeneTuring).

Scoring: proportion of ground truth key facts found in the response.
"""

import re
import json
from typing import Dict, List

from biorlhf.verifiers.base import BaseVerifier, VerifierResult


def _extract_key_terms(text: str, min_length: int = 4, max_terms: int = 10) -> List[str]:
    """Extract important terms from a text string."""
    # Remove common stopwords and short words
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are", "was",
        "were", "been", "have", "has", "had", "will", "would", "could",
        "should", "may", "might", "can", "does", "did", "but", "not",
        "its", "also", "into", "than", "then", "when", "which", "what",
        "where", "who", "how", "all", "each", "every", "both", "more",
        "most", "other", "some", "such", "only", "same", "very", "just",
    }
    words = re.findall(r"\b[a-zA-Z0-9-]+\b", text)
    terms = [
        w for w in words
        if len(w) >= min_length and w.lower() not in stopwords
    ]
    return terms[:max_terms]


def _phrase_match(phrase: str, text: str) -> bool:
    """Check if a phrase (or its key terms) appears in text."""
    text_lower = text.lower()
    phrase_lower = phrase.lower()

    # Direct substring match
    if phrase_lower in text_lower:
        return True

    # For multi-word phrases, check if key terms co-occur
    terms = _extract_key_terms(phrase, min_length=4, max_terms=5)
    if not terms:
        return phrase_lower in text_lower

    matches = sum(1 for t in terms if t.lower() in text_lower)
    # Require majority of key terms to match
    return matches >= max(1, len(terms) // 2)


class BiologicalFactVerifier(BaseVerifier):
    """V2: Verifies biological factual claims against curated knowledge."""

    name = "V2"

    def score(
        self,
        prompt: str,
        completion: str,
        ground_truth: Dict,
        question_type: str,
    ) -> VerifierResult:
        """Score based on overlap with ground truth key facts.

        Handles multiple GT formats:
          - {"key_facts": ["fact1", "fact2", ...]}
          - {"ground_truth_key_facts": [...]}
          - {"expected_answer": "text"}
          - {"expected_reasoning": [...]}
          - {"correct_steps": [...]}
        """
        gt = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)

        # Extract key facts from various GT formats
        key_facts = self._extract_facts(gt)

        if not key_facts:
            return VerifierResult(
                score=0.5,
                verifier_name=self.name,
                details={"reason": "no_key_facts_in_gt"},
                applicable=False,
            )

        # Score: proportion of key facts found in completion
        matched_facts: List[str] = []
        for fact in key_facts:
            if isinstance(fact, str) and _phrase_match(fact, completion):
                matched_facts.append(fact)

        total = len(key_facts)
        matched = len(matched_facts)
        score = matched / total if total > 0 else 0.0

        return VerifierResult(
            score=score,
            verifier_name=self.name,
            details={
                "matched_facts": matched_facts,
                "total_facts": total,
                "matched_count": matched,
                "unmatched": [f for f in key_facts if f not in matched_facts],
            },
        )

    def _extract_facts(self, gt: Dict) -> List[str]:
        """Extract verifiable facts from ground truth dictionary."""
        facts: List[str] = []

        # Direct key facts lists
        for key in ("key_facts", "ground_truth_key_facts"):
            if key in gt and isinstance(gt[key], list):
                facts.extend(str(f) for f in gt[key] if f)

        # Expected reasoning points
        if "expected_reasoning" in gt and isinstance(gt["expected_reasoning"], list):
            facts.extend(str(f) for f in gt["expected_reasoning"] if f)

        # Single expected answer
        if "expected_answer" in gt and isinstance(gt["expected_answer"], str):
            facts.append(gt["expected_answer"])

        # Protocol steps (BioEval protoreason)
        if "correct_steps" in gt and isinstance(gt["correct_steps"], list):
            facts.extend(str(s) for s in gt["correct_steps"] if s)

        # NES conservation facts
        if "conservation_level" in gt:
            facts.append(gt["conservation_level"])

        # Deduplicate while preserving order
        seen = set()
        unique_facts = []
        for f in facts:
            if f not in seen:
                seen.add(f)
                unique_facts.append(f)

        return unique_facts
