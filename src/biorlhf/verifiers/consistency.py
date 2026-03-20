"""
V3: Cross-Context Consistency Verifier.

Scores whether the model appropriately distinguishes or generalizes
across biological contexts (tissues, species, doses, timepoints).

For comparison questions: checks tissue coverage + consistency assessment.
For context-dependent questions: checks nuance and hedging.
For BioAmbiguity tasks: checks context awareness using BioEval scoring logic.
"""

import json
import re
from typing import Dict, List

from biorlhf.verifiers.base import BaseVerifier, VerifierResult


# ── Indicator patterns ─────────────────────────────────────────────────────
CONSISTENCY_TERMS = [
    "consistent", "conserved", "similar across", "same direction",
    "reproducible", "concordant", "shared", "common response",
    "universal", "uniform",
]

SPECIFICITY_TERMS = [
    "tissue-specific", "differs", "different", "opposite", "varies",
    "divergent", "heterogeneous", "discordant", "unique to",
    "distinct", "context-dependent", "tissue-dependent",
]

NUANCE_INDICATORS = [
    "depends", "context", "varies", "mission-specific",
    "not consistent", "differs", "some missions", "mixed",
    "heterogeneous", "variable", "inconsistent",
]

HEDGING_INDICATORS = [
    "uncertain", "unclear", "difficult to generalize",
    "not enough evidence", "conflicting", "limited data",
    "preliminary", "tentative", "cannot be determined",
    "more research", "caution",
]


class CrossContextConsistencyVerifier(BaseVerifier):
    """V3: Scores context-appropriate reasoning."""

    name = "V3"

    def score(
        self,
        prompt: str,
        completion: str,
        ground_truth: Dict,
        question_type: str,
    ) -> VerifierResult:
        gt = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)

        if question_type == "comparison":
            return self._score_comparison(completion, gt)
        elif question_type in ("context_dependent", "uncertainty"):
            return self._score_context_dependent(completion, gt)
        elif "contexts" in gt:
            # BioEval BioAmbiguity format
            return self._score_bioambiguity(completion, gt)
        elif "tissue_directions" in gt:
            return self._score_comparison(completion, gt)
        else:
            return VerifierResult(
                score=0.5,
                verifier_name=self.name,
                details={"reason": "not_applicable"},
                applicable=False,
            )

    def _score_comparison(self, completion: str, gt: Dict) -> VerifierResult:
        """Score cross-tissue comparison questions."""
        tissue_directions = gt.get("tissue_directions", {})
        is_consistent = gt.get("is_consistent", False)
        comp_lower = completion.lower()

        # Check tissue coverage
        tissues_mentioned = sum(
            1 for tissue in tissue_directions if tissue.lower() in comp_lower
        )
        n_tissues = len(tissue_directions) if tissue_directions else 1
        tissue_coverage = tissues_mentioned / n_tissues

        # Check consistency/specificity assessment
        claims_consistent = any(t in comp_lower for t in CONSISTENCY_TERMS)
        claims_specific = any(t in comp_lower for t in SPECIFICITY_TERMS)

        consistency_correct = False
        if is_consistent:
            consistency_correct = claims_consistent
        else:
            consistency_correct = claims_specific

        score = 0.5 * tissue_coverage + 0.5 * (1.0 if consistency_correct else 0.0)

        return VerifierResult(
            score=score,
            verifier_name=self.name,
            details={
                "tissues_mentioned": tissues_mentioned,
                "total_tissues": n_tissues,
                "is_consistent_gt": is_consistent,
                "claims_consistent": claims_consistent,
                "claims_specific": claims_specific,
                "consistency_correct": consistency_correct,
            },
        )

    def _score_context_dependent(self, completion: str, gt: Dict) -> VerifierResult:
        """Score questions where answer should acknowledge context-dependence."""
        comp_lower = completion.lower()

        nuance_hits = sum(1 for t in NUANCE_INDICATORS if t in comp_lower)
        hedging_hits = sum(1 for t in HEDGING_INDICATORS if t in comp_lower)

        # Scale: having 2-3 indicators is ideal
        nuance_score = min(nuance_hits / 2.0, 1.0)
        hedging_score = min(hedging_hits / 2.0, 1.0)

        score = 0.6 * nuance_score + 0.4 * hedging_score

        return VerifierResult(
            score=min(score, 1.0),
            verifier_name=self.name,
            details={
                "nuance_hits": nuance_hits,
                "hedging_hits": hedging_hits,
                "nuance_score": nuance_score,
                "hedging_score": hedging_score,
            },
        )

    def _score_bioambiguity(self, completion: str, gt: Dict) -> VerifierResult:
        """Score BioEval BioAmbiguity tasks.

        GT format:
            {"contexts": {context_name: {"key_terms": [...], "role": "..."}},
             "distinction_key": "..."}
        """
        contexts = gt.get("contexts", {})
        distinction_key = gt.get("distinction_key", "")
        comp_lower = completion.lower()

        if not contexts:
            return VerifierResult(
                score=0.5, verifier_name=self.name,
                details={"reason": "no_contexts"}, applicable=False,
            )

        # Context awareness: % of key terms found across all contexts
        total_terms = 0
        found_terms = 0
        context_scores = {}

        for ctx_name, ctx_info in contexts.items():
            key_terms = ctx_info.get("key_terms", [])
            if not key_terms:
                continue
            hits = sum(1 for t in key_terms if t.lower() in comp_lower)
            total_terms += len(key_terms)
            found_terms += hits
            context_scores[ctx_name] = hits / len(key_terms) if key_terms else 0

        context_awareness = found_terms / total_terms if total_terms > 0 else 0

        # Distinction quality: does response contain distinction key words?
        if distinction_key:
            dist_terms = _extract_key_terms(distinction_key)
            dist_hits = sum(1 for t in dist_terms if t.lower() in comp_lower)
            distinction_quality = dist_hits / len(dist_terms) if dist_terms else 0
        else:
            distinction_quality = 0

        # Evidence support: does response mention roles?
        role_hits = 0
        role_total = 0
        for ctx_info in contexts.values():
            role = ctx_info.get("role", "")
            if role:
                role_total += 1
                role_terms = _extract_key_terms(role)
                if any(t.lower() in comp_lower for t in role_terms):
                    role_hits += 1
        evidence_support = role_hits / role_total if role_total > 0 else 0

        # Composite: 40% context + 35% distinction + 25% evidence
        score = (
            0.40 * context_awareness
            + 0.35 * distinction_quality
            + 0.25 * evidence_support
        )

        return VerifierResult(
            score=score,
            verifier_name=self.name,
            details={
                "context_awareness": context_awareness,
                "distinction_quality": distinction_quality,
                "evidence_support": evidence_support,
                "context_scores": context_scores,
                "terms_found": found_terms,
                "terms_total": total_terms,
            },
        )


def _extract_key_terms(text: str, min_length: int = 4) -> List[str]:
    """Extract key terms from text for matching."""
    stopwords = {
        "the", "and", "for", "that", "this", "with", "from", "are",
        "was", "were", "been", "have", "has", "had", "will", "would",
        "could", "should", "may", "might", "can", "does", "between",
    }
    words = re.findall(r"\b[a-zA-Z0-9-]+\b", text)
    return [w for w in words if len(w) >= min_length and w.lower() not in stopwords]
