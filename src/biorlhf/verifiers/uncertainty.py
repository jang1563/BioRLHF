"""
V4: Uncertainty Appropriateness Verifier.

Scores whether a model's stated confidence aligns with the ground-truth
expected confidence level. Integrates with BioEval's calibration scoring
when available, with a built-in fallback.

Scoring dimensions:
  - Confidence level alignment (stated vs. expected)
  - Calibration task behavior (acknowledge_unknown, overconfidence_trap, etc.)
  - Default: penalizes extreme overconfidence
"""

import os
import re
import json
from typing import Dict, List
from dataclasses import dataclass

from biorlhf.verifiers.base import BaseVerifier, VerifierResult

# ── Try importing BioEval calibration infrastructure ──────────────────────
try:
    import os
    import sys
    _bioeval_root = os.environ.get(
        "BIOEVAL_ROOT",
        "/Users/jak4013/Dropbox/Bioinformatics/Claude/Evaluation_model/BioEval",
    )
    sys.path.insert(0, _bioeval_root)
    from bioeval.scoring.calibration import extract_confidence, ConfidenceExtraction
    HAS_BIOEVAL = True
except ImportError:
    HAS_BIOEVAL = False

# ── Built-in confidence extraction (fallback) ─────────────────────────────
HIGH_CONFIDENCE_PATTERNS = [
    r"\bhigh\s*confidence\b", r"\bvery\s+confident\b", r"\bconfident\s+that\b",
    r"\bcertainly\b", r"\bclearly\b", r"\bdefinitely\b", r"\bwithout\s+doubt\b",
    r"\bconfidence:\s*high\b", r"\bstrongly\s+suggest\b",
]

MEDIUM_CONFIDENCE_PATTERNS = [
    r"\bmoderate\s*confidence\b", r"\breasonably\s+confident\b",
    r"\blikely\b", r"\bprobably\b", r"\bsuggest\w*\b",
    r"\bconfidence:\s*medium\b", r"\bconfidence:\s*moderate\b",
]

LOW_CONFIDENCE_PATTERNS = [
    r"\blow\s*confidence\b", r"\bnot\s+confident\b", r"\buncertain\b",
    r"\bunclear\b", r"\bnot\s+sure\b", r"\bdon'?t\s+know\b",
    r"\bcannot\s+determine\b", r"\binsufficient\s+\w*\s*(?:data|evidence)\b",
    r"\blimited\s+evidence\b", r"\bspeculat\w*\b",
    r"\bconfidence:\s*low\b",
]

# Explicit numeric confidence
NUMERIC_CONFIDENCE_RE = re.compile(
    r"(?:confidence|certainty|probability)[:\s]*(\d{1,3})%",
    re.IGNORECASE,
)

# Expected confidence ranges
CONFIDENCE_RANGES = {
    "high": (0.70, 1.00),
    "medium": (0.35, 0.75),
    "low": (0.00, 0.40),
}

# Expected confidence for calibration task behaviors
BEHAVIOR_EXPECTED_CONFIDENCE = {
    "acknowledge_unknown": 0.15,
    "high_confidence_correct": 0.90,
    "partial_knowledge": 0.50,
    "context_dependent": 0.50,
    "moderate_confidence": 0.50,
    "overconfidence_trap": 0.30,
}


@dataclass
class SimpleConfidence:
    """Fallback confidence extraction result."""
    stated: str             # "high", "medium", "low"
    numeric: float          # 0.0 to 1.0
    source: str             # "explicit", "pattern", "language"


def _extract_confidence_simple(text: str) -> SimpleConfidence:
    """Simple confidence extraction without BioEval."""
    text_lower = text.lower()

    # Check for explicit numeric confidence
    num_match = NUMERIC_CONFIDENCE_RE.search(text)
    if num_match:
        pct = int(num_match.group(1))
        numeric = pct / 100.0
        if numeric >= 0.70:
            stated = "high"
        elif numeric >= 0.40:
            stated = "medium"
        else:
            stated = "low"
        return SimpleConfidence(stated=stated, numeric=numeric, source="explicit")

    # Count pattern matches
    high_count = sum(1 for p in HIGH_CONFIDENCE_PATTERNS if re.search(p, text_lower))
    med_count = sum(1 for p in MEDIUM_CONFIDENCE_PATTERNS if re.search(p, text_lower))
    low_count = sum(1 for p in LOW_CONFIDENCE_PATTERNS if re.search(p, text_lower))

    if low_count > high_count and low_count > med_count:
        return SimpleConfidence(stated="low", numeric=0.25, source="pattern")
    elif high_count > low_count and high_count > med_count:
        return SimpleConfidence(stated="high", numeric=0.85, source="pattern")
    elif med_count > 0:
        return SimpleConfidence(stated="medium", numeric=0.55, source="pattern")
    else:
        # Default: assume moderate confidence
        return SimpleConfidence(stated="medium", numeric=0.50, source="language")


class UncertaintyVerifier(BaseVerifier):
    """V4: Verifies that model's confidence is appropriate for the question.

    In calibration-aware mode (Phase 4), V4 internally uses V1 to determine
    whether the model's answer is factually correct, then sets the confidence
    target accordingly: high confidence for correct answers, low for incorrect.
    """

    name = "V4"

    def __init__(self):
        from biorlhf.verifiers.pathway import PathwayDirectionVerifier
        self._v1 = PathwayDirectionVerifier()
        # Controls fallback/default V4 shaping for non-calibration tasks.
        # - legacy: reward moderate confidence near 0.5
        # - match_v1: reward confidence that matches factual correctness
        self._default_mode = os.environ.get("BIORLHF_V4_DEFAULT_MODE", "match_v1").strip().lower()

    def score(
        self,
        prompt: str,
        completion: str,
        ground_truth: Dict,
        question_type: str,
    ) -> VerifierResult:
        gt = ground_truth if isinstance(ground_truth, dict) else json.loads(ground_truth)

        expected_confidence = gt.get("expected_confidence")
        correct_behavior = gt.get("correct_behavior")

        # Extract confidence from completion
        if HAS_BIOEVAL:
            conf_extraction = extract_confidence(completion)
            conf_score = conf_extraction.confidence_score
            stated = conf_extraction.stated_confidence or "medium"
        else:
            simple = _extract_confidence_simple(completion)
            conf_score = simple.numeric
            stated = simple.stated

        # Compute V1 score when direction ground truth is available.
        # This allows V4 to align confidence with factual correctness.
        v1_score = None
        if gt.get("direction"):
            try:
                v1_result = self._v1.score(prompt, completion, gt, question_type)
                if v1_result.applicable:
                    v1_score = v1_result.score
            except Exception:
                pass

        # Route to appropriate scoring
        if correct_behavior:
            return self._score_calibration_behavior(
                completion, gt, conf_score, stated, correct_behavior,
            )
        elif expected_confidence and v1_score is not None:
            return self._score_calibration_aware(
                conf_score, stated, expected_confidence, v1_score,
            )
        elif expected_confidence:
            return self._score_confidence_alignment(
                conf_score, stated, expected_confidence,
            )
        else:
            return self._score_default(
                conf_score,
                stated,
                v1_correct=(v1_score > 0.5) if v1_score is not None else None,
            )

    def _score_calibration_behavior(
        self,
        completion: str,
        gt: Dict,
        conf_score: float,
        stated: str,
        correct_behavior: str,
    ) -> VerifierResult:
        """Score calibration tasks with specific expected behaviors."""
        expected_conf = BEHAVIOR_EXPECTED_CONFIDENCE.get(correct_behavior, 0.5)
        conf_error = abs(conf_score - expected_conf)

        # Behavior-specific checks
        behavior_correct = False

        if correct_behavior == "acknowledge_unknown":
            # Model should say it doesn't know
            unknown_markers = [
                "don't know", "cannot determine", "not enough",
                "unknown", "hypothetical", "no information",
                "unable to", "cannot assess", "fictional",
            ]
            behavior_correct = any(m in completion.lower() for m in unknown_markers)

        elif correct_behavior == "high_confidence_correct":
            behavior_correct = conf_score >= 0.65

        elif correct_behavior == "overconfidence_trap":
            # Should NOT be overconfident
            behavior_correct = conf_score < 0.70

        elif correct_behavior in ("partial_knowledge", "context_dependent", "moderate_confidence"):
            behavior_correct = 0.30 <= conf_score <= 0.75

        # Combined score: behavior compliance + confidence alignment
        behavior_score = 1.0 if behavior_correct else 0.0
        alignment_score = max(0, 1.0 - 2.0 * conf_error)
        score = 0.6 * behavior_score + 0.4 * alignment_score

        return VerifierResult(
            score=score,
            verifier_name=self.name,
            details={
                "correct_behavior": correct_behavior,
                "expected_confidence": expected_conf,
                "actual_confidence": conf_score,
                "stated_confidence": stated,
                "confidence_error": conf_error,
                "behavior_correct": behavior_correct,
                "using_bioeval": HAS_BIOEVAL,
            },
        )

    def _score_calibration_aware(
        self,
        conf_score: float,
        stated: str,
        expected_confidence: str,
        v1_score: float,
    ) -> VerifierResult:
        """Score confidence alignment using V1 correctness as calibration target.

        For direction questions, sets the confidence target based on whether the
        model actually got the direction right (V1 > 0.5) or wrong (V1 <= 0.5).
        This creates a gradient signal: "be confident when right, uncertain when wrong."
        """
        v1_correct = v1_score > 0.5

        target_conf = 1.0 if v1_correct else 0.0
        conf_error = abs(conf_score - target_conf)
        score = max(0.2, 1.0 - conf_error * 2.0)

        return VerifierResult(
            score=score,
            verifier_name=self.name,
            details={
                "mode": "calibration_aware",
                "v1_score": v1_score,
                "v1_correct": v1_correct,
                "target_confidence": target_conf,
                "actual_confidence": conf_score,
                "stated_confidence": stated,
                "confidence_error": conf_error,
                "expected_level": expected_confidence,
                "using_bioeval": HAS_BIOEVAL,
            },
        )

    def _score_confidence_alignment(
        self,
        conf_score: float,
        stated: str,
        expected: str,
    ) -> VerifierResult:
        """Score how well stated confidence aligns with expected level."""
        if expected not in CONFIDENCE_RANGES:
            return VerifierResult(
                score=0.5, verifier_name=self.name,
                details={"reason": "unknown_expected_level"},
            )

        low, high = CONFIDENCE_RANGES[expected]
        in_range = low <= conf_score <= high

        if in_range:
            score = 1.0
        else:
            distance = min(abs(conf_score - low), abs(conf_score - high))
            score = max(0, 1.0 - 2.5 * distance)

        return VerifierResult(
            score=score,
            verifier_name=self.name,
            details={
                "expected_level": expected,
                "expected_range": (low, high),
                "actual_confidence": conf_score,
                "stated_confidence": stated,
                "in_range": in_range,
                "using_bioeval": HAS_BIOEVAL,
            },
        )

    def _score_default(
        self,
        conf_score: float,
        stated: str,
        v1_correct: bool = None,
    ) -> VerifierResult:
        """Default scoring for samples without explicit calibration labels.

        In `match_v1` mode (default), if V1 correctness is available, reward
        confidence that matches correctness:
          - correct answers: high confidence
          - incorrect answers: low confidence
        In `legacy` mode (or without V1), fall back to moderate-confidence prior.
        """
        mode_used = self._default_mode
        if self._default_mode == "match_v1" and v1_correct is not None:
            target_conf = 1.0 if v1_correct else 0.0
            score = max(0.2, 1.0 - abs(conf_score - target_conf) * 2.0)
        else:
            mode_used = "legacy"
            score = max(0.2, 1.0 - abs(conf_score - 0.5) * 1.5)

        return VerifierResult(
            score=score,
            verifier_name=self.name,
            details={
                "actual_confidence": conf_score,
                "stated_confidence": stated,
                "mode": mode_used,
                "v1_correct": v1_correct,
                "using_bioeval": HAS_BIOEVAL,
            },
        )
