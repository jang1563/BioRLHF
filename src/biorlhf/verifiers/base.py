"""Abstract base class for biological verifiers."""

from abc import ABC, abstractmethod
from typing import Dict, List
from dataclasses import dataclass, field


@dataclass
class VerifierResult:
    """Result from a single verifier."""
    score: float                    # 0.0 to 1.0
    verifier_name: str
    details: Dict = field(default_factory=dict)
    applicable: bool = True         # False if verifier doesn't apply


class BaseVerifier(ABC):
    """Abstract base class for biological verifiers.

    Each verifier scores a model completion against structured ground truth
    on a specific dimension (pathway direction, factual accuracy, etc.).
    """

    name: str = "base"

    @abstractmethod
    def score(
        self,
        prompt: str,
        completion: str,
        ground_truth: Dict,
        question_type: str,
    ) -> VerifierResult:
        """Score a single completion against ground truth.

        Args:
            prompt: The original question.
            completion: The model's generated response.
            ground_truth: Parsed ground truth dictionary.
            question_type: Type of question for routing logic.

        Returns:
            VerifierResult with score in [0, 1].
        """
        raise NotImplementedError

    def is_applicable(self, applicable_verifiers: List[str]) -> bool:
        """Check if this verifier should score this question."""
        return self.name in applicable_verifiers
