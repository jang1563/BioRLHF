"""
Verifier Composer: Weighted composition of V1-V4 into a TRL-compatible reward function.

This is THE critical integration point between the verifier stack and
TRL's GRPOTrainer. The reward function signature must match TRL's expected
interface exactly.
"""

import json
from typing import Callable, Dict, List, Optional
from dataclasses import dataclass, field

from biorlhf.verifiers.base import BaseVerifier, VerifierResult
from biorlhf.verifiers.pathway import PathwayDirectionVerifier
from biorlhf.verifiers.factual import BiologicalFactVerifier
from biorlhf.verifiers.consistency import CrossContextConsistencyVerifier
from biorlhf.verifiers.uncertainty import UncertaintyVerifier


@dataclass
class ComposedReward:
    """Result of composed reward computation."""
    total_reward: float
    verifier_scores: Dict[str, float]
    verifier_details: Dict[str, Dict]
    weights_used: Dict[str, float]


# Default weights — factual signals dominate
DEFAULT_WEIGHTS = {
    "V1": 0.35,     # Pathway direction (hard signal)
    "V2": 0.30,     # Biological facts (soft signal)
    "V3": 0.15,     # Cross-context consistency
    "V4": 0.20,     # Uncertainty appropriateness
}


class VerifierComposer:
    """Composes V1-V4 verifiers into a unified reward signal."""

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        active_verifiers: Optional[List[str]] = None,
    ):
        all_verifiers: Dict[str, BaseVerifier] = {
            "V1": PathwayDirectionVerifier(),
            "V2": BiologicalFactVerifier(),
            "V3": CrossContextConsistencyVerifier(),
            "V4": UncertaintyVerifier(),
        }

        self.weights = dict(weights or DEFAULT_WEIGHTS)

        # Filter to active verifiers if specified
        if active_verifiers:
            self.verifiers = {
                k: v for k, v in all_verifiers.items() if k in active_verifiers
            }
            # Renormalize weights
            total_w = sum(self.weights.get(k, 0) for k in self.verifiers)
            if total_w > 0:
                self.weights = {
                    k: self.weights.get(k, 0) / total_w for k in self.verifiers
                }
        else:
            self.verifiers = all_verifiers

    def compute_reward(
        self,
        prompt: str,
        completion: str,
        ground_truth: str,
        question_type: str,
        applicable_verifiers: str,
    ) -> ComposedReward:
        """Compute composed reward from all applicable verifiers.

        Args:
            prompt: The question text.
            completion: Model's generated response.
            ground_truth: JSON string of ground truth.
            question_type: Question type for routing.
            applicable_verifiers: JSON list of verifier names.
        """
        gt = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
        applicable = (
            json.loads(applicable_verifiers)
            if isinstance(applicable_verifiers, str)
            else applicable_verifiers
        )

        scores: Dict[str, float] = {}
        details: Dict[str, Dict] = {}
        weights_used: Dict[str, float] = {}

        for vname, verifier in self.verifiers.items():
            if not verifier.is_applicable(applicable):
                continue

            result = verifier.score(prompt, completion, gt, question_type)

            if not result.applicable:
                continue

            scores[vname] = result.score
            details[vname] = result.details
            weights_used[vname] = self.weights.get(vname, 0)

        # Compute weighted sum with renormalization
        if not weights_used:
            return ComposedReward(
                total_reward=0.0,
                verifier_scores=scores,
                verifier_details=details,
                weights_used=weights_used,
            )

        w_total = sum(weights_used.values())
        if w_total > 0:
            normalized = {k: v / w_total for k, v in weights_used.items()}
        else:
            normalized = weights_used

        total = sum(scores[k] * normalized.get(k, 0) for k in scores)

        return ComposedReward(
            total_reward=total,
            verifier_scores=scores,
            verifier_details=details,
            weights_used=normalized,
        )


def make_grpo_reward_function(
    weights: Optional[Dict[str, float]] = None,
    active_verifiers: Optional[List[str]] = None,
) -> Callable:
    """Create a TRL-compatible reward function from the verifier composer.

    TRL's GRPOTrainer calls reward functions with signature:
        reward_func(completions, **kwargs) -> list[float]

    where kwargs include all dataset columns except "prompt".
    The completions are list of list of dicts in chat format, or list of strings.

    Note: TRL passes prompts separately. Dataset columns (ground_truth,
    question_type, applicable_verifiers, etc.) are forwarded as kwargs.
    """
    composer = VerifierComposer(weights=weights, active_verifiers=active_verifiers)

    def reward_func(
        completions: List,
        ground_truth: Optional[List[str]] = None,
        question_type: Optional[List[str]] = None,
        applicable_verifiers: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """TRL-compatible reward function using composed biological verifiers.

        Args:
            completions: List of model completions (strings or chat messages).
            ground_truth: List of JSON ground truth strings (from dataset).
            question_type: List of question type strings (from dataset).
            applicable_verifiers: List of JSON lists of verifier names.

        Returns:
            List of float rewards, one per completion.
        """
        rewards: List[float] = []
        n = len(completions)

        # Handle missing kwargs gracefully
        if ground_truth is None:
            ground_truth = ["{}"] * n
        if question_type is None:
            question_type = ["unknown"] * n
        if applicable_verifiers is None:
            applicable_verifiers = [json.dumps(["V1", "V2", "V3", "V4"])] * n

        # Extract prompts if available in kwargs
        prompts = kwargs.get("prompts", kwargs.get("prompt", [""] * n))
        if isinstance(prompts, str):
            prompts = [prompts] * n

        for i in range(n):
            # Extract completion text
            completion_text = _extract_text(completions[i])
            prompt_text = _extract_text(prompts[i]) if i < len(prompts) else ""

            result = composer.compute_reward(
                prompt=prompt_text,
                completion=completion_text,
                ground_truth=ground_truth[i],
                question_type=question_type[i],
                applicable_verifiers=applicable_verifiers[i],
            )
            rewards.append(result.total_reward)

        return rewards

    return reward_func


def make_individual_reward_functions(
    active_verifiers: Optional[List[str]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> tuple:
    """Return (list_of_reward_funcs, list_of_weights) for TRL multi-reward.

    Each reward function wraps a single verifier and returns List[float | None].
    Non-applicable verifiers return None for samples where they don't apply;
    TRL natively excludes None rewards from the GRPO calculation.

    This enables per-verifier reward normalization in TRL, preventing a single
    low-variance verifier from dominating the gradient signal.
    """
    all_verifiers = {
        "V1": PathwayDirectionVerifier(),
        "V2": BiologicalFactVerifier(),
        "V3": CrossContextConsistencyVerifier(),
        "V4": UncertaintyVerifier(),
    }

    if active_verifiers:
        verifiers = {k: v for k, v in all_verifiers.items() if k in active_verifiers}
    else:
        verifiers = all_verifiers

    w = dict(weights or DEFAULT_WEIGHTS)
    weight_list = [w.get(k, 0) for k in verifiers]

    def _make_single_reward_fn(verifier: BaseVerifier, vname: str) -> Callable:
        """Create a closure-safe reward function for a single verifier."""

        def reward_func(
            completions: List,
            ground_truth: Optional[List[str]] = None,
            question_type: Optional[List[str]] = None,
            applicable_verifiers: Optional[List[str]] = None,
            **kwargs,
        ) -> List:
            n = len(completions)
            if ground_truth is None:
                ground_truth = ["{}"] * n
            if question_type is None:
                question_type = ["unknown"] * n
            if applicable_verifiers is None:
                applicable_verifiers = [json.dumps(list(all_verifiers.keys()))] * n

            prompts = kwargs.get("prompts", kwargs.get("prompt", [""] * n))
            if isinstance(prompts, str):
                prompts = [prompts] * n

            rewards = []
            for i in range(n):
                app = (
                    json.loads(applicable_verifiers[i])
                    if isinstance(applicable_verifiers[i], str)
                    else applicable_verifiers[i]
                )
                if not verifier.is_applicable(app):
                    rewards.append(None)
                    continue

                completion_text = _extract_text(completions[i])
                prompt_text = _extract_text(prompts[i]) if i < len(prompts) else ""
                gt = (
                    json.loads(ground_truth[i])
                    if isinstance(ground_truth[i], str)
                    else ground_truth[i]
                )

                result = verifier.score(prompt_text, completion_text, gt, question_type[i])
                if not result.applicable:
                    rewards.append(None)
                else:
                    rewards.append(result.score)

            return rewards

        reward_func.__name__ = f"reward_{vname}"
        return reward_func

    reward_funcs = [
        _make_single_reward_fn(v, name) for name, v in verifiers.items()
    ]

    return reward_funcs, weight_list


def make_single_verifier_reward(verifier_name: str) -> Callable:
    """Create a reward function using only one verifier (for ablation)."""
    return make_grpo_reward_function(active_verifiers=[verifier_name])


def _extract_text(item) -> str:
    """Extract plain text from various completion formats.

    TRL may pass completions as:
      - str: plain text
      - list[dict]: chat messages [{"role": "assistant", "content": "..."}]
    """
    if isinstance(item, str):
        return item
    elif isinstance(item, list):
        # Chat format
        texts = []
        for msg in item:
            if isinstance(msg, dict) and "content" in msg:
                texts.append(msg["content"])
        return " ".join(texts)
    else:
        return str(item)
