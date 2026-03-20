"""
Calibration evaluation metrics for BioGRPO.

Implements Expected Calibration Error (ECE), Brier score, overconfidence
rate, and reliability diagram data generation.
"""

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class CalibrationMetrics:
    """Aggregated calibration metrics."""
    ece: float                      # Expected Calibration Error
    mce: float                      # Maximum Calibration Error
    brier_score: float
    overconfidence_rate: float      # P(wrong | confidence > threshold)
    underconfidence_rate: float     # P(correct | confidence < threshold)
    mean_confidence: float
    mean_accuracy: float
    n_samples: int
    reliability_bins: List[Dict]    # For plotting reliability diagrams


def compute_ece(
    confidences: List[float],
    correctnesses: List[bool],
    n_bins: int = 10,
) -> Tuple[float, float, List[Dict]]:
    """Compute Expected and Maximum Calibration Error.

    Uses equal-width binning.

    Args:
        confidences: Model's stated confidence for each prediction (0-1).
        correctnesses: Whether each prediction was correct.
        n_bins: Number of calibration bins.

    Returns:
        (ECE, MCE, bin_data) where bin_data is list of dicts for plotting.
    """
    if not confidences:
        return 0.0, 0.0, []

    bin_width = 1.0 / n_bins
    bins: List[Dict] = []

    for i in range(n_bins):
        bin_lower = i * bin_width
        bin_upper = (i + 1) * bin_width

        # Find samples in this bin
        indices = [
            j for j, c in enumerate(confidences)
            if bin_lower <= c < bin_upper or (i == n_bins - 1 and c == 1.0)
        ]

        if not indices:
            bins.append({
                "bin_lower": bin_lower,
                "bin_upper": bin_upper,
                "mean_confidence": (bin_lower + bin_upper) / 2,
                "mean_accuracy": 0.0,
                "count": 0,
                "calibration_error": 0.0,
            })
            continue

        bin_confs = [confidences[j] for j in indices]
        bin_accs = [float(correctnesses[j]) for j in indices]
        mean_conf = sum(bin_confs) / len(bin_confs)
        mean_acc = sum(bin_accs) / len(bin_accs)

        bins.append({
            "bin_lower": bin_lower,
            "bin_upper": bin_upper,
            "mean_confidence": mean_conf,
            "mean_accuracy": mean_acc,
            "count": len(indices),
            "calibration_error": abs(mean_acc - mean_conf),
        })

    # ECE: weighted average of calibration errors
    total_samples = len(confidences)
    ece = sum(
        b["count"] / total_samples * b["calibration_error"]
        for b in bins if b["count"] > 0
    )

    # MCE: maximum calibration error across non-empty bins
    non_empty_errors = [b["calibration_error"] for b in bins if b["count"] > 0]
    mce = max(non_empty_errors) if non_empty_errors else 0.0

    return ece, mce, bins


def compute_brier_score(
    confidences: List[float],
    correctnesses: List[bool],
) -> float:
    """Compute Brier score: mean squared error between confidence and outcome.

    Lower is better. Range [0, 1].
    """
    if not confidences:
        return 0.0
    n = len(confidences)
    return sum(
        (c - float(o)) ** 2 for c, o in zip(confidences, correctnesses)
    ) / n


def compute_overconfidence_rate(
    confidences: List[float],
    correctnesses: List[bool],
    threshold: float = 0.8,
) -> float:
    """P(wrong | confidence > threshold).

    High overconfidence rate indicates the model is unreliably confident.
    """
    high_conf = [
        (c, o) for c, o in zip(confidences, correctnesses) if c > threshold
    ]
    if not high_conf:
        return 0.0
    wrong = sum(1 for _, o in high_conf if not o)
    return wrong / len(high_conf)


def compute_underconfidence_rate(
    confidences: List[float],
    correctnesses: List[bool],
    threshold: float = 0.3,
) -> float:
    """P(correct | confidence < threshold).

    High underconfidence rate means the model knows more than it admits.
    """
    low_conf = [
        (c, o) for c, o in zip(confidences, correctnesses) if c < threshold
    ]
    if not low_conf:
        return 0.0
    correct = sum(1 for _, o in low_conf if o)
    return correct / len(low_conf)


def compute_calibration_metrics(
    confidences: List[float],
    correctnesses: List[bool],
    n_bins: int = 10,
    overconf_threshold: float = 0.8,
    underconf_threshold: float = 0.3,
) -> CalibrationMetrics:
    """Compute full calibration metrics suite."""
    if not confidences:
        return CalibrationMetrics(
            ece=0.0, mce=0.0, brier_score=0.0,
            overconfidence_rate=0.0, underconfidence_rate=0.0,
            mean_confidence=0.0, mean_accuracy=0.0,
            n_samples=0, reliability_bins=[],
        )

    ece, mce, bins = compute_ece(confidences, correctnesses, n_bins)
    brier = compute_brier_score(confidences, correctnesses)
    overconf = compute_overconfidence_rate(confidences, correctnesses, overconf_threshold)
    underconf = compute_underconfidence_rate(confidences, correctnesses, underconf_threshold)

    mean_conf = sum(confidences) / len(confidences)
    mean_acc = sum(float(c) for c in correctnesses) / len(correctnesses)

    return CalibrationMetrics(
        ece=ece,
        mce=mce,
        brier_score=brier,
        overconfidence_rate=overconf,
        underconfidence_rate=underconf,
        mean_confidence=mean_conf,
        mean_accuracy=mean_acc,
        n_samples=len(confidences),
        reliability_bins=bins,
    )
