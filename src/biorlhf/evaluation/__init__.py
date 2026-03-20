"""Evaluation modules for BioRLHF."""

__all__ = [
    "evaluate_model",
    "compute_metrics",
]


def __getattr__(name):
    """Lazy imports for torch-dependent modules."""
    if name in ("evaluate_model", "compute_metrics"):
        from biorlhf.evaluation.evaluate import evaluate_model, compute_metrics
        return {"evaluate_model": evaluate_model, "compute_metrics": compute_metrics}[name]
    raise AttributeError(f"module 'biorlhf.evaluation' has no attribute {name!r}")
