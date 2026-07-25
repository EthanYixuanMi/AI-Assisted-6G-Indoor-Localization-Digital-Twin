"""Evaluation primitives for localization accuracy and robustness."""

from .ablation import (
    DEFAULT_ABLATION_SPECS,
    AblationSpec,
    ablation_specs,
    add_full_model_delta,
    evaluate_residual_ablations,
)
from .metrics import (
    add_baseline_improvements,
    baseline_improvement,
    compute_localization_metrics,
    euclidean_errors,
    evaluate_frame,
    evaluate_predictions,
    localization_metrics,
    los_nlos_metrics,
    measure_inference_time,
    measure_training_time,
    metrics_from_errors,
    model_size_mb,
    time_inference,
)
from .robustness import (
    add_geometric_comparison,
    add_robustness_degradation,
    aggregate_robustness,
    evaluate_robustness_level,
)

__all__ = [
    "DEFAULT_ABLATION_SPECS",
    "AblationSpec",
    "ablation_specs",
    "add_baseline_improvements",
    "add_full_model_delta",
    "add_geometric_comparison",
    "add_robustness_degradation",
    "aggregate_robustness",
    "baseline_improvement",
    "compute_localization_metrics",
    "euclidean_errors",
    "evaluate_frame",
    "evaluate_predictions",
    "evaluate_residual_ablations",
    "evaluate_robustness_level",
    "localization_metrics",
    "los_nlos_metrics",
    "measure_inference_time",
    "measure_training_time",
    "metrics_from_errors",
    "model_size_mb",
    "time_inference",
]
