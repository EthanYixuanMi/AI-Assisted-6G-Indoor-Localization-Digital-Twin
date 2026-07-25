"""Accuracy and runtime metrics for two-dimensional localization."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd


_METRIC_COLUMNS = (
    "mean_error",
    "rmse",
    "median_error",
    "p90_error",
    "max_error",
)


def _coordinate_array(values: Any, *, truth: bool) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        candidates = (
            (("true_x", "true_y"),)
            if truth
            else (
                ("pred_x", "pred_y"),
                ("estimated_x", "estimated_y"),
                ("geometric_x", "geometric_y"),
                ("x", "y"),
            )
        )
        for pair in candidates:
            if pair[0] in values and pair[1] in values:
                array = values.loc[:, list(pair)].to_numpy(dtype=float)
                break
        else:
            role = "truth" if truth else "prediction"
            raise ValueError(f"Could not find coordinate columns in {role} data frame.")
    else:
        array = np.asarray(values, dtype=float)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Coordinates must have shape (n_samples, 2), got {array.shape}.")
    if np.any(~np.isfinite(array)):
        raise ValueError("Coordinates contain NaN or infinite values.")
    return array


def euclidean_errors(y_true: Any, y_pred: Any) -> np.ndarray:
    """Return per-sample 2-D Euclidean localization error."""

    truth = _coordinate_array(y_true, truth=True)
    prediction = _coordinate_array(y_pred, truth=False)
    if truth.shape != prediction.shape:
        raise ValueError(
            f"Truth and prediction shapes differ: {truth.shape} != {prediction.shape}."
        )
    return np.linalg.norm(prediction - truth, axis=1)


def metrics_from_errors(errors: Any) -> dict[str, float | int]:
    """Summarize a one-dimensional vector of Euclidean errors.

    RMSE is exactly ``sqrt(mean(e_i**2))``; it is not coordinate-wise RMSE.
    Empty strata return a zero count and NaN metric values.
    """

    values = np.asarray(errors, dtype=float)
    if values.ndim != 1:
        raise ValueError("errors must be a one-dimensional vector.")
    if np.any(~np.isfinite(values)):
        raise ValueError("errors contain NaN or infinite values.")
    if len(values) == 0:
        return {
            "sample_count": 0,
            **{column: float("nan") for column in _METRIC_COLUMNS},
        }
    return {
        "sample_count": int(len(values)),
        "mean_error": float(np.mean(values)),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "median_error": float(np.median(values)),
        "p90_error": float(np.percentile(values, 90)),
        "max_error": float(np.max(values)),
    }


def localization_metrics(y_true: Any, y_pred: Any) -> dict[str, float | int]:
    """Compute the required overall localization metrics."""

    return metrics_from_errors(euclidean_errors(y_true, y_pred))


def _nlos_mask(values: Any, expected_length: int) -> np.ndarray:
    if isinstance(values, pd.DataFrame):
        if "nlos_anchor_count" not in values:
            raise ValueError("Data frame needs nlos_anchor_count for stratification.")
        mask = (
            pd.to_numeric(values["nlos_anchor_count"], errors="coerce")
            .fillna(0.0)
            .to_numpy()
            > 0
        )
    elif isinstance(values, pd.Series):
        numeric = pd.to_numeric(values, errors="coerce").fillna(0.0).to_numpy()
        mask = numeric > 0
    else:
        array = np.asarray(values)
        if array.dtype == bool:
            mask = array
        else:
            mask = np.asarray(array, dtype=float) > 0
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (expected_length,):
        raise ValueError(
            f"NLoS mask must have length {expected_length}, got {mask.shape}."
        )
    return mask


def los_nlos_metrics(
    y_true: Any,
    y_pred: Any,
    nlos: Any,
) -> dict[str, float | int]:
    """Compute metrics separately for all-LoS and at-least-one-NLoS samples."""

    errors = euclidean_errors(y_true, y_pred)
    nlos_mask = _nlos_mask(nlos, len(errors))
    result: dict[str, float | int] = {}
    for prefix, selected in (("los", ~nlos_mask), ("nlos", nlos_mask)):
        summary = metrics_from_errors(errors[selected])
        result[f"{prefix}_sample_count"] = summary["sample_count"]
        for key in _METRIC_COLUMNS:
            result[f"{prefix}_{key}"] = summary[key]

    los_mean = float(result["los_mean_error"])
    nlos_mean = float(result["nlos_mean_error"])
    if np.isfinite(los_mean) and np.isfinite(nlos_mean):
        result["nlos_los_mean_degradation"] = nlos_mean - los_mean
        result["nlos_los_mean_degradation_percent"] = (
            100.0 * (nlos_mean - los_mean) / los_mean
            if los_mean != 0.0
            else float("nan")
        )
    else:
        result["nlos_los_mean_degradation"] = float("nan")
        result["nlos_los_mean_degradation_percent"] = float("nan")
    return result


def evaluate_predictions(
    y_true: Any,
    y_pred: Any,
    *,
    nlos: Any | None = None,
) -> dict[str, float | int]:
    """Return overall metrics and, when supplied, LoS/NLoS metrics."""

    result = localization_metrics(y_true, y_pred)
    if nlos is not None:
        result.update(los_nlos_metrics(y_true, y_pred, nlos))
    return result


def evaluate_frame(
    frame: pd.DataFrame,
    predictions: Any,
    *,
    algorithm: str | None = None,
    scenario: str | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate predictions against a standard simulator data frame."""

    result: dict[str, Any] = evaluate_predictions(
        frame.loc[:, ["true_x", "true_y"]],
        predictions,
        nlos=frame,
    )
    if algorithm is not None:
        result["algorithm"] = algorithm
    if scenario is not None:
        result["scenario"] = scenario
    if seed is not None:
        result["seed"] = int(seed)
    return result


def baseline_improvement(
    method_error: float,
    baseline_error: float,
) -> dict[str, float]:
    """Compute signed change and improvement relative to a baseline.

    ``error_change`` is method minus baseline (negative is better), whereas
    ``absolute_improvement`` and ``percent_improvement`` are positive when the
    method improves on the baseline.
    """

    method = float(method_error)
    baseline = float(baseline_error)
    change = method - baseline
    improvement = baseline - method
    percent = (
        100.0 * improvement / baseline
        if baseline != 0.0 and np.isfinite(baseline)
        else float("nan")
    )
    return {
        "error_change": change,
        "absolute_improvement": improvement,
        "percent_improvement": percent,
    }


def add_baseline_improvements(
    results: pd.DataFrame,
    *,
    baseline_algorithm: str = "Geometric LS",
    metric: str = "mean_error",
    group_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add signed baseline-comparison columns to a result table."""

    required = {"algorithm", metric}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Result table is missing columns: {sorted(missing)}")
    if group_columns is None:
        group_columns = [
            column for column in ("scenario", "seed") if column in results.columns
        ]
    groups = list(group_columns)
    baseline_rows = results.loc[
        results["algorithm"] == baseline_algorithm, groups + [metric]
    ].copy()
    if baseline_rows.empty:
        raise ValueError(f"No baseline rows found for algorithm {baseline_algorithm!r}.")
    if groups and baseline_rows.duplicated(groups).any():
        raise ValueError("Multiple baseline rows exist for the same comparison group.")
    if not groups and len(baseline_rows) != 1:
        raise ValueError("Expected exactly one baseline row without grouping columns.")
    baseline_column = f"{metric}_geometric_baseline"
    baseline_rows = baseline_rows.rename(columns={metric: baseline_column})
    if groups:
        augmented = results.merge(
            baseline_rows,
            how="left",
            on=groups,
            validate="many_to_one",
        )
    else:
        augmented = results.copy()
        augmented[baseline_column] = float(baseline_rows.iloc[0][baseline_column])
    if augmented[baseline_column].isna().any():
        raise ValueError("At least one comparison group has no geometric baseline.")
    augmented[f"{metric}_change_vs_geometric"] = (
        augmented[metric] - augmented[baseline_column]
    )
    augmented[f"{metric}_absolute_improvement"] = (
        augmented[baseline_column] - augmented[metric]
    )
    denominator = augmented[baseline_column].replace(0.0, np.nan)
    augmented[f"{metric}_percent_improvement"] = (
        100.0
        * augmented[f"{metric}_absolute_improvement"]
        / denominator
    )
    # Stable generic names are convenient for the main mean-error table.
    if metric == "mean_error":
        augmented["absolute_error_change"] = augmented[
            f"{metric}_change_vs_geometric"
        ]
        augmented["absolute_improvement"] = augmented[
            f"{metric}_absolute_improvement"
        ]
        augmented["percent_improvement"] = augmented[
            f"{metric}_percent_improvement"
        ]
    return augmented


def measure_inference_time(
    predictor: Any,
    data: Any,
    *,
    warmup_runs: int = 2,
    repeats: int = 5,
    sample_count: int | None = None,
) -> dict[str, float | int]:
    """Measure warmed batch and per-sample inference latency.

    ``predictor`` can be a callable or an object exposing ``predict``.  The
    median repeat is reported to reduce sensitivity to background scheduling.
    """

    if warmup_runs < 1:
        raise ValueError("warmup_runs must be at least one.")
    if repeats < 1:
        raise ValueError("repeats must be at least one.")
    predict: Callable[[Any], Any]
    if hasattr(predictor, "predict"):
        predict = predictor.predict
    elif callable(predictor):
        predict = predictor
    else:
        raise TypeError("predictor must be callable or expose predict().")
    count = int(sample_count if sample_count is not None else len(data))
    if count <= 0:
        raise ValueError("sample_count must be positive.")

    for _ in range(warmup_runs):
        predict(data)
    elapsed: list[float] = []
    for _ in range(repeats):
        started = perf_counter()
        predict(data)
        elapsed.append(perf_counter() - started)
    batch_ms = float(np.median(elapsed) * 1000.0)
    return {
        "sample_count": count,
        "warmup_runs": int(warmup_runs),
        "timing_repeats": int(repeats),
        "batch_inference_time_ms": batch_ms,
        "inference_time_ms": batch_ms / count,
    }


def measure_training_time(
    fit_callable: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> tuple[Any, float]:
    """Run a fit callable and return ``(result, elapsed_seconds)``."""

    started = perf_counter()
    result = fit_callable(*args, **kwargs)
    return result, perf_counter() - started


def model_size_mb(path: str | Path) -> float:
    """Return serialized model size in binary megabytes."""

    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(model_path)
    return model_path.stat().st_size / (1024.0 * 1024.0)


compute_localization_metrics = localization_metrics
time_inference = measure_inference_time
