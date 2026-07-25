"""Reusable aggregation primitives for controlled robustness experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .metrics import add_baseline_improvements, evaluate_frame


def evaluate_robustness_level(
    frame: pd.DataFrame,
    predictions_by_algorithm: Mapping[str, Any],
    *,
    dimension: str,
    level: Any,
    scenario: str,
    seed: int,
    inference_times_ms: Mapping[str, float] | None = None,
) -> pd.DataFrame:
    """Evaluate all algorithms at one controlled robustness level."""

    if not predictions_by_algorithm:
        raise ValueError("predictions_by_algorithm cannot be empty.")
    rows: list[dict[str, Any]] = []
    for algorithm, predictions in predictions_by_algorithm.items():
        row = evaluate_frame(
            frame,
            predictions,
            algorithm=algorithm,
            scenario=scenario,
            seed=seed,
        )
        row["robustness_dimension"] = dimension
        row["robustness_level"] = level
        if inference_times_ms is not None and algorithm in inference_times_ms:
            row["inference_time_ms"] = float(inference_times_ms[algorithm])
        rows.append(row)
    return pd.DataFrame(rows)


def add_robustness_degradation(
    results: pd.DataFrame,
    *,
    metric: str = "mean_error",
    reference_level: Any | None = None,
    level_column: str = "robustness_level",
    group_columns: Sequence[str] = ("algorithm", "robustness_dimension", "seed"),
) -> pd.DataFrame:
    """Compare each robustness level with its clean/reference level."""

    required = {metric, level_column, *group_columns}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"Robustness table is missing columns: {sorted(missing)}")
    groups = list(group_columns)
    references: list[dict[str, Any]] = []
    for keys, subset in results.groupby(groups, dropna=False, sort=False):
        key_tuple = keys if isinstance(keys, tuple) else (keys,)
        if reference_level is None:
            numeric_levels = pd.to_numeric(subset[level_column], errors="coerce")
            if numeric_levels.notna().all():
                reference_index = numeric_levels.idxmin()
            else:
                reference_index = subset.index[0]
        else:
            matches = subset[level_column] == reference_level
            if not matches.any():
                raise ValueError(
                    f"No reference level {reference_level!r} for group {key_tuple!r}."
                )
            reference_index = subset.index[matches][0]
        reference = {column: value for column, value in zip(groups, key_tuple)}
        reference[f"{metric}_reference"] = float(
            results.loc[reference_index, metric]
        )
        references.append(reference)
    reference_frame = pd.DataFrame(references)
    augmented = results.merge(reference_frame, on=groups, how="left", validate="many_to_one")
    reference_column = f"{metric}_reference"
    augmented[f"{metric}_degradation"] = (
        augmented[metric] - augmented[reference_column]
    )
    denominator = augmented[reference_column].replace(0.0, np.nan)
    augmented[f"{metric}_degradation_percent"] = (
        100.0 * augmented[f"{metric}_degradation"] / denominator
    )
    return augmented


def aggregate_robustness(
    per_seed_results: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (
        "algorithm",
        "robustness_dimension",
        "robustness_level",
        "scenario",
    ),
    metric_columns: Sequence[str] = (
        "mean_error",
        "rmse",
        "median_error",
        "p90_error",
        "max_error",
        "los_mean_error",
        "nlos_mean_error",
        "inference_time_ms",
    ),
) -> pd.DataFrame:
    """Aggregate real per-seed robustness rows into mean/std summaries."""

    groups = [column for column in group_columns if column in per_seed_results]
    metrics = [column for column in metric_columns if column in per_seed_results]
    if not groups or not metrics:
        raise ValueError("Robustness results need grouping and metric columns.")
    grouped = per_seed_results.groupby(groups, dropna=False)[metrics].agg(["mean", "std"])
    grouped.columns = [
        f"{metric}_{statistic}" for metric, statistic in grouped.columns
    ]
    return grouped.reset_index()


def add_geometric_comparison(
    results: pd.DataFrame,
    *,
    baseline_algorithm: str = "Geometric LS",
) -> pd.DataFrame:
    """Add method-vs-geometric improvements within each robustness condition."""

    groups = [
        column
        for column in (
            "scenario",
            "seed",
            "robustness_dimension",
            "robustness_level",
        )
        if column in results
    ]
    return add_baseline_improvements(
        results,
        baseline_algorithm=baseline_algorithm,
        metric="mean_error",
        group_columns=groups,
    )
