"""Residual-model feature and training-condition ablation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd

from localization_twin.models.features import (
    ABLATION_FEATURE_GROUPS,
    FULL_RESIDUAL_FEATURE_GROUPS,
)
from localization_twin.models.residual_ai import ResidualAILocator

from .metrics import evaluate_frame


@dataclass(frozen=True)
class AblationSpec:
    """Definition of one independently reproducible residual-model ablation."""

    name: str
    feature_groups: tuple[str, ...]
    training_variant: str = "default"


DEFAULT_ABLATION_SPECS: tuple[AblationSpec, ...] = (
    AblationSpec("full", FULL_RESIDUAL_FEATURE_GROUPS),
    AblationSpec(
        "without_los_nlos",
        ABLATION_FEATURE_GROUPS["without_los_nlos"],
    ),
    AblationSpec(
        "without_geometric_residual",
        ABLATION_FEATURE_GROUPS["without_geometric_residual"],
    ),
    AblationSpec(
        "without_anchor_mask",
        ABLATION_FEATURE_GROUPS["without_anchor_mask"],
    ),
    AblationSpec(
        "without_spatial_bias_training",
        FULL_RESIDUAL_FEATURE_GROUPS,
        training_variant="without_spatial_bias",
    ),
)


def ablation_specs(
    names: Sequence[str] | None = None,
) -> tuple[AblationSpec, ...]:
    """Resolve named ablations and reject silent misspellings."""

    if names is None:
        return DEFAULT_ABLATION_SPECS
    lookup = {spec.name: spec for spec in DEFAULT_ABLATION_SPECS}
    unknown = set(names) - set(lookup)
    if unknown:
        raise ValueError(f"Unknown ablation(s): {sorted(unknown)}")
    return tuple(lookup[name] for name in names)


def evaluate_residual_ablations(
    training_frames: Mapping[str, pd.DataFrame],
    training_geometric: Mapping[str, Any],
    test_frame: pd.DataFrame,
    test_geometric: Any,
    *,
    anchor_ids: Sequence[str],
    validation_frames: Mapping[str, pd.DataFrame] | None = None,
    validation_geometric: Mapping[str, Any] | None = None,
    specs: Sequence[AblationSpec] = DEFAULT_ABLATION_SPECS,
    model_kwargs: Mapping[str, Any] | None = None,
    scenario: str = "normal",
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, ResidualAILocator]]:
    """Train and evaluate genuine residual ablations.

    The spatial-bias ablation requires a separately generated
    ``without_spatial_bias`` training frame.  Missing variants raise an error
    instead of silently relabelling the default data.
    """

    rows: list[dict[str, Any]] = []
    predictions: dict[str, Any] = {}
    models: dict[str, ResidualAILocator] = {}
    kwargs = dict(model_kwargs or {})
    for spec in specs:
        if spec.training_variant not in training_frames:
            raise ValueError(
                f"Ablation {spec.name!r} requires training frame "
                f"{spec.training_variant!r}."
            )
        if spec.training_variant not in training_geometric:
            raise ValueError(
                f"Ablation {spec.name!r} requires geometric training output "
                f"{spec.training_variant!r}."
            )
        validation_frame = (
            validation_frames.get(spec.training_variant)
            if validation_frames is not None
            else None
        )
        validation_geometry = (
            validation_geometric.get(spec.training_variant)
            if validation_geometric is not None
            else None
        )
        model = ResidualAILocator(
            anchor_ids,
            feature_groups=spec.feature_groups,
            random_state=seed,
            **kwargs,
        )
        model.fit(
            training_frames[spec.training_variant],
            training_geometric[spec.training_variant],
            validation_frame=validation_frame,
            validation_geometric=validation_geometry,
        )
        prediction = model.predict(test_frame, test_geometric)
        row = evaluate_frame(
            test_frame,
            prediction,
            algorithm="Residual AI",
            scenario=scenario,
            seed=seed,
        )
        row["ablation"] = spec.name
        row["training_variant"] = spec.training_variant
        row["feature_groups"] = ",".join(spec.feature_groups)
        row["training_time_s"] = model.training_time_s_
        rows.append(row)
        predictions[spec.name] = prediction
        models[spec.name] = model
    return pd.DataFrame(rows), predictions, models


def add_full_model_delta(
    results: pd.DataFrame,
    *,
    metric: str = "mean_error",
) -> pd.DataFrame:
    """Add degradation relative to the full residual model."""

    if "ablation" not in results or metric not in results:
        raise ValueError("Ablation table needs ablation and metric columns.")
    group_columns = [
        column for column in ("scenario", "seed") if column in results.columns
    ]
    full = results.loc[
        results["ablation"] == "full", group_columns + [metric]
    ].rename(columns={metric: f"full_{metric}"})
    if full.empty:
        raise ValueError("Ablation table has no full-model row.")
    if group_columns:
        augmented = results.merge(
            full,
            on=group_columns,
            how="left",
            validate="many_to_one",
        )
    else:
        if len(full) != 1:
            raise ValueError("Expected one full-model row.")
        augmented = results.copy()
        augmented[f"full_{metric}"] = float(full.iloc[0][f"full_{metric}"])
    augmented[f"{metric}_degradation_vs_full"] = (
        augmented[metric] - augmented[f"full_{metric}"]
    )
    denominator = augmented[f"full_{metric}"].replace(0.0, pd.NA)
    augmented[f"{metric}_degradation_percent_vs_full"] = (
        100.0 * augmented[f"{metric}_degradation_vs_full"] / denominator
    )
    return augmented
