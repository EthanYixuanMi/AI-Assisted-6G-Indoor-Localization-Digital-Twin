"""Leakage-aware dataset splitting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class SpatialHoldoutSpec:
    """One-axis holdout region separated from development data by a guard band."""

    axis: str = "x"
    threshold: float = 23.0
    buffer: float = 1.0
    side: str = "high"

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "SpatialHoldoutSpec":
        """Parse a full, sampling, or holdout-only configuration."""

        section: Any = config
        if "sampling" in section:
            section = section["sampling"]
        if "spatial_holdout" in section:
            section = section["spatial_holdout"]
        if not isinstance(section, Mapping):
            raise ValueError("spatial_holdout configuration must be a mapping.")
        spec = cls(
            axis=str(section.get("axis", "x")).lower(),
            threshold=float(section.get("threshold", 23.0)),
            buffer=float(section.get("buffer", 1.0)),
            side=str(section.get("side", "high")).lower(),
        )
        if spec.axis not in {"x", "y"}:
            raise ValueError("Spatial holdout axis must be 'x' or 'y'.")
        if spec.side not in {"high", "low"}:
            raise ValueError("Spatial holdout side must be 'high' or 'low'.")
        if spec.buffer < 0.0:
            raise ValueError("Spatial holdout buffer cannot be negative.")
        return spec


def spatial_holdout_masks(
    positions: np.ndarray, spec: SpatialHoldoutSpec | Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return development, holdout, and guard-band masks.

    No coordinate can belong to both development and holdout. Their closest
    possible separation along the selected axis is ``2 * spec.buffer``.
    """

    parsed = spec if isinstance(spec, SpatialHoldoutSpec) else SpatialHoldoutSpec.from_config(spec)
    points = np.asarray(positions, dtype=float)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("positions must have shape (n_samples, 2).")
    values = points[:, 0 if parsed.axis == "x" else 1]
    if parsed.side == "high":
        development = values <= parsed.threshold - parsed.buffer
        holdout = values >= parsed.threshold + parsed.buffer
    else:
        development = values >= parsed.threshold + parsed.buffer
        holdout = values <= parsed.threshold - parsed.buffer
    guard = ~(development | holdout)
    return development, holdout, guard


def make_spatial_predicates(
    spec: SpatialHoldoutSpec | Mapping[str, Any],
) -> tuple[Callable[[np.ndarray], bool], Callable[[np.ndarray], bool]]:
    """Create scalar development and holdout predicates for samplers."""

    parsed = spec if isinstance(spec, SpatialHoldoutSpec) else SpatialHoldoutSpec.from_config(spec)
    index = 0 if parsed.axis == "x" else 1
    if parsed.side == "high":

        def development(point: np.ndarray) -> bool:
            return bool(point[index] <= parsed.threshold - parsed.buffer)

        def holdout(point: np.ndarray) -> bool:
            return bool(point[index] >= parsed.threshold + parsed.buffer)

    else:

        def development(point: np.ndarray) -> bool:
            return bool(point[index] >= parsed.threshold + parsed.buffer)

        def holdout(point: np.ndarray) -> bool:
            return bool(point[index] <= parsed.threshold - parsed.buffer)

    return development, holdout


def random_split_dataframe(
    frame: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Seeded row split for data already restricted to a development region."""

    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be in (0, 1).")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("Train and validation fractions must sum to < 1.")
    generator = np.random.default_rng(int(seed))
    indices = generator.permutation(len(frame))
    train_end = int(round(len(frame) * train_fraction))
    validation_end = train_end + int(round(len(frame) * validation_fraction))
    return {
        "train": frame.iloc[indices[:train_end]].reset_index(drop=True),
        "validation": frame.iloc[
            indices[train_end:validation_end]
        ].reset_index(drop=True),
        "test": frame.iloc[indices[validation_end:]].reset_index(drop=True),
    }


def split_dataframe_spatial(
    frame: pd.DataFrame,
    spec: SpatialHoldoutSpec | Mapping[str, Any],
    *,
    seed: int = 42,
    train_fraction: float = 0.80,
) -> dict[str, pd.DataFrame]:
    """Create train/validation and spatial holdout frames with a guard band."""

    parsed = spec if isinstance(spec, SpatialHoldoutSpec) else SpatialHoldoutSpec.from_config(spec)
    points = frame[["true_x", "true_y"]].to_numpy(dtype=float)
    development_mask, holdout_mask, guard_mask = spatial_holdout_masks(points, parsed)
    development = frame.loc[development_mask].reset_index(drop=True)
    holdout = frame.loc[holdout_mask].reset_index(drop=True)
    generator = np.random.default_rng(int(seed))
    indices = generator.permutation(len(development))
    train_end = int(round(train_fraction * len(development)))
    return {
        "train": development.iloc[indices[:train_end]].reset_index(drop=True),
        "validation": development.iloc[indices[train_end:]].reset_index(drop=True),
        "spatial_holdout": holdout,
        "guard_band": frame.loc[guard_mask].reset_index(drop=True),
    }


def minimum_axis_separation(
    development: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    axis: str = "x",
) -> float:
    """Measure the minimum one-dimensional split separation."""

    if development.empty or holdout.empty:
        return float("nan")
    column = "true_x" if axis.lower() == "x" else "true_y"
    first = development[column].to_numpy(dtype=float)
    second = holdout[column].to_numpy(dtype=float)
    return float(np.min(np.abs(first[:, None] - second[None, :])))

