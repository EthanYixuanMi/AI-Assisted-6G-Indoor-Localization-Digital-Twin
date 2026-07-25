"""Feature extraction shared by the localization models.

The public helpers deliberately accept pandas data frames.  This keeps the
column contract visible at the model boundary and avoids coupling the models
to the simulator implementation.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd


FULL_RESIDUAL_FEATURE_GROUPS: tuple[str, ...] = (
    "rss",
    "distance",
    "geometric",
    "residual",
    "available_count",
    "mask",
    "nlos",
    "los",
)

ABLATION_FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "full": FULL_RESIDUAL_FEATURE_GROUPS,
    "without_los_nlos": tuple(
        group for group in FULL_RESIDUAL_FEATURE_GROUPS if group not in {"los", "nlos"}
    ),
    "without_geometric_residual": tuple(
        group for group in FULL_RESIDUAL_FEATURE_GROUPS if group != "residual"
    ),
    "without_anchor_mask": tuple(
        group for group in FULL_RESIDUAL_FEATURE_GROUPS if group != "mask"
    ),
}

_VALID_GROUPS = {
    "rss",
    "distance",
    "mask",
    "los",
    "geometric",
    "residual",
    "available_count",
    "nlos",
    "counts",
}


def infer_anchor_ids(frame: pd.DataFrame) -> list[str]:
    """Infer anchor identifiers from ``rss_<anchor_id>`` columns.

    Column order is retained because it is normally the environment/config
    order and therefore stable between generated splits.
    """

    ids = [column[4:] for column in frame.columns if column.startswith("rss_")]
    if not ids:
        raise ValueError("No rss_<anchor_id> columns were found in the data frame.")
    return ids


def coordinate_targets(
    frame: pd.DataFrame,
    x_column: str = "true_x",
    y_column: str = "true_y",
) -> np.ndarray:
    """Return finite ``(x, y)`` targets from a localization data frame."""

    missing = [column for column in (x_column, y_column) if column not in frame]
    if missing:
        raise ValueError(f"Missing target column(s): {', '.join(missing)}")
    targets = frame.loc[:, [x_column, y_column]].to_numpy(dtype=float)
    if targets.ndim != 2 or targets.shape[1] != 2:
        raise ValueError("Targets must have shape (n_samples, 2).")
    if not np.all(np.isfinite(targets)):
        raise ValueError("Target coordinates contain NaN or infinite values.")
    return targets


def availability_matrix(
    frame: pd.DataFrame,
    anchor_ids: Sequence[str] | None = None,
) -> np.ndarray:
    """Extract a binary anchor-availability matrix.

    Explicit ``available_<id>`` columns take precedence.  When a legacy frame
    omits them, availability is inferred from a finite RSS or estimated
    distance value.
    """

    ids = list(anchor_ids) if anchor_ids is not None else infer_anchor_ids(frame)
    columns: list[np.ndarray] = []
    for anchor_id in ids:
        available_column = f"available_{anchor_id}"
        rss_column = f"rss_{anchor_id}"
        distance_column = f"estimated_distance_{anchor_id}"
        if available_column in frame:
            values = pd.to_numeric(frame[available_column], errors="coerce").to_numpy()
            available = np.isfinite(values) & (values > 0.5)
        else:
            available = np.zeros(len(frame), dtype=bool)
            if rss_column in frame:
                available |= np.isfinite(
                    pd.to_numeric(frame[rss_column], errors="coerce").to_numpy()
                )
            if distance_column in frame:
                available |= np.isfinite(
                    pd.to_numeric(frame[distance_column], errors="coerce").to_numpy()
                )
        columns.append(available.astype(float))
    if not columns:
        return np.empty((len(frame), 0), dtype=float)
    return np.column_stack(columns)


def _numeric_column(
    frame: pd.DataFrame,
    column: str,
    default: float = np.nan,
) -> np.ndarray:
    if column not in frame:
        return np.full(len(frame), default, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _resolve_geometric(
    frame: pd.DataFrame,
    geometric: Any | None,
    anchor_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Resolve geometric coordinates, residual cost, and anchor count."""

    source = frame if geometric is None else geometric
    residual = np.full(len(frame), np.nan, dtype=float)
    count = np.full(len(frame), np.nan, dtype=float)
    for candidate in (
        "residual_cost",
        "geometric_residual_cost",
        "geometric_residual",
    ):
        if candidate in frame:
            residual = _numeric_column(frame, candidate)
            break
    for candidate in (
        "available_count",
        "geometric_available_count",
        "available_anchor_count",
    ):
        if candidate in frame:
            count = _numeric_column(frame, candidate)
            break

    if isinstance(source, pd.DataFrame):
        coordinate_pairs = (
            ("geometric_x", "geometric_y"),
            ("pred_x", "pred_y"),
            ("estimated_x", "estimated_y"),
            ("x", "y"),
        )
        coordinates: np.ndarray | None = None
        for x_name, y_name in coordinate_pairs:
            if x_name in source and y_name in source:
                coordinates = source.loc[:, [x_name, y_name]].to_numpy(dtype=float)
                break
        if coordinates is None:
            raise ValueError(
                "Geometric data frame needs geometric_x/geometric_y or pred_x/pred_y."
            )
        for candidate in (
            "residual_cost",
            "geometric_residual_cost",
            "geometric_residual",
        ):
            if candidate in source:
                residual = _numeric_column(source, candidate)
                break
        for candidate in (
            "available_count",
            "geometric_available_count",
            "available_anchor_count",
        ):
            if candidate in source:
                count = _numeric_column(source, candidate)
                break
    elif source is frame and geometric is None:
        coordinate_pairs = (
            ("geometric_x", "geometric_y"),
            ("pred_x", "pred_y"),
        )
        coordinates = None
        for x_name, y_name in coordinate_pairs:
            if x_name in frame and y_name in frame:
                coordinates = frame.loc[:, [x_name, y_name]].to_numpy(dtype=float)
                break
        if coordinates is None:
            raise ValueError(
                "No geometric coordinates found. Pass geometric predictions or add "
                "geometric_x/geometric_y columns."
            )
    else:
        coordinates = np.asarray(source, dtype=float)

    if coordinates.shape != (len(frame), 2):
        raise ValueError(
            f"Geometric predictions must have shape ({len(frame)}, 2), "
            f"got {coordinates.shape}."
        )
    if np.any(~np.isfinite(coordinates)):
        raise ValueError("Geometric predictions contain NaN or infinite values.")

    missing_count = ~np.isfinite(count)
    if np.any(missing_count):
        inferred_count = availability_matrix(frame, anchor_ids).sum(axis=1)
        count = np.where(missing_count, inferred_count, count)
    residual = np.where(np.isfinite(residual), residual, 0.0)
    return coordinates, residual, count


def extract_features(
    frame: pd.DataFrame,
    anchor_ids: Sequence[str] | None = None,
    groups: Iterable[str] = ("rss", "mask"),
    *,
    geometric: Any | None = None,
    return_names: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Build a deterministic numeric feature matrix.

    RSS and distance values belonging to unavailable anchors are set to NaN so
    that a model pipeline can impute them without confusing dropout with a real
    measurement.  Mask features remain present to retain that information.

    Supported groups are ``rss``, ``distance``, ``mask``, ``los``,
    ``geometric``, ``residual``, ``available_count``, ``nlos``, and ``counts``
    (the latter expands to available-count plus NLoS-count).
    """

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("Feature extraction expects a pandas DataFrame.")
    ids = list(anchor_ids) if anchor_ids is not None else infer_anchor_ids(frame)
    requested = list(dict.fromkeys(groups))
    unknown = set(requested) - _VALID_GROUPS
    if unknown:
        raise ValueError(f"Unknown feature group(s): {sorted(unknown)}")

    available = availability_matrix(frame, ids)
    matrices: list[np.ndarray] = []
    names: list[str] = []
    geometric_values: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None

    for group in requested:
        if group == "rss":
            values = np.column_stack(
                [_numeric_column(frame, f"rss_{anchor_id}") for anchor_id in ids]
            )
            matrices.append(np.where(available > 0.5, values, np.nan))
            names.extend(f"rss_{anchor_id}" for anchor_id in ids)
        elif group == "distance":
            values = np.column_stack(
                [
                    _numeric_column(frame, f"estimated_distance_{anchor_id}")
                    for anchor_id in ids
                ]
            )
            matrices.append(np.where(available > 0.5, values, np.nan))
            names.extend(f"estimated_distance_{anchor_id}" for anchor_id in ids)
        elif group == "mask":
            matrices.append(available)
            names.extend(f"available_{anchor_id}" for anchor_id in ids)
        elif group == "los":
            los = np.column_stack(
                [
                    np.nan_to_num(
                        _numeric_column(frame, f"los_{anchor_id}", default=0.0),
                        nan=0.0,
                    )
                    for anchor_id in ids
                ]
            )
            matrices.append(np.where(available > 0.5, los, 0.0))
            names.extend(f"los_{anchor_id}" for anchor_id in ids)
        elif group in {
            "geometric",
            "residual",
            "available_count",
            "counts",
        }:
            if geometric_values is None:
                geometric_values = _resolve_geometric(frame, geometric, ids)
            coordinates, residual, count = geometric_values
            if group == "geometric":
                matrices.append(coordinates)
                names.extend(("geometric_x", "geometric_y"))
            elif group == "residual":
                matrices.append(residual[:, None])
                names.append("residual_cost")
            elif group == "available_count":
                matrices.append(count[:, None])
                names.append("available_count")
            else:
                nlos = _numeric_column(frame, "nlos_anchor_count", default=0.0)
                matrices.append(np.column_stack((count, np.nan_to_num(nlos, nan=0.0))))
                names.extend(("available_count", "nlos_anchor_count"))
        elif group == "nlos":
            nlos = np.nan_to_num(
                _numeric_column(frame, "nlos_anchor_count", default=0.0),
                nan=0.0,
            )
            matrices.append(nlos[:, None])
            names.append("nlos_anchor_count")

    matrix = np.column_stack(matrices) if matrices else np.empty((len(frame), 0))
    matrix = matrix.astype(float, copy=False)
    if return_names:
        return matrix, names
    return matrix


def extract_rss_mask_features(
    frame: pd.DataFrame,
    anchor_ids: Sequence[str] | None = None,
    *,
    return_names: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Convenience wrapper for fingerprinting features."""

    return extract_features(
        frame,
        anchor_ids,
        ("rss", "mask"),
        return_names=return_names,
    )


def extract_residual_features(
    frame: pd.DataFrame,
    anchor_ids: Sequence[str] | None = None,
    *,
    geometric: Any | None = None,
    feature_groups: Iterable[str] = FULL_RESIDUAL_FEATURE_GROUPS,
    return_names: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[str]]:
    """Convenience wrapper for residual-correction features."""

    return extract_features(
        frame,
        anchor_ids,
        feature_groups,
        geometric=geometric,
        return_names=return_names,
    )


def get_ablation_feature_groups(variant: str) -> tuple[str, ...]:
    """Return residual feature groups for a named ablation variant."""

    try:
        return ABLATION_FEATURE_GROUPS[variant]
    except KeyError as exc:
        raise ValueError(
            f"Unknown ablation variant {variant!r}; expected one of "
            f"{sorted(ABLATION_FEATURE_GROUPS)}."
        ) from exc
