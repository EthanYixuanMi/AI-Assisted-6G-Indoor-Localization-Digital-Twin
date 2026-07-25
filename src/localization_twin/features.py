"""Consistent feature construction for localization estimators."""

from __future__ import annotations

import json
import re
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _natural_key(value: str) -> tuple[Any, ...]:
    return tuple(
        int(part) if part.isdigit() else part.lower()
        for part in re.split(r"(\d+)", value)
    )


def infer_anchor_ids(frame: pd.DataFrame) -> list[str]:
    """Infer stable, naturally sorted anchor IDs from RSS columns."""

    identifiers = [column[4:] for column in frame.columns if column.startswith("rss_")]
    if not identifiers:
        raise ValueError("No rss_<anchor_id> columns were found.")
    return sorted(identifiers, key=_natural_key)


def decode_availability_mask(mask: Any, anchor_count: int) -> np.ndarray:
    """Decode bit strings or JSON-style masks into a float vector."""

    if isinstance(mask, str):
        stripped = mask.strip()
        if len(stripped) == anchor_count and set(stripped).issubset({"0", "1"}):
            return np.fromiter((character == "1" for character in stripped), dtype=float)
        try:
            mask = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid anchor availability mask: {mask!r}") from exc
    if isinstance(mask, (list, tuple, np.ndarray)):
        values = np.asarray(mask, dtype=float).reshape(-1)
        if len(values) != anchor_count:
            raise ValueError(
                f"Availability mask length {len(values)} != {anchor_count}."
            )
        return (values > 0.0).astype(float)
    if isinstance(mask, (int, np.integer)):
        bits = format(int(mask), f"0{anchor_count}b")[-anchor_count:]
        return np.fromiter((character == "1" for character in bits), dtype=float)
    raise ValueError(f"Unsupported anchor availability mask: {mask!r}")


def availability_matrix(
    frame: pd.DataFrame, anchor_ids: Sequence[str] | None = None
) -> np.ndarray:
    """Return per-anchor availability, preferring explicit boolean columns."""

    ids = list(anchor_ids) if anchor_ids is not None else infer_anchor_ids(frame)
    columns = [f"available_{anchor_id}" for anchor_id in ids]
    if all(column in frame for column in columns):
        return frame[columns].fillna(False).to_numpy(dtype=float)
    if "anchor_availability_mask" not in frame:
        raise ValueError("Frame has neither availability columns nor mask column.")
    return np.vstack(
        [
            decode_availability_mask(mask, len(ids))
            for mask in frame["anchor_availability_mask"]
        ]
    )


def _append(
    matrices: list[np.ndarray],
    names: list[str],
    values: np.ndarray,
    value_names: Sequence[str],
) -> None:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        array = array[:, None]
    matrices.append(array)
    names.extend(value_names)


def build_feature_matrix(
    frame: pd.DataFrame,
    feature_set: str = "direct",
    anchor_ids: Sequence[str] | None = None,
    geometric: pd.DataFrame | Mapping[str, Sequence[float]] | np.ndarray | None = None,
    fill_rss: float = -110.0,
) -> tuple[np.ndarray, list[str]]:
    """Build KNN, direct-regression, or residual-correction features.

    Missing anchor RSS values are filled only after availability indicators
    have been retained, so a model can distinguish dropout from a weak signal.
    """

    ids = list(anchor_ids) if anchor_ids is not None else infer_anchor_ids(frame)
    normalized_set = feature_set.lower().replace("-", "_")
    if normalized_set not in {"knn", "direct", "residual"}:
        raise ValueError("feature_set must be 'knn', 'direct', or 'residual'.")
    matrices: list[np.ndarray] = []
    names: list[str] = []

    rss_columns = [f"rss_{anchor_id}" for anchor_id in ids]
    missing_rss = [column for column in rss_columns if column not in frame]
    if missing_rss:
        raise ValueError(f"Missing RSS columns: {missing_rss}")
    rss = frame[rss_columns].to_numpy(dtype=float)
    rss = np.nan_to_num(rss, nan=float(fill_rss), posinf=fill_rss, neginf=fill_rss)
    _append(matrices, names, rss, rss_columns)

    available = availability_matrix(frame, ids)
    available_names = [f"available_{anchor_id}" for anchor_id in ids]
    _append(matrices, names, available, available_names)
    if normalized_set == "knn":
        return np.column_stack(matrices), names

    los_columns = [f"los_{anchor_id}" for anchor_id in ids]
    los = np.column_stack(
        [
            frame[column].fillna(False).to_numpy(dtype=float)
            if column in frame
            else np.zeros(len(frame), dtype=float)
            for column in los_columns
        ]
    )
    _append(matrices, names, los, los_columns)
    nlos = (
        frame["nlos_anchor_count"].to_numpy(dtype=float)
        if "nlos_anchor_count" in frame
        else np.sum(available * (1.0 - los), axis=1)
    )
    _append(matrices, names, nlos, ["nlos_anchor_count"])
    if normalized_set == "direct":
        return np.column_stack(matrices), names

    distance_columns = [f"estimated_distance_{anchor_id}" for anchor_id in ids]
    missing_distance = [column for column in distance_columns if column not in frame]
    if missing_distance:
        raise ValueError(f"Missing estimated-distance columns: {missing_distance}")
    distances = frame[distance_columns].to_numpy(dtype=float)
    finite = distances[np.isfinite(distances)]
    distance_fill = max(50.0, float(np.max(finite))) if finite.size else 50.0
    distances = np.nan_to_num(
        distances, nan=distance_fill, posinf=distance_fill, neginf=distance_fill
    )
    _append(matrices, names, distances, distance_columns)

    if geometric is None:
        preferred = [
            "geometric_x",
            "geometric_y",
            "geometric_residual_cost",
            "geometric_residual",
            "geometric_available_count",
            "available_anchor_count",
        ]
        geometric_columns = [column for column in preferred if column in frame]
        if geometric_columns:
            values = frame[geometric_columns].to_numpy(dtype=float)
            _append(matrices, names, values, geometric_columns)
    elif isinstance(geometric, pd.DataFrame):
        values = geometric.to_numpy(dtype=float)
        _append(matrices, names, values, [str(column) for column in geometric.columns])
    elif isinstance(geometric, Mapping):
        geometric_names = [str(name) for name in geometric]
        values = np.column_stack(
            [np.asarray(geometric[name], dtype=float) for name in geometric]
        )
        _append(matrices, names, values, geometric_names)
    else:
        values = np.asarray(geometric, dtype=float)
        if values.ndim == 1:
            values = values[:, None]
        if values.shape[0] != len(frame):
            raise ValueError("geometric features must have one row per sample.")
        default_names = (
            ["geometric_x", "geometric_y"]
            if values.shape[1] == 2
            else [f"geometric_feature_{index}" for index in range(values.shape[1])]
        )
        _append(matrices, names, values, default_names)

    return np.column_stack(matrices), names


def target_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Return ground-truth ``(x, y)`` coordinates."""

    missing = {"true_x", "true_y"}.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing target columns: {sorted(missing)}")
    return frame[["true_x", "true_y"]].to_numpy(dtype=float)


build_features = build_feature_matrix

