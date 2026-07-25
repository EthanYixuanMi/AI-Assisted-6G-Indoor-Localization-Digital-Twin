"""Nonlinear geometric least-squares indoor localization."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .features import infer_anchor_ids


def _normalise_anchors(
    anchors: Mapping[str, Any] | Sequence[Any] | np.ndarray,
    anchor_ids: Sequence[str] | None,
) -> tuple[list[str], np.ndarray]:
    if isinstance(anchors, Mapping):
        ids = list(anchor_ids) if anchor_ids is not None else list(anchors)
        coordinates: list[tuple[float, float]] = []
        for anchor_id in ids:
            value = anchors[anchor_id]
            if isinstance(value, Mapping):
                coordinates.append((float(value["x"]), float(value["y"])))
            elif hasattr(value, "x") and hasattr(value, "y"):
                coordinates.append((float(value.x), float(value.y)))
            else:
                coordinates.append((float(value[0]), float(value[1])))
        positions = np.asarray(coordinates, dtype=float)
    else:
        values = list(anchors)
        if values and isinstance(values[0], Mapping):
            ids = (
                list(anchor_ids)
                if anchor_ids is not None
                else [str(value["anchor_id"]) for value in values]
            )
            lookup = {str(value["anchor_id"]): value for value in values}
            positions = np.asarray(
                [(float(lookup[key]["x"]), float(lookup[key]["y"])) for key in ids],
                dtype=float,
            )
        elif values and hasattr(values[0], "x") and hasattr(values[0], "y"):
            ids = (
                list(anchor_ids)
                if anchor_ids is not None
                else [
                    str(getattr(value, "anchor_id", index))
                    for index, value in enumerate(values)
                ]
            )
            positions = np.asarray(
                [(float(value.x), float(value.y)) for value in values], dtype=float
            )
        else:
            positions = np.asarray(anchors, dtype=float)
            ids = (
                list(anchor_ids)
                if anchor_ids is not None
                else [str(index) for index in range(len(positions))]
            )

    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("Anchor positions must have shape (n_anchors, 2).")
    if len(ids) != len(positions):
        raise ValueError("anchor_ids and anchor positions have different lengths.")
    if len(ids) == 0:
        raise ValueError("At least one anchor must be supplied.")
    if len(set(ids)) != len(ids):
        raise ValueError("Anchor identifiers must be unique.")
    if not np.all(np.isfinite(positions)):
        raise ValueError("Anchor positions must be finite.")
    return ids, positions


class GeometricLocator:
    """Weighted nonlinear least-squares locator.

    With fewer than three usable anchors the estimate falls back to the
    previous valid position (for a sequential frame), otherwise to the bounded
    centroid of the usable anchors.  This makes anchor-failure experiments
    explicit and deterministic instead of allowing the optimizer to return an
    underdetermined arbitrary point.
    """

    def __init__(
        self,
        anchors: Mapping[str, Any] | Sequence[Any] | np.ndarray,
        anchor_ids: Sequence[str] | None = None,
        *,
        bounds: tuple[Sequence[float], Sequence[float]] | None = None,
        use_rss_weights: bool = False,
        minimum_weight: float = 0.2,
        use_previous_fallback: bool = True,
        max_nfev: int = 100,
    ) -> None:
        environment_bounds: tuple[np.ndarray, np.ndarray] | None = None
        if (
            hasattr(anchors, "anchors")
            and hasattr(anchors, "width")
            and hasattr(anchors, "height")
        ):
            environment = anchors
            if anchor_ids is None and hasattr(environment, "anchor_ids"):
                anchor_ids = list(environment.anchor_ids)
            anchors = environment.anchors
            environment_bounds = (
                np.asarray([0.0, 0.0]),
                np.asarray([float(environment.width), float(environment.height)]),
            )
        self.anchor_ids, self.anchor_positions = _normalise_anchors(
            anchors, anchor_ids
        )
        if bounds is None:
            if environment_bounds is not None:
                lower, upper = environment_bounds
            else:
                lower = np.min(self.anchor_positions, axis=0)
                upper = np.max(self.anchor_positions, axis=0)
                degenerate = upper <= lower
                lower = np.where(degenerate, lower - 1.0, lower)
                upper = np.where(degenerate, upper + 1.0, upper)
        else:
            lower = np.asarray(bounds[0], dtype=float)
            upper = np.asarray(bounds[1], dtype=float)
        if lower.shape != (2,) or upper.shape != (2,) or np.any(lower >= upper):
            raise ValueError("bounds must be two length-2 vectors with lower < upper.")
        self.bounds = (lower, upper)
        self.use_rss_weights = bool(use_rss_weights)
        self.minimum_weight = float(minimum_weight)
        if not 0.0 < self.minimum_weight <= 1.0:
            raise ValueError("minimum_weight must be in (0, 1].")
        self.use_previous_fallback = bool(use_previous_fallback)
        self.max_nfev = int(max_nfev)
        if self.max_nfev <= 0:
            raise ValueError("max_nfev must be positive.")

    def fit(self, frame: pd.DataFrame | None = None, y: Any | None = None) -> "GeometricLocator":
        """Return ``self`` for compatibility with learned locators."""

        return self

    def _fallback(
        self,
        available_positions: np.ndarray,
        previous: np.ndarray | None,
    ) -> np.ndarray:
        lower, upper = self.bounds
        if (
            self.use_previous_fallback
            and previous is not None
            and previous.shape == (2,)
            and np.all(np.isfinite(previous))
        ):
            return np.clip(previous, lower, upper)
        if len(available_positions):
            return np.clip(np.mean(available_positions, axis=0), lower, upper)
        return np.clip(np.mean(self.anchor_positions, axis=0), lower, upper)

    def _rss_weights(self, rss: np.ndarray) -> np.ndarray:
        finite = np.isfinite(rss)
        if not self.use_rss_weights or finite.sum() <= 1:
            return np.ones(len(rss), dtype=float)
        valid = rss[finite]
        span = float(np.max(valid) - np.min(valid))
        weights = np.ones(len(rss), dtype=float)
        if span > 1e-12:
            scaled = (valid - np.min(valid)) / span
            weights[finite] = self.minimum_weight + (1.0 - self.minimum_weight) * scaled
        return weights

    def predict_details(
        self,
        frame: pd.DataFrame,
        *,
        previous_position: Sequence[float] | None = None,
    ) -> pd.DataFrame:
        """Predict coordinates and return optimizer diagnostics.

        The returned columns are ``geometric_x``, ``geometric_y``,
        ``residual_cost``, ``available_count``, ``optimization_success``, and
        ``used_fallback``.  Its index matches the input frame.
        """

        if not isinstance(frame, pd.DataFrame):
            raise TypeError("GeometricLocator.predict expects a pandas DataFrame.")
        missing = [
            f"estimated_distance_{anchor_id}"
            for anchor_id in self.anchor_ids
            if f"estimated_distance_{anchor_id}" not in frame
        ]
        if missing:
            raise ValueError(f"Missing distance column(s): {', '.join(missing)}")

        previous = (
            np.asarray(previous_position, dtype=float)
            if previous_position is not None
            else None
        )
        if previous is not None and previous.shape != (2,):
            raise ValueError("previous_position must be a length-2 coordinate.")

        rows: list[dict[str, float | int | bool]] = []
        lower, upper = self.bounds
        for _, row in frame.iterrows():
            distances = np.asarray(
                [
                    pd.to_numeric(
                        pd.Series([row[f"estimated_distance_{anchor_id}"]]),
                        errors="coerce",
                    ).iloc[0]
                    for anchor_id in self.anchor_ids
                ],
                dtype=float,
            )
            explicit_available_values = [
                row.get(f"available_{anchor_id}", True)
                for anchor_id in self.anchor_ids
            ]
            explicit_available = np.asarray(
                [
                    False
                    if pd.isna(value)
                    else bool(float(value) > 0.5)
                    for value in explicit_available_values
                ],
                dtype=bool,
            )
            usable = explicit_available & np.isfinite(distances) & (distances > 0.0)
            positions = self.anchor_positions[usable]
            observed = distances[usable]
            count = int(np.sum(usable))

            success = False
            used_fallback = count < 3
            residual_cost = np.nan
            if count >= 3:
                rss = np.asarray(
                    [
                        row.get(f"rss_{anchor_id}", np.nan)
                        for anchor_id, is_usable in zip(self.anchor_ids, usable)
                        if is_usable
                    ],
                    dtype=float,
                )
                weights = self._rss_weights(rss)
                # A previous boundary solution can be a poor local minimum for
                # the next unrelated/static sample.  Use the anchor centroid
                # for every determined solve; reserve ``previous`` strictly
                # for the documented failure/underdetermined fallback.
                initial = np.clip(np.mean(positions, axis=0), lower, upper)

                def residuals(point: np.ndarray) -> np.ndarray:
                    return np.sqrt(weights) * (
                        np.linalg.norm(point - positions, axis=1) - observed
                    )

                try:
                    result = least_squares(
                        residuals,
                        x0=initial,
                        bounds=(lower, upper),
                        max_nfev=self.max_nfev,
                        method="trf",
                    )
                    if result.success and np.all(np.isfinite(result.x)):
                        estimate = np.clip(result.x, lower, upper)
                        success = True
                        residual_cost = float(np.mean(np.square(residuals(estimate))))
                    else:
                        estimate = self._fallback(positions, previous)
                        used_fallback = True
                except (ValueError, FloatingPointError, RuntimeError):
                    estimate = self._fallback(positions, previous)
                    used_fallback = True
            else:
                estimate = self._fallback(positions, previous)

            if not np.isfinite(residual_cost):
                if count:
                    raw_residual = np.linalg.norm(estimate - positions, axis=1) - observed
                    residual_cost = float(np.mean(np.square(raw_residual)))
                else:
                    residual_cost = 0.0
            rows.append(
                {
                    "geometric_x": float(estimate[0]),
                    "geometric_y": float(estimate[1]),
                    "residual_cost": residual_cost,
                    "available_count": count,
                    "optimization_success": success,
                    "used_fallback": used_fallback,
                }
            )
            previous = estimate
        return pd.DataFrame(rows, index=frame.index)

    def predict(
        self,
        frame: pd.DataFrame,
        *,
        return_details: bool = False,
        previous_position: Sequence[float] | None = None,
    ) -> np.ndarray | pd.DataFrame:
        """Return ``(x, y)`` predictions, optionally with diagnostics."""

        details = self.predict_details(frame, previous_position=previous_position)
        if return_details:
            return details
        return details.loc[:, ["geometric_x", "geometric_y"]].to_numpy()

    def save(self, path: str | Path) -> Path:
        """Serialize this deterministic locator with joblib."""

        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "GeometricLocator":
        """Load a locator previously written by :meth:`save`."""

        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"{path!s} does not contain a {cls.__name__}.")
        return loaded


GeometricLeastSquares = GeometricLocator
