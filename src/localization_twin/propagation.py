"""Seeded RSS propagation with structured indoor impairments."""

from __future__ import annotations

import json
import logging
from math import hypot, log10
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter

from .environment import Environment
from .geometry import PointLike, as_point

LOGGER = logging.getLogger(__name__)


class SpatialBiasField:
    """Smooth, anchor-specific Gaussian random fields over a floorplan.

    The field is generated on a small grid, Gaussian-smoothed, normalized, and
    evaluated with bilinear interpolation. It is deterministic for a given
    seed and creates repeatable local RSS patterns without expensive ray
    tracing.
    """

    def __init__(
        self,
        width: float,
        height: float,
        anchor_count: int,
        *,
        amplitude: float = 0.0,
        grid_shape: Sequence[int] = (9, 13),
        correlation_sigma: float = 1.25,
        seed: int = 0,
        enabled: bool = True,
    ) -> None:
        if len(grid_shape) != 2:
            raise ValueError("spatial_bias.grid_shape must contain [rows, columns].")
        rows, columns = int(grid_shape[0]), int(grid_shape[1])
        if rows < 2 or columns < 2:
            raise ValueError("Spatial bias grid dimensions must both be at least 2.")
        if anchor_count < 1:
            raise ValueError("Spatial bias field requires at least one anchor.")
        self.width = float(width)
        self.height = float(height)
        self.amplitude = max(0.0, float(amplitude))
        self.enabled = bool(enabled) and self.amplitude > 0.0
        self.grid_shape = rows, columns
        rng = np.random.default_rng(int(seed))
        raw = rng.normal(size=(anchor_count, rows, columns))
        sigma = max(0.0, float(correlation_sigma))
        if sigma > 0.0:
            raw = gaussian_filter(raw, sigma=(0.0, sigma, sigma), mode="reflect")
        flat_mean = raw.mean(axis=(1, 2), keepdims=True)
        flat_std = raw.std(axis=(1, 2), keepdims=True)
        self._fields = (raw - flat_mean) / np.maximum(flat_std, 1e-12)
        self._fields *= self.amplitude

    def evaluate(self, position: PointLike, anchor_index: int) -> float:
        """Bilinearly interpolate one anchor's field at *position*."""

        if not self.enabled:
            return 0.0
        x, y = as_point(position)
        rows, columns = self.grid_shape
        gx = np.clip(x / self.width * (columns - 1), 0.0, columns - 1)
        gy = np.clip(y / self.height * (rows - 1), 0.0, rows - 1)
        x0, y0 = int(np.floor(gx)), int(np.floor(gy))
        x1, y1 = min(x0 + 1, columns - 1), min(y0 + 1, rows - 1)
        tx, ty = gx - x0, gy - y0
        field = self._fields[int(anchor_index)]
        top = (1.0 - tx) * field[y0, x0] + tx * field[y0, x1]
        bottom = (1.0 - tx) * field[y1, x0] + tx * field[y1, x1]
        return float((1.0 - ty) * top + ty * bottom)

    def evaluate_many(
        self, positions: np.ndarray, anchor_index: int
    ) -> np.ndarray:
        """Vector-friendly field evaluation for a set of coordinates."""

        points = np.asarray(positions, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("positions must have shape (n_samples, 2).")
        return np.asarray(
            [self.evaluate(point, anchor_index) for point in points], dtype=float
        )


class PropagationModel:
    """Simplified RSS model with wall loss, NLoS bias, drift, and dropout."""

    def __init__(
        self, environment: Environment, config: Mapping[str, Any], seed: int = 42
    ) -> None:
        self.environment = environment
        section = config.get("propagation", config)
        if not isinstance(section, Mapping):
            raise ValueError("Propagation configuration must be a mapping.")
        self.config = dict(section)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.reference_distance = float(section.get("reference_distance", 1.0))
        self.minimum_distance = float(section.get("minimum_distance", 0.35))
        if self.reference_distance <= 0.0 or self.minimum_distance <= 0.0:
            raise ValueError("Propagation reference and minimum distances must be > 0.")
        self.default_reference_power = float(
            section.get("default_reference_power", -38.0)
        )
        self.default_path_loss_exponent = float(
            section.get("default_path_loss_exponent", 2.0)
        )
        self.noise_std = float(section.get("noise_std", 0.0))
        self.nlos_bias_mean = float(section.get("nlos_bias_mean", 0.0))
        self.nlos_bias_std = float(section.get("nlos_bias_std", 0.0))
        self.wall_loss_multiplier = float(section.get("wall_loss_multiplier", 1.0))
        self.obstacle_loss_multiplier = float(
            section.get("obstacle_loss_multiplier", 1.0)
        )
        self.dropout_probability = float(section.get("dropout_probability", 0.0))
        if self.noise_std < 0.0 or self.nlos_bias_std < 0.0:
            raise ValueError("Noise standard deviations cannot be negative.")
        if not 0.0 <= self.dropout_probability <= 1.0:
            raise ValueError("dropout_probability must be in [0, 1].")
        self.fixed_offline_anchors = frozenset(
            str(value) for value in section.get("fixed_offline_anchors", [])
        )
        unknown = self.fixed_offline_anchors.difference(environment.anchor_ids)
        if unknown:
            raise ValueError(f"Unknown fixed offline anchors: {sorted(unknown)}")

        spatial_config = section.get("spatial_bias", {})
        if not isinstance(spatial_config, Mapping):
            raise ValueError("propagation.spatial_bias must be a mapping.")
        field_seed = self.seed + int(spatial_config.get("seed_offset", 0))
        self.spatial_field = SpatialBiasField(
            environment.width,
            environment.height,
            len(environment.anchors),
            amplitude=float(spatial_config.get("amplitude", 0.0)),
            grid_shape=spatial_config.get("grid_shape", (9, 13)),
            correlation_sigma=float(
                spatial_config.get("correlation_sigma", 1.25)
            ),
            seed=field_seed,
            enabled=bool(spatial_config.get("enabled", True)),
        )

    def reset(self, seed: int | None = None) -> None:
        """Reset independent measurement/dropout randomness.

        The spatial field remains the environment realization constructed for
        this model. Instantiate a new model to obtain another field.
        """

        if seed is not None:
            self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

    def _anchor_measurement(
        self,
        position: PointLike,
        anchor_index: int,
        rng: np.random.Generator,
    ) -> dict[str, float | bool]:
        anchor = self.environment.anchors[anchor_index]
        x, y = as_point(position)
        distance = hypot(x - anchor.x, y - anchor.y)
        modeled_distance = max(distance, self.minimum_distance)
        reference_power = float(anchor.reference_power)
        calibrated_exponent = float(anchor.path_loss_exponent)
        if calibrated_exponent <= 0.0:
            calibrated_exponent = self.default_path_loss_exponent
        region = self.environment.region_at(position)
        exponent = calibrated_exponent + (
            region.path_loss_exponent_offset if region is not None else 0.0
        )
        exponent = max(0.1, exponent)
        local_noise_std = self.noise_std * (
            region.noise_std_multiplier if region is not None else 1.0
        )
        local_nlos_mean = self.nlos_bias_mean + (
            region.nlos_bias_offset if region is not None else 0.0
        )

        walls = self.environment.walls_crossed(position, anchor.position)
        obstacles = self.environment.obstacles_crossed(position, anchor.position)
        los = len(walls) == 0 and len(obstacles) == 0
        material_loss = (
            self.wall_loss_multiplier
            * sum(wall.attenuation for wall in walls)
            + self.obstacle_loss_multiplier
            * sum(obstacle.attenuation for obstacle in obstacles)
        )
        nlos_bias = 0.0
        if not los:
            nlos_bias = max(
                0.05,
                float(rng.normal(local_nlos_mean, self.nlos_bias_std)),
            )
        spatial_bias = self.spatial_field.evaluate(position, anchor_index)
        random_noise = float(rng.normal(0.0, local_noise_std))
        rss = (
            reference_power
            - 10.0
            * exponent
            * log10(modeled_distance / self.reference_distance)
            - material_loss
            - nlos_bias
            + spatial_bias
            + random_noise
            + float(anchor.hardware_bias)
        )
        available = (
            anchor.online
            and anchor.anchor_id not in self.fixed_offline_anchors
            and float(rng.random()) >= self.dropout_probability
        )
        if available:
            estimated_distance = self.reference_distance * 10.0 ** (
                (reference_power - rss) / (10.0 * calibrated_exponent)
            )
        else:
            rss = float("nan")
            estimated_distance = float("nan")
        return {
            "rss": float(rss),
            "true_distance": float(distance),
            "estimated_distance": float(estimated_distance),
            "los": bool(los),
            "available": bool(available),
        }

    def measure(
        self, position: PointLike, rng: np.random.Generator | None = None
    ) -> dict[str, Any]:
        """Generate a complete seeded measurement row for one position."""

        x, y = as_point(position)
        if not self.environment.contains((x, y)):
            raise ValueError(f"Measurement position is outside the environment: {(x, y)}")
        generator = self.rng if rng is None else rng
        result: dict[str, Any] = {"true_x": x, "true_y": y}
        availability_bits: list[str] = []
        nlos_count = 0
        for index, anchor in enumerate(self.environment.anchors):
            measurement = self._anchor_measurement((x, y), index, generator)
            anchor_id = anchor.anchor_id
            result[f"rss_{anchor_id}"] = measurement["rss"]
            result[f"true_distance_{anchor_id}"] = measurement["true_distance"]
            result[f"estimated_distance_{anchor_id}"] = measurement[
                "estimated_distance"
            ]
            result[f"los_{anchor_id}"] = measurement["los"]
            result[f"available_{anchor_id}"] = measurement["available"]
            available = bool(measurement["available"])
            availability_bits.append("1" if available else "0")
            if available and not bool(measurement["los"]):
                nlos_count += 1
        result["nlos_anchor_count"] = int(nlos_count)
        result["anchor_availability_mask"] = json.dumps(
            [int(bit) for bit in availability_bits], separators=(",", ":")
        )
        result["sample_class"] = "NLoS" if nlos_count > 0 else "LoS"
        return result

    simulate = measure

    def simulate_positions(
        self,
        positions: Sequence[PointLike] | np.ndarray,
        *,
        split: str,
        scenario_name: str | None = None,
        trajectory_ids: Sequence[str | None] | str | None = None,
        timesteps: Sequence[int | float | None] | None = None,
        random_seed: int | None = None,
    ) -> pd.DataFrame:
        """Simulate an ordered collection of positions into a tidy dataframe."""

        points = np.asarray(positions, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2:
            raise ValueError("positions must have shape (n_samples, 2).")
        sample_count = len(points)
        if isinstance(trajectory_ids, str) or trajectory_ids is None:
            trajectory_values = [trajectory_ids] * sample_count
        else:
            trajectory_values = list(trajectory_ids)
            if len(trajectory_values) != sample_count:
                raise ValueError("trajectory_ids length must match positions.")
        if timesteps is None:
            timestep_values: list[int | float | None] = [None] * sample_count
        else:
            timestep_values = list(timesteps)
            if len(timestep_values) != sample_count:
                raise ValueError("timesteps length must match positions.")

        effective_seed = self.seed if random_seed is None else int(random_seed)
        scenario = scenario_name or "normal"
        rows: list[dict[str, Any]] = []
        for index, point in enumerate(points):
            row = self.measure(point)
            row["scenario_name"] = scenario
            row["trajectory_id"] = trajectory_values[index]
            row["timestep"] = timestep_values[index]
            row["random_seed"] = effective_seed
            row["split"] = str(split)
            rows.append(row)
        frame = pd.DataFrame.from_records(rows)
        if not frame.empty:
            frame["trajectory_id"] = frame["trajectory_id"].astype("string")
        return frame


RSSPropagationModel = PropagationModel
