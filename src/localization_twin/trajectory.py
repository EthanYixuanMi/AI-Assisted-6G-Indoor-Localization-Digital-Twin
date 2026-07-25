"""Static-position and continuous-trajectory sampling utilities."""

from __future__ import annotations

import logging
from math import ceil, sqrt
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .environment import Environment
from .geometry import PointLike, as_point

LOGGER = logging.getLogger(__name__)
PositionPredicate = Callable[[np.ndarray], bool]


def _generator(
    seed: int | None = None, rng: np.random.Generator | None = None
) -> np.random.Generator:
    if seed is not None and rng is not None:
        raise ValueError("Pass either seed or rng, not both.")
    return np.random.default_rng(seed) if rng is None else rng


def sample_uniform_positions(
    environment: Environment,
    count: int,
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    clearance: float = 0.0,
    predicate: PositionPredicate | None = None,
    max_batches: int = 1000,
) -> np.ndarray:
    """Draw exactly *count* uniform navigable positions by rejection sampling."""

    count = int(count)
    if count < 0:
        raise ValueError("count cannot be negative.")
    if count == 0:
        return np.empty((0, 2), dtype=float)
    generator = _generator(seed, rng)
    accepted: list[np.ndarray] = []
    remaining = count
    margin = max(environment.boundary_margin, float(clearance))
    if 2.0 * margin >= min(environment.width, environment.height):
        raise ValueError("Sampling clearance leaves no interior floor area.")

    for _ in range(int(max_batches)):
        batch_size = max(128, int(ceil(remaining * 1.6)))
        x = generator.uniform(margin, environment.width - margin, batch_size)
        y = generator.uniform(margin, environment.height - margin, batch_size)
        candidates = np.column_stack((x, y))
        for point in candidates:
            if not environment.is_navigable(point, clearance=clearance):
                continue
            if predicate is not None and not bool(predicate(point)):
                continue
            accepted.append(point.copy())
            remaining -= 1
            if remaining == 0:
                return np.asarray(accepted, dtype=float)
    raise RuntimeError(
        f"Unable to sample {count} navigable points after {max_batches} batches; "
        "check obstacle coverage and spatial predicates."
    )


def sample_grid_positions(
    environment: Environment,
    count: int,
    *,
    clearance: float = 0.0,
    predicate: PositionPredicate | None = None,
) -> np.ndarray:
    """Select approximately uniform grid points, returning exactly *count*."""

    count = int(count)
    if count < 0:
        raise ValueError("count cannot be negative.")
    if count == 0:
        return np.empty((0, 2), dtype=float)
    margin = max(environment.boundary_margin, float(clearance))
    oversampling = 1.35
    for _ in range(12):
        target = max(count, int(ceil(count * oversampling)))
        columns = max(
            2, int(ceil(sqrt(target * environment.width / environment.height)))
        )
        rows = max(2, int(ceil(target / columns)))
        xs = np.linspace(margin, environment.width - margin, columns)
        ys = np.linspace(margin, environment.height - margin, rows)
        candidates = np.asarray(
            [
                (x, y)
                for y in ys
                for x in xs
                if environment.is_navigable((x, y), clearance=clearance)
                and (predicate is None or bool(predicate(np.asarray((x, y)))))
            ],
            dtype=float,
        )
        if len(candidates) >= count:
            indices = np.linspace(0, len(candidates) - 1, count, dtype=int)
            return candidates[indices]
        oversampling *= 1.7
    raise RuntimeError(
        f"Unable to construct {count} valid grid samples; only "
        f"{len(candidates)} candidates were found."
    )


def sample_static_positions(
    environment: Environment,
    count: int,
    *,
    seed: int = 42,
    grid_fraction: float = 0.35,
    clearance: float = 0.0,
    predicate: PositionPredicate | None = None,
) -> np.ndarray:
    """Combine deterministic grid coverage with seeded random positions."""

    if not 0.0 <= float(grid_fraction) <= 1.0:
        raise ValueError("grid_fraction must be in [0, 1].")
    count = int(count)
    grid_count = int(round(count * float(grid_fraction)))
    random_count = count - grid_count
    grid = sample_grid_positions(
        environment,
        grid_count,
        clearance=clearance,
        predicate=predicate,
    )
    random = sample_uniform_positions(
        environment,
        random_count,
        seed=seed,
        clearance=clearance,
        predicate=predicate,
    )
    combined = np.vstack((grid, random)) if count else np.empty((0, 2))
    generator = np.random.default_rng(int(seed) + 7919)
    generator.shuffle(combined, axis=0)
    return combined


def interpolate_waypoints(
    waypoints: Sequence[PointLike], sample_count: int
) -> np.ndarray:
    """Interpolate a polyline at equal arc-length intervals."""

    normalized = np.asarray([as_point(point) for point in waypoints], dtype=float)
    if len(normalized) < 2:
        raise ValueError("A trajectory requires at least two waypoints.")
    sample_count = int(sample_count)
    if sample_count < 2:
        raise ValueError("A trajectory requires at least two samples.")
    deltas = np.diff(normalized, axis=0)
    lengths = np.linalg.norm(deltas, axis=1)
    if np.any(lengths <= 0.0):
        raise ValueError("Consecutive trajectory waypoints must be distinct.")
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.linspace(0.0, cumulative[-1], sample_count)
    points = np.empty((sample_count, 2), dtype=float)
    segment = 0
    for index, distance in enumerate(distances):
        while segment < len(lengths) - 1 and distance > cumulative[segment + 1]:
            segment += 1
        fraction = (distance - cumulative[segment]) / lengths[segment]
        points[index] = normalized[segment] + fraction * deltas[segment]
    points[-1] = normalized[-1]
    return points


def _trajectory_is_clear(
    environment: Environment, points: np.ndarray, clearance: float
) -> bool:
    """Check points and connecting segments against solid floorplan geometry."""

    if not all(
        environment.is_navigable(point, clearance=clearance) for point in points
    ):
        return False
    return all(
        environment.wall_count(start, end) == 0
        and environment.obstacle_count(start, end) == 0
        for start, end in zip(points[:-1], points[1:])
    )


def generate_trajectories(
    environment: Environment,
    config: Mapping[str, Any],
    *,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate configured continuous trajectories outside all obstacles."""

    sampling = config.get("sampling", config)
    if not isinstance(sampling, Mapping):
        raise ValueError("Sampling configuration must be a mapping.")
    trajectory_config = sampling.get("trajectories", {})
    if not isinstance(trajectory_config, Mapping):
        raise ValueError("sampling.trajectories must be a mapping.")
    default_count = int(trajectory_config.get("points_per_trajectory", 100))
    clearance = float(sampling.get("obstacle_clearance", 0.0))
    definitions = trajectory_config.get("definitions", [])
    if not definitions:
        raise ValueError("No trajectory definitions are configured.")

    rows: list[dict[str, Any]] = []
    for definition in definitions:
        trajectory_id = str(definition["trajectory_id"])
        sample_count = int(definition.get("sample_count", default_count))
        points = interpolate_waypoints(definition["waypoints"], sample_count)
        if not _trajectory_is_clear(environment, points, clearance):
            raise ValueError(
                f"Configured trajectory {trajectory_id!r} enters an obstacle "
                f"or leaves the floorplan (clearance={clearance})."
            )
        for timestep, (x, y) in enumerate(points):
            rows.append(
                {
                    "true_x": float(x),
                    "true_y": float(y),
                    "trajectory_id": trajectory_id,
                    "timestep": int(timestep),
                    "random_seed": int(seed),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    LOGGER.info(
        "Generated %d trajectory samples across %d paths",
        len(frame),
        len(definitions),
    )
    return frame
