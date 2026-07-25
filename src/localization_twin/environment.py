"""Environment entities and line-of-sight queries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import logging
from typing import Any, Mapping, Sequence

import numpy as np

from .geometry import (
    Point,
    PointLike,
    as_point,
    distance_point_to_segment,
    point_in_rectangle,
    segment_intersects_rectangle,
    segments_intersect,
)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Anchor:
    """Fixed radio anchor and its calibration parameters."""

    anchor_id: str
    x: float
    y: float
    reference_power: float = -38.0
    path_loss_exponent: float = 2.0
    online: bool = True
    hardware_bias: float = 0.0

    @property
    def position(self) -> Point:
        """Anchor coordinate as an ``(x, y)`` tuple."""

        return self.x, self.y


@dataclass(frozen=True, slots=True)
class Wall:
    """Finite line wall carrying an attenuation in decibels."""

    wall_id: str
    start: Point
    end: Point
    attenuation: float
    wall_type: str = "generic"


@dataclass(frozen=True, slots=True)
class RectangleObstacle:
    """Axis-aligned obstacle excluded from target sampling."""

    obstacle_id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    attenuation: float
    material: str = "generic"

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Rectangle bounds in geometry-module order."""

        return self.x_min, self.y_min, self.x_max, self.y_max


@dataclass(frozen=True, slots=True)
class PropagationRegion:
    """Rectangular region that locally adjusts propagation statistics."""

    region_id: str
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    path_loss_exponent_offset: float = 0.0
    noise_std_multiplier: float = 1.0
    nlos_bias_offset: float = 0.0

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        """Rectangle bounds in geometry-module order."""

        return self.x_min, self.y_min, self.x_max, self.y_max


class Environment:
    """Two-dimensional floorplan with anchors, walls, and obstacles."""

    def __init__(
        self,
        width: float,
        height: float,
        anchors: Sequence[Anchor],
        walls: Sequence[Wall] = (),
        obstacles: Sequence[RectangleObstacle] = (),
        regions: Sequence[PropagationRegion] = (),
        boundary_margin: float = 0.0,
    ) -> None:
        if width <= 0.0 or height <= 0.0:
            raise ValueError("Environment width and height must be positive.")
        if len(anchors) < 1:
            raise ValueError("Environment requires at least one anchor.")
        identifiers = [anchor.anchor_id for anchor in anchors]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("Anchor IDs must be unique.")
        self.width = float(width)
        self.height = float(height)
        self.anchors = tuple(anchors)
        self.walls = tuple(walls)
        self.obstacles = tuple(obstacles)
        self.regions = tuple(regions)
        self.boundary_margin = max(0.0, float(boundary_margin))
        self._anchor_by_id = {anchor.anchor_id: anchor for anchor in anchors}

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "Environment":
        """Construct an environment from a full or environment-only config."""

        section = config.get("environment", config)
        if not isinstance(section, Mapping):
            raise ValueError("Environment configuration must be a mapping.")
        anchors = tuple(
            Anchor(
                anchor_id=str(item["anchor_id"]),
                x=float(item["x"]),
                y=float(item["y"]),
                reference_power=float(item.get("reference_power", -38.0)),
                path_loss_exponent=float(item.get("path_loss_exponent", 2.0)),
                online=bool(item.get("online", True)),
                hardware_bias=float(item.get("hardware_bias", 0.0)),
            )
            for item in section.get("anchors", [])
        )
        walls = tuple(
            Wall(
                wall_id=str(item.get("wall_id", f"W{index + 1}")),
                start=as_point(item["start"]),
                end=as_point(item["end"]),
                attenuation=float(item.get("attenuation", 0.0)),
                wall_type=str(item.get("wall_type", "generic")),
            )
            for index, item in enumerate(section.get("walls", []))
        )
        obstacles = tuple(
            RectangleObstacle(
                obstacle_id=str(item.get("obstacle_id", f"O{index + 1}")),
                x_min=float(item["x_min"]),
                y_min=float(item["y_min"]),
                x_max=float(item["x_max"]),
                y_max=float(item["y_max"]),
                attenuation=float(item.get("attenuation", 0.0)),
                material=str(item.get("material", "generic")),
            )
            for index, item in enumerate(section.get("obstacles", []))
        )
        regions = tuple(
            PropagationRegion(
                region_id=str(item.get("region_id", f"R{index + 1}")),
                x_min=float(item["x_min"]),
                y_min=float(item["y_min"]),
                x_max=float(item["x_max"]),
                y_max=float(item["y_max"]),
                path_loss_exponent_offset=float(
                    item.get("path_loss_exponent_offset", 0.0)
                ),
                noise_std_multiplier=float(
                    item.get("noise_std_multiplier", 1.0)
                ),
                nlos_bias_offset=float(item.get("nlos_bias_offset", 0.0)),
            )
            for index, item in enumerate(section.get("regions", []))
        )
        environment = cls(
            width=float(section["width"]),
            height=float(section["height"]),
            anchors=anchors,
            walls=walls,
            obstacles=obstacles,
            regions=regions,
            boundary_margin=float(section.get("boundary_margin", 0.0)),
        )
        environment._validate_entities()
        return environment

    def _validate_entities(self) -> None:
        for anchor in self.anchors:
            if not self.contains(anchor.position, include_boundary=True):
                raise ValueError(f"Anchor {anchor.anchor_id} is outside the floorplan.")
        for obstacle in self.obstacles:
            if (
                obstacle.x_min >= obstacle.x_max
                or obstacle.y_min >= obstacle.y_max
                or obstacle.x_min < 0.0
                or obstacle.y_min < 0.0
                or obstacle.x_max > self.width
                or obstacle.y_max > self.height
            ):
                raise ValueError(
                    f"Obstacle {obstacle.obstacle_id} has invalid bounds."
                )
        for wall in self.walls:
            if not self.contains(wall.start) or not self.contains(wall.end):
                raise ValueError(f"Wall {wall.wall_id} extends outside the floorplan.")
        for region in self.regions:
            if (
                region.x_min >= region.x_max
                or region.y_min >= region.y_max
                or region.x_min < 0.0
                or region.y_min < 0.0
                or region.x_max > self.width
                or region.y_max > self.height
                or region.noise_std_multiplier < 0.0
            ):
                raise ValueError(f"Propagation region {region.region_id} is invalid.")

    @property
    def anchor_ids(self) -> tuple[str, ...]:
        """Stable ordered anchor identifiers."""

        return tuple(anchor.anchor_id for anchor in self.anchors)

    @property
    def anchor_positions(self) -> np.ndarray:
        """Anchor coordinates as an ``(n_anchors, 2)`` NumPy array."""

        return np.asarray([anchor.position for anchor in self.anchors], dtype=float)

    def anchor(self, anchor_id: str) -> Anchor:
        """Look up an anchor by identifier."""

        try:
            return self._anchor_by_id[anchor_id]
        except KeyError as exc:
            raise KeyError(f"Unknown anchor ID: {anchor_id}") from exc

    def contains(
        self, point: PointLike, *, include_boundary: bool = True, margin: float = 0.0
    ) -> bool:
        """Return whether a point lies inside the floorplan bounds."""

        x, y = as_point(point)
        low = max(0.0, float(margin))
        if include_boundary:
            return low <= x <= self.width - low and low <= y <= self.height - low
        return low < x < self.width - low and low < y < self.height - low

    def obstacle_at(
        self, point: PointLike, *, clearance: float = 0.0
    ) -> RectangleObstacle | None:
        """Return the first obstacle containing a point, if any."""

        for obstacle in self.obstacles:
            if point_in_rectangle(
                point, obstacle.bounds, inclusive=True, margin=max(0.0, clearance)
            ):
                return obstacle
        return None

    def region_at(self, point: PointLike) -> PropagationRegion | None:
        """Return the first configured propagation region containing a point."""

        for region in self.regions:
            if point_in_rectangle(point, region.bounds, inclusive=True):
                return region
        return None

    def is_navigable(self, point: PointLike, clearance: float = 0.0) -> bool:
        """Check floorplan bounds and obstacle/wall clearance."""

        clearance = max(0.0, float(clearance))
        required_margin = max(self.boundary_margin, clearance)
        if not self.contains(point, include_boundary=True, margin=required_margin):
            return False
        if self.obstacle_at(point, clearance=clearance) is not None:
            return False
        if clearance > 0.0 and any(
            distance_point_to_segment(point, wall.start, wall.end) <= clearance
            for wall in self.walls
        ):
            return False
        return True

    def walls_crossed(self, start: PointLike, end: PointLike) -> tuple[Wall, ...]:
        """Return walls intersected by the direct propagation segment."""

        return tuple(
            wall
            for wall in self.walls
            if segments_intersect(start, end, wall.start, wall.end)
        )

    def obstacles_crossed(
        self, start: PointLike, end: PointLike
    ) -> tuple[RectangleObstacle, ...]:
        """Return rectangular obstacles intersected by a direct path."""

        return tuple(
            obstacle
            for obstacle in self.obstacles
            if segment_intersects_rectangle(start, end, obstacle.bounds)
        )

    def wall_count(self, start: PointLike, end: PointLike) -> int:
        """Count wall segments intersected by a radio path."""

        return len(self.walls_crossed(start, end))

    count_walls = wall_count

    def obstacle_count(self, start: PointLike, end: PointLike) -> int:
        """Count rectangular obstacles intersected by a radio path."""

        return len(self.obstacles_crossed(start, end))

    def is_los(self, start: PointLike, end: PointLike) -> bool:
        """Return true only when no configured wall or obstacle blocks the path."""

        return self.wall_count(start, end) == 0 and self.obstacle_count(start, end) == 0

    def attenuation(
        self,
        start: PointLike,
        end: PointLike,
        *,
        wall_multiplier: float = 1.0,
        obstacle_multiplier: float = 1.0,
    ) -> float:
        """Sum configured wall and obstacle losses for a direct path."""

        wall_loss = sum(wall.attenuation for wall in self.walls_crossed(start, end))
        obstacle_loss = sum(
            obstacle.attenuation for obstacle in self.obstacles_crossed(start, end)
        )
        return float(wall_multiplier * wall_loss + obstacle_multiplier * obstacle_loss)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable floorplan description."""

        return {
            "width": self.width,
            "height": self.height,
            "boundary_margin": self.boundary_margin,
            "anchors": [asdict(anchor) for anchor in self.anchors],
            "walls": [asdict(wall) for wall in self.walls],
            "obstacles": [asdict(obstacle) for obstacle in self.obstacles],
            "regions": [asdict(region) for region in self.regions],
        }
