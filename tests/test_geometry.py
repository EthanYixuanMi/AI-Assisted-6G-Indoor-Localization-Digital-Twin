"""Geometry and floorplan intersection tests."""

from __future__ import annotations

from localization_twin.environment import Anchor, Environment, RectangleObstacle, Wall
from localization_twin.geometry import (
    point_in_rectangle,
    segment_intersects_rectangle,
    segments_intersect,
)


def test_segment_intersection_handles_crossing_touching_and_collinear() -> None:
    assert segments_intersect((0, 0), (4, 4), (0, 4), (4, 0))
    assert segments_intersect((0, 0), (2, 0), (2, 0), (2, 2))
    assert segments_intersect((0, 0), (4, 0), (2, 0), (6, 0))
    assert not segments_intersect((0, 0), (1, 0), (2, 0), (3, 0))
    assert not segments_intersect((0, 0), (1, 1), (0, 2), (1, 3))
    assert not segments_intersect(
        (0, 0), (2, 0), (2, 0), (2, 2), include_endpoints=False
    )


def test_rectangle_predicates_include_boundary_and_crossing() -> None:
    rectangle = (1.0, 1.0, 3.0, 3.0)
    assert point_in_rectangle((2.0, 2.0), rectangle)
    assert point_in_rectangle((1.0, 2.0), rectangle)
    assert not point_in_rectangle((1.0, 2.0), rectangle, inclusive=False)
    assert segment_intersects_rectangle((0.0, 2.0), (4.0, 2.0), rectangle)
    assert segment_intersects_rectangle((0.0, 1.0), (4.0, 1.0), rectangle)
    assert not segment_intersects_rectangle((0.0, 0.0), (0.5, 0.5), rectangle)


def test_environment_counts_each_crossed_wall_and_obstacle() -> None:
    environment = Environment(
        10.0,
        8.0,
        anchors=(Anchor("A1", 1.0, 1.0), Anchor("A2", 9.0, 7.0)),
        walls=(
            Wall("W1", (3.0, 0.0), (3.0, 8.0), 4.0),
            Wall("W2", (7.0, 0.0), (7.0, 8.0), 5.0),
        ),
        obstacles=(
            RectangleObstacle("O1", 4.0, 3.0, 6.0, 5.0, 6.0),
        ),
    )
    assert environment.wall_count((1.0, 4.0), (9.0, 4.0)) == 2
    assert environment.obstacle_count((1.0, 4.0), (9.0, 4.0)) == 1
    assert not environment.is_los((1.0, 4.0), (9.0, 4.0))
    assert environment.is_los((1.0, 1.0), (2.0, 1.0))
    assert environment.attenuation((1.0, 4.0), (9.0, 4.0)) == 15.0


def test_navigability_rejects_obstacle_interior_and_clearance() -> None:
    environment = Environment(
        10.0,
        8.0,
        anchors=(Anchor("A1", 1.0, 1.0),),
        walls=(Wall("W1", (5.0, 0.0), (5.0, 8.0), 4.0),),
        obstacles=(
            RectangleObstacle("O1", 2.0, 2.0, 4.0, 4.0, 6.0),
        ),
        boundary_margin=0.2,
    )
    assert environment.is_navigable((1.0, 1.0))
    assert not environment.is_navigable((3.0, 3.0))
    assert not environment.is_navigable((4.1, 3.0), clearance=0.2)
    assert not environment.is_navigable((5.05, 6.0), clearance=0.1)
    assert not environment.is_navigable((0.1, 7.0))

