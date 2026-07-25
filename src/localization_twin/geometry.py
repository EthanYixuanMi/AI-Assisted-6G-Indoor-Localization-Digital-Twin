"""Numerically robust two-dimensional geometry primitives."""

from __future__ import annotations

from math import hypot
from typing import Iterable, Sequence, TypeAlias

Point: TypeAlias = tuple[float, float]
PointLike: TypeAlias = Sequence[float]


def as_point(value: PointLike) -> Point:
    """Return a validated two-dimensional point."""

    if len(value) != 2:
        raise ValueError(f"A 2-D point requires two coordinates, got {value!r}.")
    return float(value[0]), float(value[1])


def cross(a: PointLike, b: PointLike, c: PointLike) -> float:
    """Signed twice-area of triangle ``a-b-c``."""

    ax, ay = as_point(a)
    bx, by = as_point(b)
    cx, cy = as_point(c)
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


def point_on_segment(
    point: PointLike,
    start: PointLike,
    end: PointLike,
    *,
    epsilon: float = 1e-9,
) -> bool:
    """Return whether *point* lies on the closed line segment."""

    px, py = as_point(point)
    ax, ay = as_point(start)
    bx, by = as_point(end)
    scale = max(1.0, hypot(bx - ax, by - ay))
    if abs(cross((ax, ay), (bx, by), (px, py))) > epsilon * scale:
        return False
    return (
        min(ax, bx) - epsilon <= px <= max(ax, bx) + epsilon
        and min(ay, by) - epsilon <= py <= max(ay, by) + epsilon
    )


def segments_intersect(
    first_start: PointLike,
    first_end: PointLike,
    second_start: PointLike,
    second_end: PointLike,
    *,
    include_endpoints: bool = True,
    epsilon: float = 1e-9,
) -> bool:
    """Test two closed 2-D segments, including collinear overlap.

    ``include_endpoints=False`` tests only a proper interior crossing. The
    default is appropriate for radio blockage, where grazing a wall still
    represents a blocked path.
    """

    p1, p2 = as_point(first_start), as_point(first_end)
    q1, q2 = as_point(second_start), as_point(second_end)
    o1 = cross(p1, p2, q1)
    o2 = cross(p1, p2, q2)
    o3 = cross(q1, q2, p1)
    o4 = cross(q1, q2, p2)

    proper = (
        ((o1 > epsilon and o2 < -epsilon) or (o1 < -epsilon and o2 > epsilon))
        and ((o3 > epsilon and o4 < -epsilon) or (o3 < -epsilon and o4 > epsilon))
    )
    if proper:
        return True
    if not include_endpoints:
        return False
    return (
        (abs(o1) <= epsilon and point_on_segment(q1, p1, p2, epsilon=epsilon))
        or (abs(o2) <= epsilon and point_on_segment(q2, p1, p2, epsilon=epsilon))
        or (abs(o3) <= epsilon and point_on_segment(p1, q1, q2, epsilon=epsilon))
        or (abs(o4) <= epsilon and point_on_segment(p2, q1, q2, epsilon=epsilon))
    )


def point_in_rectangle(
    point: PointLike,
    rectangle: Sequence[float],
    *,
    inclusive: bool = True,
    margin: float = 0.0,
) -> bool:
    """Test a point against ``(x_min, y_min, x_max, y_max)``.

    A positive *margin* expands the rectangle and is useful for obstacle
    clearance checks.
    """

    if len(rectangle) != 4:
        raise ValueError("A rectangle requires (x_min, y_min, x_max, y_max).")
    x, y = as_point(point)
    x_min, y_min, x_max, y_max = (float(value) for value in rectangle)
    if x_min > x_max or y_min > y_max:
        raise ValueError(f"Invalid rectangle bounds: {rectangle!r}")
    x_min -= margin
    y_min -= margin
    x_max += margin
    y_max += margin
    if inclusive:
        return x_min <= x <= x_max and y_min <= y <= y_max
    return x_min < x < x_max and y_min < y < y_max


def rectangle_edges(rectangle: Sequence[float]) -> tuple[tuple[Point, Point], ...]:
    """Return the four closed edges of an axis-aligned rectangle."""

    if len(rectangle) != 4:
        raise ValueError("A rectangle requires four bounds.")
    x_min, y_min, x_max, y_max = (float(value) for value in rectangle)
    if x_min > x_max or y_min > y_max:
        raise ValueError(f"Invalid rectangle bounds: {rectangle!r}")
    return (
        ((x_min, y_min), (x_max, y_min)),
        ((x_max, y_min), (x_max, y_max)),
        ((x_max, y_max), (x_min, y_max)),
        ((x_min, y_max), (x_min, y_min)),
    )


def segment_intersects_rectangle(
    start: PointLike,
    end: PointLike,
    rectangle: Sequence[float],
    *,
    include_boundary: bool = True,
    epsilon: float = 1e-9,
) -> bool:
    """Return whether a segment enters or touches an axis-aligned rectangle."""

    if point_in_rectangle(start, rectangle, inclusive=include_boundary):
        return True
    if point_in_rectangle(end, rectangle, inclusive=include_boundary):
        return True
    return any(
        segments_intersect(
            start,
            end,
            edge_start,
            edge_end,
            include_endpoints=include_boundary,
            epsilon=epsilon,
        )
        for edge_start, edge_end in rectangle_edges(rectangle)
    )


def distance_point_to_segment(
    point: PointLike, start: PointLike, end: PointLike
) -> float:
    """Compute the Euclidean distance from a point to a finite segment."""

    px, py = as_point(point)
    ax, ay = as_point(start)
    bx, by = as_point(end)
    dx, dy = bx - ax, by - ay
    denominator = dx * dx + dy * dy
    if denominator == 0.0:
        return hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / denominator))
    return hypot(px - (ax + t * dx), py - (ay + t * dy))


def polyline_length(points: Iterable[PointLike]) -> float:
    """Return the total length of consecutive polyline segments."""

    normalized = [as_point(point) for point in points]
    return sum(
        hypot(b[0] - a[0], b[1] - a[1])
        for a, b in zip(normalized[:-1], normalized[1:])
    )

