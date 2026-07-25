"""Reusable floor-plan drawing for publication and dashboard views."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import matplotlib.axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle
import numpy as np

from .style import DARK, NEUTRAL, OKABE_ITO


def draw_floorplan(
    ax: matplotlib.axes.Axes,
    environment: dict[str, Any],
    *,
    dark: bool = False,
    anchor_status: dict[str, bool] | None = None,
    show_anchor_labels: bool = True,
    show_grid: bool = True,
) -> None:
    """Draw room bounds, walls, obstacles, and anchors on ``ax``."""

    width = float(environment["width"])
    height = float(environment["height"])
    ink = DARK["text"] if dark else NEUTRAL["ink"]
    wall_color = "#A9C4D8" if dark else "#3D4852"
    obstacle_face = "#20384C" if dark else "#DDE3E8"
    obstacle_edge = "#7892A5" if dark else "#7A8793"
    boundary = Rectangle(
        (0.0, 0.0),
        width,
        height,
        fill=False,
        linewidth=1.7,
        edgecolor=wall_color,
        zorder=2,
    )
    ax.add_patch(boundary)

    for obstacle in environment.get("obstacles", []):
        if "vertices" in obstacle:
            patch = Polygon(
                obstacle["vertices"],
                closed=True,
                facecolor=obstacle_face,
                edgecolor=obstacle_edge,
                linewidth=0.9,
                hatch="////" if not dark else None,
                alpha=0.85,
                zorder=1,
            )
        else:
            patch = Rectangle(
                (float(obstacle["x"]), float(obstacle["y"])),
                float(obstacle["width"]),
                float(obstacle["height"]),
                facecolor=obstacle_face,
                edgecolor=obstacle_edge,
                linewidth=0.9,
                hatch="////" if not dark else None,
                alpha=0.85,
                zorder=1,
            )
        ax.add_patch(patch)

    for wall in environment.get("walls", []):
        start = wall["start"]
        end = wall["end"]
        ax.plot(
            [start[0], end[0]],
            [start[1], end[1]],
            color=wall_color,
            linewidth=2.7,
            solid_capstyle="round",
            zorder=3,
        )

    status = anchor_status or {}
    for anchor in environment.get("anchors", []):
        anchor_id = str(anchor["anchor_id"])
        online = bool(status.get(anchor_id, anchor.get("online", True)))
        color = OKABE_ITO["sky_blue"] if online else NEUTRAL["light"]
        marker = "^" if online else "X"
        ax.scatter(
            [float(anchor["x"])],
            [float(anchor["y"])],
            s=52,
            marker=marker,
            c=[color],
            edgecolors=ink,
            linewidths=0.75,
            zorder=8,
        )
        if show_anchor_labels:
            ax.annotate(
                anchor_id,
                (float(anchor["x"]), float(anchor["y"])),
                xytext=(4, 4),
                textcoords="offset points",
                color=ink,
                fontsize=7.0 if not dark else 8.0,
                fontweight="bold",
                zorder=9,
            )

    ax.set_xlim(-0.4, width + 0.4)
    ax.set_ylim(-0.4, height + 0.4)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    if show_grid:
        ax.grid(True, linewidth=0.45, alpha=0.45)
    else:
        ax.grid(False)


def draw_signal_links(
    ax: matplotlib.axes.Axes,
    target_xy: tuple[float, float],
    telemetry: Iterable[dict[str, Any]],
    *,
    dark: bool = False,
) -> None:
    """Draw LoS/NLoS radio links using color plus line-style redundancy."""

    target_x, target_y = target_xy
    for item in telemetry:
        if not _as_bool(item.get("available"), default=True):
            continue
        is_los = _as_bool(item.get("los"), default=True)
        color = (
            OKABE_ITO["sky_blue"] if is_los else OKABE_ITO["vermillion"]
        )
        ax.plot(
            [float(item["x"]), target_x],
            [float(item["y"]), target_y],
            color=color,
            linestyle="-" if is_los else "--",
            linewidth=1.0,
            alpha=0.62 if dark else 0.5,
            zorder=4,
        )


def floorplan_legend(dark: bool = False) -> list[Line2D]:
    """Return a compact semantic legend for map overlays."""

    ink = DARK["text"] if dark else NEUTRAL["ink"]
    return [
        Line2D(
            [0],
            [0],
            marker="^",
            color="none",
            markerfacecolor=OKABE_ITO["sky_blue"],
            markeredgecolor=ink,
            label="Online anchor",
        ),
        Line2D(
            [0],
            [0],
            marker="X",
            color="none",
            markerfacecolor=NEUTRAL["light"],
            markeredgecolor=ink,
            label="Offline anchor",
        ),
        Line2D(
            [0],
            [0],
            color=OKABE_ITO["sky_blue"],
            linestyle="-",
            label="LoS link",
        ),
        Line2D(
            [0],
            [0],
            color=OKABE_ITO["vermillion"],
            linestyle="--",
            label="NLoS link",
        ),
    ]


def _as_bool(value: object, *, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return default
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "online", "los", "available"}:
        return True
    if text in {"0", "false", "no", "offline", "nlos", "unavailable"}:
        return False
    return default


def telemetry_status_map(
    telemetry: Iterable[dict[str, Any]],
) -> dict[str, bool]:
    """Convert telemetry rows to the status mapping expected by floorplan."""

    return {
        str(item["anchor_id"]): _as_bool(item.get("available"), default=True)
        for item in telemetry
    }
