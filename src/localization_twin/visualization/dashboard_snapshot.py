"""Browser-free, high-resolution dashboard composites from saved replays."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd

from .data import (
    ResultBundle,
    VisualDataError,
    anchor_telemetry,
    infer_anchor_columns,
    select_replay,
    select_scenario,
)
from .floorplan import (
    draw_floorplan,
    draw_signal_links,
    telemetry_status_map,
)
from .style import (
    ALGORITHM_LINESTYLES,
    DARK,
    NEUTRAL,
    OKABE_ITO,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    algorithm_color,
    canonical_scenario,
    configure_dashboard_style,
    ordered_algorithms,
)


SNAPSHOT_DPI = 200
SCREENSHOT_NAMES = {
    scenario: f"dashboard_{scenario}.png" for scenario in SCENARIO_ORDER
}


def _bool_value(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isnan(value):
        return default
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "online", "available", "los"}:
        return True
    if text in {"false", "0", "no", "offline", "unavailable", "nlos"}:
        return False
    return default


def _current_rows(replay: pd.DataFrame, timestep: float) -> pd.DataFrame:
    records: list[pd.Series] = []
    for _, rows in replay.groupby("algorithm", sort=False):
        index = (rows["timestep"] - timestep).abs().idxmin()
        records.append(rows.loc[index])
    if not records:
        raise VisualDataError("The selected replay contains no algorithm rows.")
    return pd.DataFrame(records)


def _preferred_algorithm(algorithms: list[str]) -> str:
    for candidate in (
        "Residual AI + Kalman",
        "Residual AI",
        "Direct AI",
        "KNN",
        "Geometric LS",
    ):
        if candidate in algorithms:
            return candidate
    return algorithms[-1]


def _metric_value(
    metrics: pd.DataFrame | None,
    predictions: pd.DataFrame,
    scenario: str,
    algorithm: str,
    column: str,
) -> float:
    if metrics is not None and column in metrics:
        subset = metrics[
            (metrics["scenario"] == canonical_scenario(scenario))
            & (metrics["algorithm"] == algorithm)
        ]
        values = pd.to_numeric(subset[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.mean())
    errors = predictions.loc[
        predictions["algorithm"] == algorithm, "error"
    ].dropna()
    if errors.empty:
        return float("nan")
    if column == "mean_error":
        return float(errors.mean())
    if column == "median_error":
        return float(errors.median())
    if column == "p90_error":
        return float(errors.quantile(0.9))
    if column == "nlos_mean_error":
        if "nlos_anchor_count" in predictions:
            count = pd.to_numeric(
                predictions["nlos_anchor_count"], errors="coerce"
            )
            nlos = predictions.loc[
                (predictions["algorithm"] == algorithm) & (count > 0), "error"
            ].dropna()
            return float(nlos.mean()) if not nlos.empty else float("nan")
    if column == "inference_time_ms":
        return float("nan")
    return float("nan")


def _format_metric(value: float, unit: str, decimals: int = 2) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{decimals}f} {unit}".strip()


def _draw_cards(
    fig: plt.Figure,
    cards: list[tuple[str, str]],
) -> None:
    start_x = 0.035
    end_x = 0.965
    gap = 0.008
    width = (end_x - start_x - gap * (len(cards) - 1)) / len(cards)
    for index, (label, value) in enumerate(cards):
        x = start_x + index * (width + gap)
        patch = FancyBboxPatch(
            (x, 0.82),
            width,
            0.085,
            boxstyle="round,pad=0.004,rounding_size=0.008",
            transform=fig.transFigure,
            facecolor=DARK["panel_alt"],
            edgecolor=DARK["grid"],
            linewidth=0.9,
        )
        fig.patches.append(patch)
        fig.text(
            x + 0.012,
            0.878,
            label.upper(),
            fontsize=7.4,
            color=DARK["muted"],
            weight="bold",
        )
        fig.text(
            x + 0.012,
            0.84,
            value,
            fontsize=12.5,
            color=DARK["text"],
            weight="bold",
        )


def _map_panel(
    ax: plt.Axes,
    environment: dict[str, Any],
    replay: pd.DataFrame,
    current: pd.DataFrame,
    timestep: float,
    telemetry: pd.DataFrame,
) -> None:
    status = telemetry_status_map(telemetry.to_dict("records"))
    draw_floorplan(
        ax,
        environment,
        dark=True,
        anchor_status=status,
        show_anchor_labels=True,
    )
    truth = (
        replay.sort_values("timestep")
        .drop_duplicates("timestep")
        [["timestep", "true_x", "true_y"]]
    )
    history_truth = truth[truth["timestep"] <= timestep]
    ax.plot(
        truth["true_x"],
        truth["true_y"],
        color=DARK["muted"],
        linestyle=":",
        linewidth=1.0,
        alpha=0.55,
        label="Full ground-truth path",
        zorder=3,
    )
    ax.plot(
        history_truth["true_x"],
        history_truth["true_y"],
        color=DARK["text"],
        linewidth=2.4,
        label="Ground truth",
        zorder=5,
    )
    target = current.iloc[0]
    draw_signal_links(
        ax,
        (float(target["true_x"]), float(target["true_y"])),
        telemetry.to_dict("records"),
        dark=True,
    )
    ax.scatter(
        [float(target["true_x"])],
        [float(target["true_y"])],
        s=90,
        marker="o",
        c=[OKABE_ITO["yellow"]],
        edgecolors=DARK["background"],
        linewidths=1.2,
        label="True position",
        zorder=12,
    )
    for algorithm in ordered_algorithms(replay["algorithm"]):
        rows = replay[
            (replay["algorithm"] == algorithm)
            & (replay["timestep"] <= timestep)
        ].sort_values("timestep")
        if rows.empty:
            continue
        ax.plot(
            rows["pred_x"],
            rows["pred_y"],
            color=algorithm_color(algorithm),
            linestyle=ALGORITHM_LINESTYLES.get(algorithm, "-"),
            linewidth=1.55,
            alpha=0.9,
            label=algorithm,
            zorder=6,
        )
        ax.scatter(
            [rows["pred_x"].iloc[-1]],
            [rows["pred_y"].iloc[-1]],
            s=42,
            marker="D",
            c=[algorithm_color(algorithm)],
            edgecolors=DARK["background"],
            linewidths=0.8,
            zorder=11,
        )
    ax.set_title(
        "Indoor Digital Twin Map", loc="left", weight="bold", pad=10
    )
    handles, labels = ax.get_legend_handles_labels()
    seen: set[str] = set()
    keep = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=False)
        if not (label in seen or seen.add(label))
    ]
    ax.legend(
        [item[0] for item in keep],
        [item[1] for item in keep],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.11),
        ncol=min(4, max(1, len(keep))),
        frameon=False,
        labelcolor=DARK["text"],
    )


def _error_panel(
    ax: plt.Axes,
    replay: pd.DataFrame,
    timestep: float,
) -> None:
    for algorithm in ordered_algorithms(replay["algorithm"]):
        rows = replay[replay["algorithm"] == algorithm].sort_values("timestep")
        ax.plot(
            rows["timestep"],
            rows["error"],
            color=algorithm_color(algorithm),
            linestyle=ALGORITHM_LINESTYLES.get(algorithm, "-"),
            linewidth=1.4,
            label=algorithm,
        )
    ax.axvline(
        timestep,
        color=OKABE_ITO["yellow"],
        linewidth=1.0,
        linestyle="--",
        alpha=0.9,
    )
    ax.set_title("Live Error Curve", loc="left", weight="bold")
    ax.set_xlabel("Trajectory timestep")
    ax.set_ylabel("Error (m)")
    ax.legend(frameon=False, ncol=2, labelcolor=DARK["text"])


def _anchor_panel(ax: plt.Axes, telemetry: pd.DataFrame) -> None:
    ax.set_title("Anchor Signal Panel", loc="left", weight="bold")
    rss = pd.to_numeric(telemetry["rss_dbm"], errors="coerce")
    if rss.notna().sum() == 0:
        ax.axis("off")
        ax.text(
            0.02,
            0.62,
            "Per-anchor RSS was not recorded.",
            transform=ax.transAxes,
            color=DARK["text"],
            fontsize=10,
            weight="bold",
        )
        ax.text(
            0.02,
            0.36,
            "Add rss_A1, est_distance_A1, los_A1 and available_A1\n"
            "columns to per_sample_predictions.csv.",
            transform=ax.transAxes,
            color=DARK["muted"],
            fontsize=8.5,
        )
        return
    labels = telemetry["anchor_id"].astype(str).tolist()
    positions = np.arange(len(labels))
    colors = []
    hatches = []
    for _, row in telemetry.iterrows():
        online = _bool_value(row["available"], default=True)
        los = _bool_value(row["los"], default=True)
        colors.append(
            (
                OKABE_ITO["sky_blue"]
                if online and los
                else (
                    OKABE_ITO["vermillion"]
                    if online
                    else NEUTRAL["mid"]
                )
            )
        )
        hatches.append("" if online and los else ("//" if online else "xx"))
    baseline = min(-105.0, float(rss.min()) - 5.0)
    bars = ax.barh(
        positions,
        rss.fillna(baseline) - baseline,
        left=baseline,
        color=colors,
        edgecolor=DARK["text"],
        linewidth=0.35,
    )
    for bar, hatch in zip(bars, hatches, strict=False):
        bar.set_hatch(hatch)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("RSS (dBm)")
    valid_min = float(rss.min())
    valid_max = float(rss.max())
    ax.set_xlim(min(-105.0, valid_min - 4), min(-20.0, valid_max + 5))
    for position, (_, row) in enumerate(telemetry.iterrows()):
        distance = pd.to_numeric(
            pd.Series([row["estimated_distance_m"]]), errors="coerce"
        ).iloc[0]
        link = "LoS" if _bool_value(row["los"], default=True) else "NLoS"
        status = (
            link
            if _bool_value(row["available"], default=True)
            else "offline"
        )
        suffix = (
            f"{float(distance):.1f} m · {status}"
            if np.isfinite(distance)
            else status
        )
        value = rss.iloc[position]
        if np.isfinite(value):
            ax.text(
                float(value) + 1.0,
                position,
                suffix,
                va="center",
                fontsize=7.1,
                color=DARK["text"],
            )


def _comparison_panel(
    ax: plt.Axes,
    metrics: pd.DataFrame | None,
    replay: pd.DataFrame,
    scenario: str,
) -> None:
    if metrics is not None:
        rows = select_scenario(metrics, scenario, require_rows=False)
        if not rows.empty:
            comparison = (
                rows.groupby("algorithm", as_index=False)["mean_error"]
                .mean()
                .dropna()
            )
        else:
            comparison = pd.DataFrame()
    else:
        comparison = pd.DataFrame()
    if comparison.empty:
        comparison = (
            replay.groupby("algorithm", as_index=False)["error"]
            .mean()
            .rename(columns={"error": "mean_error"})
        )
    algorithms = ordered_algorithms(comparison["algorithm"])
    comparison = (
        comparison.set_index("algorithm").reindex(algorithms).reset_index()
    )
    x = np.arange(len(comparison))
    bars = ax.bar(
        x,
        comparison["mean_error"],
        color=[algorithm_color(name) for name in algorithms],
        edgecolor=DARK["text"],
        linewidth=0.35,
    )
    ax.set_xticks(x, algorithms, rotation=20, ha="right")
    ax.set_ylabel("Mean error (m)")
    ax.set_title("Method Comparison", loc="left", weight="bold")
    for bar, value in zip(bars, comparison["mean_error"], strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            float(value),
            f"{float(value):.2f}",
            ha="center",
            va="bottom",
            color=DARK["text"],
            fontsize=7.2,
        )


def build_dashboard_snapshot(
    bundle: ResultBundle,
    scenario: str,
    *,
    title_suffix: str | None = None,
) -> tuple[plt.Figure, pd.DataFrame]:
    """Build a 16:9 dashboard view for one saved scenario."""

    bundle.require("predictions")
    if bundle.environment is None:
        raise VisualDataError("Dashboard snapshots require environment.json.")
    canonical = canonical_scenario(scenario)
    replay = select_replay(bundle.predictions, canonical)
    algorithms = ordered_algorithms(replay["algorithm"])
    if not algorithms:
        raise VisualDataError(f"No algorithms are present for scenario {canonical}.")
    timesteps = np.sort(replay["timestep"].dropna().unique())
    if timesteps.size == 0:
        raise VisualDataError(f"No timesteps are present for scenario {canonical}.")
    frame_index = min(len(timesteps) - 1, max(0, int(round(0.72 * (len(timesteps) - 1)))))
    timestep = float(timesteps[frame_index])
    current = _current_rows(replay, timestep)
    preferred = _preferred_algorithm(algorithms)
    preferred_row = current[current["algorithm"] == preferred].iloc[0]

    anchor_columns = infer_anchor_columns(replay, bundle.environment["anchors"])
    telemetry = anchor_telemetry(
        preferred_row, bundle.environment, anchor_columns
    )
    available_count = int(
        sum(_bool_value(value, default=True) for value in telemetry["available"])
    )
    current_error = float(preferred_row["error"])
    mean_error = _metric_value(
        bundle.metrics, replay, canonical, preferred, "mean_error"
    )
    median_error = _metric_value(
        bundle.metrics, replay, canonical, preferred, "median_error"
    )
    p90_error = _metric_value(
        bundle.metrics, replay, canonical, preferred, "p90_error"
    )
    nlos_error = _metric_value(
        bundle.metrics, replay, canonical, preferred, "nlos_mean_error"
    )
    inference = _metric_value(
        bundle.metrics, replay, canonical, preferred, "inference_time_ms"
    )
    if not np.isfinite(inference) and bundle.runtime is not None:
        runtime_values = pd.to_numeric(
            bundle.runtime.loc[
                bundle.runtime["algorithm"] == preferred,
                "inference_time_ms",
            ],
            errors="coerce",
        ).dropna()
        if not runtime_values.empty:
            inference = float(runtime_values.mean())

    configure_dashboard_style()
    fig = plt.figure(figsize=(16.0, 9.0))
    scenario_label = SCENARIO_LABELS.get(canonical, canonical.replace("_", " ").title())
    fig.text(
        0.035,
        0.955,
        "AI-ASSISTED 6G INDOOR LOCALIZATION DIGITAL TWIN",
        fontsize=18,
        weight="bold",
        color=DARK["text"],
        ha="left",
    )
    fig.text(
        0.035,
        0.923,
        f"{scenario_label} · saved simulation replay · frame "
        f"{frame_index + 1}/{len(timesteps)}"
        + (f" · {title_suffix}" if title_suffix else ""),
        fontsize=9.5,
        color=DARK["muted"],
        ha="left",
    )
    fig.text(
        0.965,
        0.948,
        "SIMULATION — NOT A REAL 6G DEPLOYMENT",
        fontsize=8.0,
        color=OKABE_ITO["orange"],
        weight="bold",
        ha="right",
    )
    cards = [
        ("Current error", _format_metric(current_error, "m")),
        ("Mean error", _format_metric(mean_error, "m")),
        ("Median", _format_metric(median_error, "m")),
        ("P90", _format_metric(p90_error, "m")),
        ("NLoS error", _format_metric(nlos_error, "m")),
        ("Available", f"{available_count}/{len(telemetry)} anchors"),
        ("Inference", _format_metric(inference, "ms", decimals=3)),
    ]
    _draw_cards(fig, cards)

    # Extra left margin keeps the metre-labelled y axis intact in 16:9 exports.
    map_ax = fig.add_axes([0.055, 0.13, 0.505, 0.65])
    error_ax = fig.add_axes([0.60, 0.55, 0.36, 0.23])
    anchor_ax = fig.add_axes([0.60, 0.305, 0.36, 0.18])
    comparison_ax = fig.add_axes([0.60, 0.09, 0.36, 0.16])
    _map_panel(map_ax, bundle.environment, replay, current, timestep, telemetry)
    _error_panel(error_ax, replay, timestep)
    _anchor_panel(anchor_ax, telemetry)
    _comparison_panel(comparison_ax, bundle.metrics, replay, canonical)

    scenario_bits = [
        f"Scenario: {scenario_label}",
        f"Replay method focus: {preferred}",
        f"Seed: {preferred_row.get('seed', 'N/A')}",
        f"Trajectory: {preferred_row.get('trajectory_id', 'N/A')}",
    ]
    config_columns = [
        ("noise_std_db", "Noise", "dB"),
        ("wall_attenuation_db", "Wall loss", "dB"),
        ("nlos_bias_db", "NLoS bias", "dB"),
        ("dropped_anchor_count", "Dropped anchors", ""),
    ]
    for column, label, unit in config_columns:
        if column in preferred_row and pd.notna(preferred_row[column]):
            scenario_bits.append(f"{label}: {preferred_row[column]} {unit}".strip())
    fig.text(
        0.04,
        0.035,
        "  •  ".join(scenario_bits),
        fontsize=7.8,
        color=DARK["muted"],
        ha="left",
    )
    fig.text(
        0.96,
        0.035,
        "Source: saved per-sample predictions, metrics, and environment metadata",
        fontsize=7.2,
        color=DARK["muted"],
        ha="right",
    )
    return fig, replay


def _save_snapshot(
    fig: plt.Figure, path: Path, *, dpi: int = SNAPSHOT_DPI
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(
            path,
            format="png",
            dpi=dpi,
            facecolor=fig.get_facecolor(),
            bbox_inches=None,
        )
    finally:
        plt.close(fig)
    if not path.is_file() or path.stat().st_size < 100:
        raise VisualDataError(f"Dashboard snapshot export is empty: {path}")
    return path


def export_dashboard_snapshots(
    bundle: ResultBundle,
    figures_dir: str | Path,
    screenshots_dir: str | Path,
    data_dir: str | Path,
) -> list[Path]:
    """Export the overview plus one browser-free image for each scenario."""

    figures_root = Path(figures_dir)
    screenshots_root = Path(screenshots_dir)
    source_root = Path(data_dir)
    source_root.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []

    overview, source = build_dashboard_snapshot(
        bundle, "normal", title_suffix="overview"
    )
    overview_path = figures_root / "dashboard_overview.png"
    exported.append(_save_snapshot(overview, overview_path))
    source.to_csv(
        source_root / "dashboard_overview_source.csv",
        index=False,
        lineterminator="\n",
    )

    for scenario in SCENARIO_ORDER:
        fig, source = build_dashboard_snapshot(bundle, scenario)
        path = screenshots_root / SCREENSHOT_NAMES[scenario]
        exported.append(_save_snapshot(fig, path))
        source.to_csv(
            source_root / f"dashboard_{scenario}_source.csv",
            index=False,
            lineterminator="\n",
        )
    return exported
