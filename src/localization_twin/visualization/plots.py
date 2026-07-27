"""Publication-grade figures generated only from persisted experiment results."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import textwrap
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

from .data import ResultBundle, VisualDataError, select_replay, select_scenario
from .floorplan import draw_floorplan
from .style import (
    ALGORITHM_LINESTYLES,
    ALGORITHM_MARKERS,
    NEUTRAL,
    OKABE_ITO,
    algorithm_color,
    canonical_algorithm,
    configure_paper_style,
    ordered_algorithms,
)


PAPER_DPI = 320
FIGURE_STEMS = (
    "trajectory_comparison",
    "error_cdf",
    "spatial_error_heatmaps",
    "robustness_noise",
    "robustness_anchor_failure",
    "robustness_results",
    "los_nlos_comparison",
    "ablation_results",
    "runtime_comparison",
)

POINT_ALGORITHMS = (
    "Geometric LS",
    "KNN",
    "Direct AI",
    "Residual AI",
)


def _ensure_nonempty(frame: pd.DataFrame, description: str) -> None:
    if frame.empty:
        raise VisualDataError(f"No source rows are available for {description}.")


def _source_note(fig: plt.Figure, filenames: str) -> None:
    fig.text(
        0.01,
        0.008,
        f"Simulation source: {filenames}",
        ha="left",
        va="bottom",
        fontsize=6.0,
        color=NEUTRAL["mid"],
    )


def _save_pair(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = [output_dir / f"{stem}.pdf", output_dir / f"{stem}.png"]
    try:
        fig.savefig(paths[0], format="pdf", bbox_inches="tight", pad_inches=0.04)
        fig.savefig(
            paths[1],
            format="png",
            dpi=PAPER_DPI,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    finally:
        plt.close(fig)
    for path in paths:
        if not path.is_file() or path.stat().st_size < 100:
            raise VisualDataError(f"Figure export failed or is empty: {path}")
    return paths


def _save_source(frame: pd.DataFrame, data_dir: Path, stem: str) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{stem}_source.csv"
    frame.to_csv(path, index=False, lineterminator="\n")
    if not path.is_file() or path.stat().st_size == 0:
        raise VisualDataError(f"Could not save figure source data: {path}")
    return path


def _trajectory_figure(
    bundle: ResultBundle, data_dir: Path
) -> tuple[plt.Figure, pd.DataFrame]:
    bundle.require("predictions")
    if bundle.environment is None:
        raise VisualDataError("Trajectory export requires environment.json.")
    replay = select_replay(bundle.predictions, "normal")
    algorithms = ordered_algorithms(replay["algorithm"])
    if len(algorithms) < 2:
        raise VisualDataError(
            "trajectory_comparison requires predictions from at least two methods."
        )

    fig, ax = plt.subplots(figsize=(7.16, 4.0))
    draw_floorplan(ax, bundle.environment, dark=False)
    truth = (
        replay.sort_values("timestep")
        .drop_duplicates("timestep")
        [["timestep", "true_x", "true_y"]]
    )
    ax.plot(
        truth["true_x"],
        truth["true_y"],
        color=NEUTRAL["ink"],
        linewidth=2.2,
        label="Ground truth",
        zorder=6,
    )
    for algorithm in algorithms:
        rows = replay[replay["algorithm"] == algorithm].sort_values("timestep")
        ax.plot(
            rows["pred_x"],
            rows["pred_y"],
            color=algorithm_color(algorithm),
            linestyle=ALGORITHM_LINESTYLES.get(algorithm, "-"),
            linewidth=1.55,
            label=algorithm,
            alpha=0.94,
            zorder=5,
        )
    ax.scatter(
        [truth["true_x"].iloc[0], truth["true_x"].iloc[-1]],
        [truth["true_y"].iloc[0], truth["true_y"].iloc[-1]],
        marker="o",
        s=[24, 36],
        c=[OKABE_ITO["orange"], OKABE_ITO["bluish_green"]],
        edgecolors=NEUTRAL["ink"],
        linewidths=0.6,
        zorder=9,
    )
    ax.set_title("Trajectory localization under the Normal simulation scenario")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=min(5, len(algorithms) + 1),
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.23)
    _source_note(fig, "per_sample_predictions.csv; environment.json")
    return fig, replay


def _cdf_figure(
    bundle: ResultBundle, data_dir: Path
) -> tuple[plt.Figure, pd.DataFrame]:
    bundle.require("predictions")
    rows = select_scenario(bundle.predictions, "normal")
    if "split" not in rows:
        raise VisualDataError(
            "error_cdf requires a split column so the spatial-holdout "
            "protocol can be selected explicitly."
        )
    split = rows["split"].astype(str).str.strip().str.lower()
    rows = rows[split == "spatial_holdout"].copy()
    rows = rows[rows["algorithm"].isin(POINT_ALGORITHMS)].copy()
    _ensure_nonempty(rows, "error CDF")
    missing = [
        algorithm
        for algorithm in POINT_ALGORITHMS
        if algorithm not in set(rows["algorithm"])
    ]
    if missing:
        raise VisualDataError(
            "error_cdf requires the same spatial-holdout protocol for all "
            "four point estimators; missing: " + ", ".join(missing)
        )
    fig, ax = plt.subplots(figsize=(3.5, 2.55))
    for algorithm in POINT_ALGORITHMS:
        errors = (
            pd.to_numeric(
                rows.loc[rows["algorithm"] == algorithm, "error"],
                errors="coerce",
            )
            .dropna()
            .sort_values()
            .to_numpy()
        )
        if len(errors) == 0:
            raise VisualDataError(
                f"Error CDF has no valid error values for {algorithm}."
            )
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.step(
            errors,
            cdf,
            where="post",
            color=algorithm_color(algorithm),
            linestyle=ALGORITHM_LINESTYLES.get(algorithm, "-"),
            label=algorithm,
        )
    ax.set_xlabel("Euclidean localization error (m)")
    ax.set_ylabel("Empirical CDF")
    ax.set_ylim(0.0, 1.01)
    ax.set_title("Localization error distribution")
    ax.legend(frameon=False, loc="lower right")
    fig.subplots_adjust(bottom=0.19)
    _source_note(fig, "per_sample_predictions.csv")
    return fig, rows


def _spatial_heatmap_figure(
    bundle: ResultBundle, data_dir: Path
) -> tuple[plt.Figure, pd.DataFrame]:
    bundle.require("predictions")
    if bundle.environment is None:
        raise VisualDataError("Spatial heat maps require environment.json.")
    rows = select_scenario(bundle.predictions, "normal")
    algorithms = ordered_algorithms(rows["algorithm"])[:4]
    if len(algorithms) < 2:
        raise VisualDataError(
            "spatial_error_heatmaps requires at least two localization methods."
        )
    source = rows[rows["algorithm"].isin(algorithms)].copy()
    values = source["error"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise VisualDataError("Spatial heat-map errors contain no finite values.")
    vmin = 0.0
    vmax = float(np.quantile(finite, 0.95))
    if vmax <= vmin:
        vmax = float(finite.max()) + 1e-6

    columns = 2
    rows_count = int(np.ceil(len(algorithms) / columns))
    fig, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(7.16, 2.75 * rows_count),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    for ax, algorithm in zip(axes.flat, algorithms, strict=False):
        subset = source[source["algorithm"] == algorithm]
        draw_floorplan(
            ax,
            bundle.environment,
            dark=False,
            show_anchor_labels=False,
            show_grid=False,
        )
        ax.hexbin(
            subset["true_x"],
            subset["true_y"],
            C=subset["error"],
            reduce_C_function=np.mean,
            gridsize=23,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            mincnt=1,
            alpha=0.9,
            linewidths=0.15,
            edgecolors="white",
            zorder=4,
        )
        ax.set_title(algorithm)
    for ax in axes.flat[len(algorithms) :]:
        ax.set_visible(False)
    scalar = ScalarMappable(norm=Normalize(vmin=vmin, vmax=vmax), cmap="viridis")
    colorbar = fig.colorbar(
        scalar,
        ax=[ax for ax in axes.flat if ax.get_visible()],
        shrink=0.78,
        pad=0.025,
    )
    colorbar.set_label("Mean Euclidean error (m)")
    fig.suptitle(
        "Spatial localization-error distribution (95th-percentile color cap)",
        y=0.995,
    )
    fig.subplots_adjust(
        left=0.075,
        right=0.89,
        top=0.92,
        bottom=0.08,
        hspace=0.28,
        wspace=0.2,
    )
    _source_note(fig, "per_sample_predictions.csv; environment.json")
    return fig, source


def _robustness_subset(frame: pd.DataFrame, experiment: str) -> pd.DataFrame:
    normalized = frame["experiment"].astype(str).str.lower()
    aliases = (
        {"noise", "high_noise", "noise_robustness"}
        if experiment == "noise"
        else {
            "anchor_failure",
            "anchor_dropout",
            "missing_anchor",
            "anchors_dropped",
        }
    )
    result = frame[normalized.isin(aliases)].copy()
    _ensure_nonempty(result, f"{experiment} robustness")
    result["level"] = pd.to_numeric(result["level"], errors="coerce")
    result["mean_error"] = pd.to_numeric(
        result["mean_error"], errors="coerce"
    )
    result = result.dropna(subset=["level", "mean_error"])
    _ensure_nonempty(result, f"{experiment} robustness numeric values")
    return result


def _plot_robustness_axis(
    ax: plt.Axes,
    frame: pd.DataFrame,
    *,
    x_label: str,
    title: str,
) -> None:
    aggregate = (
        frame.groupby(["algorithm", "level"], as_index=False)
        .agg(
            mean_error=("mean_error", "mean"),
            std_error=("mean_error", "std"),
            seed_count=("mean_error", "count"),
        )
        .sort_values("level")
    )
    for algorithm in ordered_algorithms(aggregate["algorithm"]):
        rows = aggregate[aggregate["algorithm"] == algorithm]
        ax.plot(
            rows["level"],
            rows["mean_error"],
            color=algorithm_color(algorithm),
            marker=ALGORITHM_MARKERS.get(algorithm, "o"),
            linestyle=ALGORITHM_LINESTYLES.get(algorithm, "-"),
            label=algorithm,
        )
        if rows["seed_count"].max() > 1:
            standard = rows["std_error"].fillna(0.0).to_numpy()
            ax.fill_between(
                rows["level"].to_numpy(),
                (rows["mean_error"] - standard).to_numpy(),
                (rows["mean_error"] + standard).to_numpy(),
                color=algorithm_color(algorithm),
                alpha=0.12,
                linewidth=0,
            )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Mean Euclidean error (m)")
    ax.set_title(title)


def _robustness_noise_figure(
    bundle: ResultBundle, data_dir: Path
) -> tuple[plt.Figure, pd.DataFrame]:
    bundle.require("robustness")
    rows = _robustness_subset(bundle.robustness, "noise")
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    _plot_robustness_axis(
        ax,
        rows,
        x_label="RSS noise standard deviation (dB)",
        title="Robustness to measurement noise",
    )
    ax.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(bottom=0.2)
    _source_note(fig, "robustness_results.csv")
    return fig, rows


def _robustness_anchor_figure(
    bundle: ResultBundle, data_dir: Path
) -> tuple[plt.Figure, pd.DataFrame]:
    bundle.require("robustness")
    rows = _robustness_subset(bundle.robustness, "anchor_failure")
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    _plot_robustness_axis(
        ax,
        rows,
        x_label="Unavailable anchors (count)",
        title="Robustness to anchor failure",
    )
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(bottom=0.2)
    _source_note(fig, "robustness_results.csv")
    return fig, rows


def _robustness_combined_figure(
    bundle: ResultBundle, data_dir: Path
) -> tuple[plt.Figure, pd.DataFrame]:
    bundle.require("robustness")
    noise = _robustness_subset(bundle.robustness, "noise")
    anchors = _robustness_subset(bundle.robustness, "anchor_failure")
    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.75))
    _plot_robustness_axis(
        axes[0],
        noise,
        x_label="RSS noise std. dev. (dB)",
        title="(a) Measurement noise",
    )
    _plot_robustness_axis(
        axes[1],
        anchors,
        x_label="Unavailable anchors (count)",
        title="(b) Anchor failure",
    )
    axes[1].xaxis.set_major_locator(MaxNLocator(integer=True))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=min(4, len(labels)),
        frameon=False,
    )
    fig.subplots_adjust(left=0.09, right=0.985, top=0.91, bottom=0.28, wspace=0.31)
    _source_note(fig, "robustness_results.csv")
    return fig, pd.concat([noise, anchors], ignore_index=True)


def _los_nlos_source(bundle: ResultBundle) -> pd.DataFrame:
    bundle.require("metrics")
    metrics = select_scenario(bundle.metrics, "normal")
    columns = {"los_mean_error", "nlos_mean_error"}
    if columns <= set(metrics.columns):
        result = (
            metrics.groupby("algorithm", as_index=False)[
                ["los_mean_error", "nlos_mean_error"]
            ]
            .mean(numeric_only=True)
            .dropna(how="all", subset=["los_mean_error", "nlos_mean_error"])
        )
        if not result.empty:
            return result

    bundle.require("predictions")
    predictions = select_scenario(bundle.predictions, "normal")
    label_column = next(
        (
            candidate
            for candidate in ("is_los", "los", "sample_is_los")
            if candidate in predictions.columns
        ),
        None,
    )
    if label_column is None and "nlos_anchor_count" in predictions:
        predictions = predictions.copy()
        predictions["is_los"] = (
            pd.to_numeric(
                predictions["nlos_anchor_count"], errors="coerce"
            ).fillna(0)
            == 0
        )
        label_column = "is_los"
    if label_column is None:
        raise VisualDataError(
            "LoS/NLoS comparison requires metrics columns los_mean_error and "
            "nlos_mean_error, or a per-sample LoS indicator."
        )
    flags = predictions[label_column].astype(str).str.lower().map(
        {
            "true": True,
            "1": True,
            "los": True,
            "false": False,
            "0": False,
            "nlos": False,
        }
    )
    predictions = predictions.assign(_is_los=flags).dropna(subset=["_is_los"])
    records = []
    for algorithm, rows in predictions.groupby("algorithm"):
        records.append(
            {
                "algorithm": algorithm,
                "los_mean_error": rows.loc[rows["_is_los"], "error"].mean(),
                "nlos_mean_error": rows.loc[~rows["_is_los"], "error"].mean(),
            }
        )
    result = pd.DataFrame.from_records(records)
    _ensure_nonempty(result, "LoS/NLoS comparison")
    return result


def _los_nlos_figure(
    bundle: ResultBundle, data_dir: Path
) -> tuple[plt.Figure, pd.DataFrame]:
    source = _los_nlos_source(bundle)
    algorithms = ordered_algorithms(source["algorithm"])
    source = source.set_index("algorithm").reindex(algorithms).reset_index()
    x = np.arange(len(source))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.16, 2.75))
    ax.bar(
        x - width / 2,
        source["los_mean_error"],
        width,
        label="LoS samples",
        color=OKABE_ITO["sky_blue"],
        edgecolor=NEUTRAL["ink"],
        linewidth=0.45,
    )
    ax.bar(
        x + width / 2,
        source["nlos_mean_error"],
        width,
        label="NLoS samples",
        color=OKABE_ITO["vermillion"],
        edgecolor=NEUTRAL["ink"],
        linewidth=0.45,
        hatch="//",
    )
    ax.set_xticks(x, algorithms, rotation=18, ha="right")
    ax.set_ylabel("Mean Euclidean error (m)")
    ax.set_title("LoS and NLoS localization performance")
    ax.legend(frameon=False, ncol=2)
    fig.subplots_adjust(bottom=0.29)
    _source_note(fig, "metrics.csv / per_sample_predictions.csv")
    return fig, source


def _ablation_figure(
    bundle: ResultBundle, data_dir: Path
) -> tuple[plt.Figure, pd.DataFrame]:
    bundle.require("ablation")
    rows = bundle.ablation.dropna(subset=["variant", "mean_error"]).copy()
    # Kalman on/off rows are evaluated on continuous trajectories, whereas
    # feature ablations use the spatial holdout.  Combining their absolute
    # errors in one ranking would conflate filtering with point localization.
    kalman_rows = rows["variant"].astype(str).str.contains(
        "kalman", case=False, regex=False
    )
    rows = rows[~kalman_rows].copy()
    _ensure_nonempty(rows, "ablation comparison")
    if len(rows["variant"].unique()) < 2:
        raise VisualDataError(
            "ablation_results requires at least two non-Kalman feature "
            "variants evaluated on the point-localization protocol."
        )
    aggregate = (
        rows.groupby("variant", as_index=False)
        .agg(
            mean_error=("mean_error", "mean"),
            std_error=("mean_error", "std"),
            seed_count=("mean_error", "count"),
        )
        .sort_values("mean_error", ascending=True)
    )
    y = np.arange(len(aggregate))
    colors = [
        (
            OKABE_ITO["bluish_green"]
            if any(
                token in str(variant).lower()
                for token in ("full", "complete", "residual ai")
            )
            else OKABE_ITO["sky_blue"]
        )
        for variant in aggregate["variant"]
    ]
    fig_height = max(2.75, 0.38 * len(aggregate) + 1.2)
    fig, ax = plt.subplots(figsize=(7.16, fig_height))
    error = (
        aggregate["std_error"].fillna(0.0)
        if aggregate["seed_count"].max() > 1
        else None
    )
    ax.barh(
        y,
        aggregate["mean_error"],
        xerr=error,
        color=colors,
        edgecolor=NEUTRAL["ink"],
        linewidth=0.45,
        capsize=2.0,
    )
    display_labels = {
        "full": "Full model",
        "without_los_nlos": "Without LoS/NLoS features",
        "without_geometric_residual": "Without residual-cost feature",
        "without_anchor_mask": "Without anchor mask",
        "without_spatial_bias_training": "Without spatial-bias training",
    }
    labels = [
        textwrap.fill(
            display_labels.get(str(label), str(label).replace("_", " ")),
            width=32,
        )
        for label in aggregate["variant"].tolist()
    ]
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Mean Euclidean error (m), lower is better")
    ax.set_title("Residual-model ablation study")
    for index, value in enumerate(aggregate["mean_error"]):
        ax.text(
            float(value),
            index,
            f"  {float(value):.2f}",
            va="center",
            ha="left",
            fontsize=7.0,
        )
    fig.subplots_adjust(left=0.34, bottom=0.14)
    _source_note(fig, "ablation_results.csv")
    return fig, rows


def _runtime_figure(
    bundle: ResultBundle, data_dir: Path
) -> tuple[plt.Figure, pd.DataFrame]:
    bundle.require("runtime")
    rows = bundle.runtime.copy()
    numeric_columns = [
        column
        for column in (
            "inference_time_ms",
            "training_time_s",
            "model_size_mb",
            "dashboard_update_ms",
        )
        if column in rows
    ]
    if "inference_time_ms" not in numeric_columns:
        raise VisualDataError(
            "runtime_results.csv does not contain inference_time_ms."
        )
    aggregate = (
        rows.groupby("algorithm", as_index=False)[numeric_columns]
        .mean(numeric_only=True)
    )
    algorithms = ordered_algorithms(aggregate["algorithm"])
    aggregate = aggregate.set_index("algorithm").reindex(algorithms).reset_index()
    x = np.arange(len(aggregate))
    colors = [algorithm_color(name) for name in algorithms]

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.8))
    axes[0].bar(
        x,
        aggregate["inference_time_ms"],
        color=colors,
        edgecolor=NEUTRAL["ink"],
        linewidth=0.45,
    )
    axes[0].set_xticks(x, algorithms, rotation=24, ha="right")
    axes[0].set_ylabel("Inference latency (ms/sample)")
    axes[0].set_title("(a) Online inference")
    if (aggregate["inference_time_ms"] > 0).all():
        ratio = (
            float(aggregate["inference_time_ms"].max())
            / float(aggregate["inference_time_ms"].min())
        )
        if ratio > 25:
            axes[0].set_yscale("log")

    secondary = (
        "training_time_s"
        if "training_time_s" in aggregate
        and aggregate["training_time_s"].notna().any()
        else (
            "model_size_mb"
            if "model_size_mb" in aggregate
            and aggregate["model_size_mb"].notna().any()
            else None
        )
    )
    if secondary is None:
        raise VisualDataError(
            "runtime_comparison additionally requires training_time_s or "
            "model_size_mb."
        )
    axes[1].bar(
        x,
        aggregate[secondary],
        color=colors,
        edgecolor=NEUTRAL["ink"],
        linewidth=0.45,
        hatch="//",
    )
    axes[1].set_xticks(x, algorithms, rotation=24, ha="right")
    axes[1].set_ylabel(
        "Training time (s)"
        if secondary == "training_time_s"
        else "Serialized model size (MB)"
    )
    axes[1].set_title(
        "(b) Offline training"
        if secondary == "training_time_s"
        else "(b) Model footprint"
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.9, bottom=0.31, wspace=0.29)
    _source_note(fig, "runtime_results.csv")
    return fig, rows


_BUILDERS: dict[
    str, Callable[[ResultBundle, Path], tuple[plt.Figure, pd.DataFrame]]
] = {
    "trajectory_comparison": _trajectory_figure,
    "error_cdf": _cdf_figure,
    "spatial_error_heatmaps": _spatial_heatmap_figure,
    "robustness_noise": _robustness_noise_figure,
    "robustness_anchor_failure": _robustness_anchor_figure,
    "robustness_results": _robustness_combined_figure,
    "los_nlos_comparison": _los_nlos_figure,
    "ablation_results": _ablation_figure,
    "runtime_comparison": _runtime_figure,
}


def export_publication_figures(
    bundle: ResultBundle,
    output_dir: str | Path,
    data_dir: str | Path,
) -> list[Path]:
    """Export every required report figure as PDF and 300+ DPI PNG."""

    configure_paper_style()
    figure_root = Path(output_dir)
    source_root = Path(data_dir)
    exported: list[Path] = []
    for stem in FIGURE_STEMS:
        builder = _BUILDERS[stem]
        fig, source = builder(bundle, source_root)
        _save_source(source, source_root, stem)
        exported.extend(_save_pair(fig, figure_root, stem))
    return exported
