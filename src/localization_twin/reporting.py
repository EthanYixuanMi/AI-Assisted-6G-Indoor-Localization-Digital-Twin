"""Export traceable CSV, LaTeX, and narrative report assets.

Every number written by this module is read from a completed experiment CSV.
The module deliberately refuses empty inputs so a failed experiment cannot be
mistaken for a successful report export.
"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml


METHOD_ORDER = [
    "Geometric LS",
    "KNN",
    "Direct AI",
    "Residual AI",
    "Residual AI + Kalman",
]


def _require_csv(path: Path, columns: Iterable[str]) -> pd.DataFrame:
    """Read a non-empty CSV and validate its required columns."""

    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Required non-empty result file is missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Result file contains no rows: {path}")
    missing = set(columns).difference(frame.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
    return frame


def _format_number(value: Any, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "--"
    return f"{float(value):.{decimals}f}"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in text)


def _best_indices(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, int]:
    best: dict[str, int] = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.notna().any():
            best[column] = int(numeric.idxmin())
    return best


def _latex_table(
    frame: pd.DataFrame,
    *,
    headers: Sequence[str],
    columns: Sequence[str],
    caption: str,
    label: str,
    numeric_columns: Sequence[str] = (),
    best_columns: Sequence[str] = (),
    full_width: bool = False,
    note: str | None = None,
) -> str:
    """Render an IEEE-compatible booktabs table with conservative bolding."""

    if frame.empty:
        raise ValueError(f"Cannot render empty LaTeX table {label}")
    environment = "table*" if full_width else "table"
    align = "l" + "r" * (len(columns) - 1)
    best = _best_indices(frame, best_columns)
    lines = [
        rf"\begin{{{environment}}}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\small",
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    for index, row in frame.iterrows():
        cells: list[str] = []
        for column in columns:
            value = row[column]
            if column in numeric_columns:
                rendered = _format_number(value)
                if best.get(column) == index and rendered != "--":
                    rendered = rf"\textbf{{{rendered}}}"
            else:
                rendered = _latex_escape(value)
            cells.append(rendered)
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    if note:
        lines.append(rf"\par\smallskip\footnotesize\textit{{Note:}} {note}")
    lines.append(rf"\end{{{environment}}}")
    return "\n".join(lines) + "\n"


def _flatten_parameters(config: Mapping[str, Any]) -> pd.DataFrame:
    environment = config["environment"]
    propagation = config["propagation"]
    sampling = config["sampling"]
    path_loss_values = [
        float(anchor.get("path_loss_exponent", propagation["default_path_loss_exponent"]))
        for anchor in environment["anchors"]
    ]
    path_loss_display = (
        f"{min(path_loss_values):.2f}--{max(path_loss_values):.2f}"
        if min(path_loss_values) != max(path_loss_values)
        else f"{path_loss_values[0]:.2f}"
    )
    rows = [
        ("Area size", f"{environment['width']} x {environment['height']}", "m"),
        ("Number of anchors", len(environment["anchors"]), "count"),
        ("Path-loss exponent", path_loss_display, "dimensionless"),
        ("Reference distance", propagation.get("reference_distance", 1.0), "m"),
        ("RSS noise standard deviation", propagation["noise_std"], "dB"),
        ("NLoS bias mean", propagation["nlos_bias_mean"], "dB"),
        ("Anchor dropout probability", propagation.get("dropout_probability", 0.0), "ratio"),
        ("Training samples", sampling["train_count"], "samples"),
        ("Validation samples", sampling["validation_count"], "samples"),
        ("In-domain test samples", sampling["test_count"], "samples"),
        ("Spatial holdout samples", sampling["spatial_holdout_count"], "samples"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "unit"])


def _copy_csv(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)


def _main_results(metrics: pd.DataFrame) -> pd.DataFrame:
    normal = metrics.loc[metrics["scenario"].str.lower() == "normal"].copy()
    if normal.empty:
        raise ValueError("metrics.csv has no Normal scenario rows")

    def weighted_mean(
        frame: pd.DataFrame, value_column: str, weight_column: str
    ) -> float:
        values = pd.to_numeric(frame[value_column], errors="coerce")
        if weight_column not in frame:
            return float(values.mean())
        weights = pd.to_numeric(frame[weight_column], errors="coerce")
        valid = values.notna() & weights.notna() & (weights > 0)
        if not valid.any():
            return float(values.mean())
        return float(np.average(values[valid], weights=weights[valid]))

    records: list[dict[str, Any]] = []
    for algorithm, group in normal.groupby("algorithm", sort=False):
        counts = pd.to_numeric(group["sample_count"], errors="coerce").fillna(0.0)
        total_count = int(counts.sum())
        if total_count <= 0:
            raise ValueError(f"Normal rows for {algorithm} have no evaluated samples")
        rmse_values = pd.to_numeric(group["rmse"], errors="coerce")
        valid_rmse = rmse_values.notna() & (counts > 0)
        pooled_rmse = float(
            np.sqrt(
                np.average(
                    np.square(rmse_values[valid_rmse]),
                    weights=counts[valid_rmse],
                )
            )
        )
        records.append(
            {
                "method": algorithm,
                "mean_error": weighted_mean(group, "mean_error", "sample_count"),
                "mean_error_std": float(
                    pd.to_numeric(group["mean_error"], errors="coerce").std(ddof=1)
                    if len(group) > 1
                    else 0.0
                ),
                "rmse": pooled_rmse,
                "rmse_std": float(
                    pd.to_numeric(group["rmse"], errors="coerce").std(ddof=1)
                    if len(group) > 1
                    else 0.0
                ),
                # Exact pooled quantiles require all per-sample predictions for
                # every seed.  The independent-run table therefore reports the
                # mean and dispersion of each seed's quantile explicitly.
                "median_error": float(
                    pd.to_numeric(group["median_error"], errors="coerce").mean()
                ),
                "median_error_std": float(
                    pd.to_numeric(group["median_error"], errors="coerce").std(ddof=1)
                    if len(group) > 1
                    else 0.0
                ),
                "p90_error": float(
                    pd.to_numeric(group["p90_error"], errors="coerce").mean()
                ),
                "p90_error_std": float(
                    pd.to_numeric(group["p90_error"], errors="coerce").std(ddof=1)
                    if len(group) > 1
                    else 0.0
                ),
                "max_error": float(
                    pd.to_numeric(group["max_error"], errors="coerce").max()
                ),
                "los_mean_error": weighted_mean(
                    group, "los_mean_error", "los_sample_count"
                ),
                "nlos_mean_error": weighted_mean(
                    group, "nlos_mean_error", "nlos_sample_count"
                ),
                "inference_time_ms": float(
                    pd.to_numeric(group["inference_time_ms"], errors="coerce").mean()
                ),
                "training_time_s": float(
                    pd.to_numeric(group["training_time_s"], errors="coerce").mean()
                ),
                "model_size_mb": float(
                    pd.to_numeric(group["model_size_mb"], errors="coerce").mean()
                ),
                "sample_count": total_count,
                "seed_count": int(group["seed"].nunique()),
                "aggregation": (
                    "independent-seed aggregate; mean/RMSE are sample-weighted, "
                    "median/P90 are means of per-seed quantiles, max is the "
                    "maximum across seeds"
                ),
            }
        )
    grouped = pd.DataFrame(records)
    order = {name: index for index, name in enumerate(METHOD_ORDER)}
    grouped["_order"] = grouped["method"].map(order).fillna(len(order))
    grouped = grouped.sort_values(["_order", "method"]).drop(columns="_order").reset_index(drop=True)
    geometric = grouped.loc[grouped["method"] == "Geometric LS", "mean_error"]
    if geometric.empty:
        raise ValueError("Geometric LS baseline is absent from metrics.csv")
    baseline = float(geometric.iloc[0])
    grouped["mean_error_geometric_baseline"] = baseline
    grouped["mean_error_change_vs_geometric"] = grouped["mean_error"] - baseline
    grouped["mean_error_absolute_improvement"] = baseline - grouped["mean_error"]
    grouped["improvement_vs_geometric_pct"] = (
        100.0 * (baseline - grouped["mean_error"]) / baseline
    )
    return grouped


def _los_nlos_table(metrics: pd.DataFrame) -> pd.DataFrame:
    normal = metrics.loc[metrics["scenario"].str.lower() == "normal"].copy()
    if normal.empty:
        raise ValueError("Cannot export LoS/NLoS table without Normal scenario rows")
    aggregations: dict[str, tuple[str, str]] = {
        "los_mean_error": ("los_mean_error", "mean"),
        "los_median_error": ("los_median_error", "mean"),
        "nlos_mean_error": ("nlos_mean_error", "mean"),
        "nlos_median_error": ("nlos_median_error", "mean"),
        "nlos_p90_error": ("nlos_p90_error", "mean"),
    }
    available = {
        output: source_agg
        for output, source_agg in aggregations.items()
        if source_agg[0] in normal.columns
    }
    result = normal.groupby("algorithm", as_index=False).agg(**available)
    result = result.rename(columns={"algorithm": "method"})
    result["seed_count"] = int(normal["seed"].nunique())
    if "los_mean_error" in result and "nlos_mean_error" in result:
        result["nlos_degradation_pct"] = (
            100.0
            * (result["nlos_mean_error"] - result["los_mean_error"])
            / result["los_mean_error"].replace(0, float("nan"))
        )
    return result


def _write_latex_assets(
    latex_dir: Path,
    parameters: pd.DataFrame,
    main: pd.DataFrame,
    los_nlos: pd.DataFrame,
    robustness: pd.DataFrame,
    ablation: pd.DataFrame,
    runtime: pd.DataFrame,
) -> None:
    latex_dir.mkdir(parents=True, exist_ok=True)
    (latex_dir / "table_simulation_parameters.tex").write_text(
        _latex_table(
            parameters,
            headers=("Parameter", "Value", "Unit"),
            columns=("parameter", "value", "unit"),
            caption="Digital-twin simulation parameters used for the reported run.",
            label="tab:parameters",
        ),
        encoding="utf-8",
    )
    (latex_dir / "table_main_results.tex").write_text(
        _latex_table(
            main,
            headers=(
                "Method",
                "Mean (m)",
                "RMSE (m)",
                "Median (m)",
                "P90 (m)",
                "NLoS (m)",
                "Time (ms)",
            ),
            columns=(
                "method",
                "mean_error",
                "rmse",
                "median_error",
                "p90_error",
                "nlos_mean_error",
                "inference_time_ms",
            ),
            caption=(
                "Localization accuracy over independent simulation/training seeds "
                "and warmed one-row inference time in the simulated Normal spatial "
                "holdout. Lower is better for every numeric column."
            ),
            label="tab:main_results",
            numeric_columns=(
                "mean_error",
                "rmse",
                "median_error",
                "p90_error",
                "nlos_mean_error",
                "inference_time_ms",
            ),
            best_columns=(
                "mean_error",
                "rmse",
                "median_error",
                "p90_error",
                "nlos_mean_error",
                "inference_time_ms",
            ),
            full_width=True,
            note=(
                "Mean and RMSE are sample-weighted across seeds; median and P90 are "
                "means of per-seed quantiles. All values are software simulation "
                "results; bold marks the minimum in a column."
            ),
        ),
        encoding="utf-8",
    )
    los_columns = [
        column
        for column in (
            "method",
            "los_mean_error",
            "los_median_error",
            "nlos_mean_error",
            "nlos_median_error",
            "nlos_p90_error",
            "nlos_degradation_pct",
        )
        if column in los_nlos.columns
    ]
    header_map = {
        "method": "Method",
        "los_mean_error": "LoS Mean (m)",
        "los_median_error": "LoS Med. (m)",
        "nlos_mean_error": "NLoS Mean (m)",
        "nlos_median_error": "NLoS Med. (m)",
        "nlos_p90_error": "NLoS P90 (m)",
        "nlos_degradation_pct": "Degradation (\\%)",
    }
    (latex_dir / "table_los_nlos.tex").write_text(
        _latex_table(
            los_nlos,
            headers=tuple(header_map[column] for column in los_columns),
            columns=tuple(los_columns),
            caption="Localization error stratified by simulated LoS and NLoS status.",
            label="tab:los_nlos",
            numeric_columns=tuple(los_columns[1:]),
            best_columns=tuple(column for column in los_columns[1:] if column != "nlos_degradation_pct"),
            full_width=True,
            note="A sample is NLoS when at least one available anchor path intersects a wall or obstacle.",
        ),
        encoding="utf-8",
    )

    robustness_view = robustness.copy()
    modes = robustness_view.get(
        "failure_mode", pd.Series("not_applicable", index=robustness_view.index)
    ).astype(str)
    robustness_view["condition"] = np.where(
        robustness_view["experiment"].eq("anchor_failure"),
        (
            robustness_view["experiment"].astype(str)
            + "/"
            + modes
            + ": "
            + robustness_view["level"].astype(str)
        ),
        (
            robustness_view["experiment"].astype(str)
            + ": "
            + robustness_view["level"].astype(str)
        ),
    )
    improvement_column = (
        "mean_error_percent_improvement"
        if "mean_error_percent_improvement" in robustness_view
        else "percent_improvement"
    )
    robustness_view = (
        robustness_view.groupby(["algorithm", "condition"], as_index=False)
        .agg(
            mean_error=("mean_error", "mean"),
            p90_error=("p90_error", "mean"),
            improvement_pct=(improvement_column, "mean"),
            seed_count=("seed", "nunique"),
        )
        .rename(columns={"algorithm": "method"})
    )
    (latex_dir / "table_robustness.tex").write_text(
        _latex_table(
            robustness_view,
            headers=("Method", "Condition", "Mean (m)", "P90 (m)", "Improve (\\%)"),
            columns=(
                "method",
                "condition",
                "mean_error",
                "p90_error",
                "improvement_pct",
            ),
            caption=(
                "Robustness under simulated RSS noise, wall/NLoS severity, and "
                "random or fixed-critical anchor failures."
            ),
            label="tab:robustness",
            numeric_columns=("mean_error", "p90_error", "improvement_pct"),
            full_width=True,
            note=(
                "Improvement is relative to Geometric LS in the same condition; "
                "negative values are retained and indicate degradation."
            ),
        ),
        encoding="utf-8",
    )

    spatial_ablation = ablation.loc[
        ablation["evaluation_split"].astype(str).eq("spatial_holdout")
        & ~ablation["variant"].astype(str).str.contains(
            "kalman", case=False, na=False
        )
    ].copy()
    if spatial_ablation.empty:
        raise ValueError("No spatial feature-ablation rows are available")
    ablation_grouped = (
        spatial_ablation.groupby("variant", as_index=False)
        .agg(
            mean_error=("mean_error", "mean"),
            rmse=("rmse", "mean"),
            p90_error=("p90_error", "mean"),
        )
    )
    (latex_dir / "table_ablation.tex").write_text(
        _latex_table(
            ablation_grouped,
            headers=("Variant", "Mean (m)", "RMSE (m)", "P90 (m)"),
            columns=("variant", "mean_error", "rmse", "p90_error"),
            caption="Residual-model feature ablation in the simulated test environment.",
            label="tab:ablation",
            numeric_columns=("mean_error", "rmse", "p90_error"),
            best_columns=("mean_error", "rmse", "p90_error"),
            note=(
                "Only spatial-holdout feature/training variants are compared; "
                "trajectory filtering is reported separately."
            ),
        ),
        encoding="utf-8",
    )

    trajectory_filtering = ablation.loc[
        ablation["evaluation_split"].astype(str).eq("trajectories")
    ].copy()
    if trajectory_filtering.empty:
        raise ValueError("No trajectory-filtering ablation rows are available")
    trajectory_grouped = (
        trajectory_filtering.groupby(["algorithm", "variant"], as_index=False)
        .agg(
            mean_error=("mean_error", "mean"),
            rmse=("rmse", "mean"),
            p90_error=("p90_error", "mean"),
        )
        .rename(columns={"algorithm": "method"})
    )
    (latex_dir / "table_trajectory_filtering.tex").write_text(
        _latex_table(
            trajectory_grouped,
            headers=("Method", "Filter variant", "Mean (m)", "RMSE (m)", "P90 (m)"),
            columns=("method", "variant", "mean_error", "rmse", "p90_error"),
            caption=(
                "Kalman filtering comparison on the same simulated trajectory "
                "measurements; these rows are not mixed with point localization."
            ),
            label="tab:trajectory_filtering",
            numeric_columns=("mean_error", "rmse", "p90_error"),
            best_columns=("mean_error", "rmse", "p90_error"),
            full_width=True,
        ),
        encoding="utf-8",
    )

    runtime_grouped = (
        runtime.groupby("algorithm", as_index=False)
        .agg(
            training_time_s=("training_time_s", "mean"),
            inference_time_ms=("inference_time_ms", "mean"),
            batch_inference_time_ms=("batch_inference_time_ms", "mean"),
            batch_amortized_time_ms=("batch_amortized_time_ms", "mean"),
            model_size_mb=("model_size_mb", "mean"),
            dashboard_update_ms=("dashboard_update_ms", "mean"),
        )
        .rename(columns={"algorithm": "method"})
    )
    (latex_dir / "table_runtime.tex").write_text(
        _latex_table(
            runtime_grouped,
            headers=(
                "Method",
                "Train (s)",
                "One row (ms)",
                "Batch (ms)",
                "Amort. (ms)",
                "Size (MB)",
                "Data prep (ms)",
            ),
            columns=(
                "method",
                "training_time_s",
                "inference_time_ms",
                "batch_inference_time_ms",
                "batch_amortized_time_ms",
                "model_size_mb",
                "dashboard_update_ms",
            ),
            caption=(
                "CPU-only training, warmed one-row inference, batch throughput, "
                "storage, and saved-frame dashboard data-preparation costs."
            ),
            label="tab:runtime",
            numeric_columns=(
                "training_time_s",
                "inference_time_ms",
                "batch_inference_time_ms",
                "batch_amortized_time_ms",
                "model_size_mb",
                "dashboard_update_ms",
            ),
            best_columns=(
                "training_time_s",
                "inference_time_ms",
                "batch_inference_time_ms",
                "batch_amortized_time_ms",
                "model_size_mb",
                "dashboard_update_ms",
            ),
            full_width=True,
            note=(
                "Times are local medians after prediction warm-up and are hardware "
                "dependent. Data prep selects a saved frame and computes KPIs; it "
                "excludes Streamlit and Plotly rendering."
            ),
        ),
        encoding="utf-8",
    )


def _select_algorithm_row(main: pd.DataFrame, method: str) -> pd.Series:
    selected = main.loc[main["method"] == method]
    if selected.empty:
        raise ValueError(f"Required method is absent from main results: {method}")
    return selected.iloc[0]


def _narrative_assets(
    assets_dir: Path,
    main: pd.DataFrame,
    los_nlos: pd.DataFrame,
    robustness: pd.DataFrame,
    runtime: pd.DataFrame,
) -> None:
    geometric = _select_algorithm_row(main, "Geometric LS")
    residual = _select_algorithm_row(main, "Residual AI")
    best = main.loc[main["mean_error"].idxmin()]
    improvement = 100.0 * (
        float(geometric["mean_error"]) - float(residual["mean_error"])
    ) / float(geometric["mean_error"])
    direction = "reduced" if improvement >= 0 else "increased"
    magnitude = abs(improvement)

    best_los = los_nlos.loc[los_nlos["los_mean_error"].idxmin()]
    best_nlos = los_nlos.loc[los_nlos["nlos_mean_error"].idxmin()]
    robust_aggregate = (
        robustness.groupby(
            ["experiment", "failure_mode", "level", "algorithm"],
            as_index=False,
            dropna=False,
        )
        .agg(
            mean_error=("mean_error", "mean"),
            p90_error=("p90_error", "mean"),
            seed_count=("seed", "nunique"),
        )
    )
    noise = robust_aggregate.loc[
        robust_aggregate["experiment"] == "noise"
    ].copy()
    dropout = robust_aggregate.loc[
        robust_aggregate["experiment"] == "anchor_failure"
    ].copy()
    if noise.empty or dropout.empty:
        raise ValueError("Robustness results require both noise and anchor_failure rows")
    noise["numeric_level"] = pd.to_numeric(noise["level"], errors="coerce")
    dropout["numeric_level"] = pd.to_numeric(dropout["level"], errors="coerce")
    maximum_noise = float(noise["numeric_level"].max())
    hardest_noise = noise.loc[
        noise["numeric_level"].eq(maximum_noise)
    ].sort_values("mean_error").iloc[0]
    dropout_findings: list[str] = []
    for failure_mode, mode_rows in dropout.groupby("failure_mode"):
        maximum_failure = float(mode_rows["numeric_level"].max())
        hardest_mode = mode_rows.loc[
            mode_rows["numeric_level"].eq(maximum_failure)
        ].sort_values("mean_error").iloc[0]
        dropout_findings.append(
            f"For {failure_mode} failures at {int(maximum_failure)} unavailable "
            f"anchors, {hardest_mode['algorithm']} had the lowest seed-averaged "
            f"mean error ({hardest_mode['mean_error']:.2f} m)"
        )
    dropout_text = "; ".join(dropout_findings) + "."

    best_runtime = runtime.loc[runtime["inference_time_ms"].idxmin()]
    draft = f"""# Preliminary Results Draft

All quantitative values in this section were generated by the local,
software-only digital-twin simulator. They describe the configured synthetic
environment and do not establish accuracy in a physical 6G deployment.

## 1. Overall Localization Accuracy

Across independent simulation and training seeds in the Normal spatial holdout,
**{best['method']}** achieved the lowest
mean Euclidean localization error of **{best['mean_error']:.2f} m**, with an
RMSE of **{best['rmse']:.2f} m**, a median error of
**{best['median_error']:.2f} m**, and a 90th-percentile error of
**{best['p90_error']:.2f} m**. The Geometric LS baseline obtained a mean error
of **{geometric['mean_error']:.2f} m**. Relative to that baseline, Residual AI
{direction} the mean error by **{magnitude:.1f}%**
({geometric['mean_error']:.2f} m versus {residual['mean_error']:.2f} m).
This comparison is conditional on the spatial holdout protocol and the
simulator parameters saved with the run.

## 2. LoS and NLoS Performance

The lowest simulated LoS mean error was produced by
**{best_los['method']}** at **{best_los['los_mean_error']:.2f} m**. Under the
NLoS definition used by the twin, **{best_nlos['method']}** obtained the lowest
mean error of **{best_nlos['nlos_mean_error']:.2f} m**. The separation between
LoS and NLoS results confirms that wall-induced, non-zero bias is a materially
harder condition than unblocked propagation in this simulation. These labels
are derived from geometric path intersections rather than measured radio
ground truth.

## 3. Robustness to Noise and Anchor Failure

The robustness sweep retained every tested method even when performance
degraded. At the largest tested RSS-noise level
(**{hardest_noise['level']} dB**), the lowest seed-averaged mean error was
**{hardest_noise['mean_error']:.2f} m** for
**{hardest_noise['algorithm']}**, with a P90 error of
**{hardest_noise['p90_error']:.2f} m**. {dropout_text} The domain-shift rows in
`metrics.csv` should be interpreted as a stress test of simulator mismatch, not
as evidence of transfer to a real building.

## 4. Runtime and Demonstration Performance

After warm-up, the lowest measured one-row inference time was
**{best_runtime['inference_time_ms']:.3f} ms** for
**{best_runtime['algorithm']}** on the execution machine. Model training time,
batch prediction time, amortized batch throughput, serialized size, and the
saved-frame data-preparation proxy are reported separately in
`table_runtime.tex`. The proxy excludes Streamlit and Plotly rendering. These
measurements are hardware- and software-version dependent.
"""
    (assets_dir / "REPORT_RESULTS_DRAFT.md").write_text(draft, encoding="utf-8")

    snippet = r"""\section{Preliminary Results}
All values reported below were generated by the software-only indoor
localization digital twin. They characterize the configured simulation and do
not constitute measurements from a physical 6G system.

\subsection{Overall Localization Accuracy}
Table~\ref{tab:main_results} summarizes point-localization accuracy and
inference latency. Figure~\ref{fig:error_cdf} complements the summary with the
full empirical error distribution, while Fig.~\ref{fig:trajectory} illustrates
the corresponding trajectory behavior.
\input{report_assets/latex/table_main_results.tex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{report_assets/figures/error_cdf.pdf}
\caption{Empirical CDF of Euclidean localization error in the simulated test set.}
\label{fig:error_cdf}
\end{figure}
\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{report_assets/figures/trajectory_comparison.pdf}
\caption{Ground-truth and estimated trajectories under the saved simulation configuration.}
\label{fig:trajectory}
\end{figure*}

\subsection{LoS and NLoS Performance}
Table~\ref{tab:los_nlos} and Fig.~\ref{fig:los_nlos} separate errors using
wall-intersection labels produced by the environment twin.
\input{report_assets/latex/table_los_nlos.tex}
\begin{figure}[t]
\centering
\includegraphics[width=\columnwidth]{report_assets/figures/los_nlos_comparison.pdf}
\caption{Localization error under simulated LoS and NLoS conditions.}
\label{fig:los_nlos}
\end{figure}

\subsection{Robustness to Noise and Anchor Failure}
Figure~\ref{fig:robustness} and Table~\ref{tab:robustness} report controlled
noise and anchor-failure sweeps. Negative improvements are retained.
\input{report_assets/latex/table_robustness.tex}
\begin{figure*}[t]
\centering
\includegraphics[width=0.95\textwidth]{report_assets/figures/robustness_results.pdf}
\caption{Robustness to simulated RSS noise and anchor unavailability.}
\label{fig:robustness}
\end{figure*}

\subsection{Runtime and Demonstration Performance}
Table~\ref{tab:runtime} reports CPU-only training, warmed-up prediction,
serialized model size, and saved-frame data preparation. The latter excludes
Streamlit and Plotly rendering.
\input{report_assets/latex/table_runtime.tex}

\subsection{Trajectory Filtering}
Table~\ref{tab:trajectory_filtering} compares Residual AI before and after
Kalman filtering on the same trajectory measurements. It is intentionally
separate from the point-localization feature ablation.
\input{report_assets/latex/table_trajectory_filtering.tex}
"""
    (assets_dir / "latex" / "preliminary_results_snippet.tex").write_text(
        snippet, encoding="utf-8"
    )

    residual_claim = (
        f"Residual AI 相对 Geometric LS 的平均误差改善为 {improvement:.1f}%"
        if improvement >= 0
        else f"Residual AI 相对 Geometric LS 的平均误差恶化为 {abs(improvement):.1f}%"
    )
    summary = f"""# 结果摘要

本摘要只解释本地软件仿真的实测输出，不代表真实 6G 网络或真实建筑部署。

## 最重要的发现

- 独立仿真与训练 seed 的 Normal 空间留出结果中，平均误差最低的是 **{best['method']}**：
  **{best['mean_error']:.2f} m**，P90 为 **{best['p90_error']:.2f} m**。
- {residual_claim}；该结论仅适用于保存的配置和空间留出协议。
- LoS 条件下最优方法为 **{best_los['method']}**
  （{best_los['los_mean_error']:.2f} m），NLoS 条件下最优方法为
  **{best_nlos['method']}**（{best_nlos['nlos_mean_error']:.2f} m）。
- 推理最快的是 **{best_runtime['algorithm']}**，本机预热后一行输入时间为
  **{best_runtime['inference_time_ms']:.3f} ms**。

## 稳定性与限制

噪声、墙体偏差、锚点掉线和 domain shift 会改变方法排序，因此不能从
Normal 场景推导“某方法始终最好”。模型是在同一类简化 RSS 数字孪生中训练
和测试的；空间留出可以降低位置泄漏，却不能消除 simulation-to-reality gap。

## 可以写进报告的结论

可以报告各方法在本次仿真中的绝对误差、相对 Geometric LS 的变化、LoS/NLoS
差异、鲁棒性曲线以及本机 CPU 延迟。必须同时写明配置、seed、样本量和仿真
边界。

## 不能写的结论

不能声称已实现真实 6G 系统、真实厘米级定位、标准 3GPP 信道验证或真实建筑
泛化；本项目没有相应硬件和实测证据。

## 展示建议

先用 `dashboard_normal.png` 解释数字孪生，再展示
`trajectory_comparison.png` 与 `error_cdf.png`，随后切换
`dashboard_strong_blockage.png` 和 `dashboard_anchor_failure.png`，最后用
`robustness_results.png` 与 `runtime_comparison.png` 收束结论。
"""
    (assets_dir / "RESULTS_SUMMARY.md").write_text(summary, encoding="utf-8")


def export_report_tables_and_text(
    results_dir: Path,
    report_assets_dir: Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> list[Path]:
    """Export all required report tables and evidence-bounded prose."""

    results_dir = Path(results_dir)
    report_assets_dir = Path(report_assets_dir)
    tables_dir = report_assets_dir / "tables"
    latex_dir = report_assets_dir / "latex"
    data_dir = report_assets_dir / "data"
    for directory in (tables_dir, latex_dir, data_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metrics = _require_csv(
        results_dir / "metrics.csv",
        (
            "algorithm",
            "scenario",
            "seed",
            "sample_count",
            "mean_error",
            "rmse",
            "median_error",
            "p90_error",
            "max_error",
            "los_mean_error",
            "nlos_mean_error",
            "inference_time_ms",
            "training_time_s",
            "model_size_mb",
        ),
    )
    robustness = _require_csv(
        results_dir / "robustness_results.csv",
        (
            "experiment",
            "algorithm",
            "scenario",
            "seed",
            "measurement_seed",
            "model_training_seed",
            "seed_scope",
            "level",
            "failure_mode",
            "evaluation_split",
            "mean_error",
            "p90_error",
            "los_mean_error",
            "nlos_mean_error",
            "mean_error_percent_improvement",
            "inference_time_ms",
            "training_time_s",
            "model_size_mb",
        ),
    )
    ablation = _require_csv(
        results_dir / "ablation_results.csv",
        (
            "variant",
            "algorithm",
            "scenario",
            "evaluation_split",
            "seed",
            "sample_count",
            "mean_error",
            "rmse",
            "p90_error",
            "mean_error_geometric_baseline",
            "mean_error_change_vs_geometric",
            "mean_error_absolute_improvement",
            "mean_error_percent_improvement",
            "inference_time_ms",
            "training_time_s",
            "model_size_mb",
        ),
    )
    runtime = _require_csv(
        results_dir / "runtime_results.csv",
        (
            "algorithm",
            "training_time_s",
            "inference_time_ms",
            "batch_inference_time_ms",
            "batch_amortized_time_ms",
            "model_size_mb",
            "dashboard_update_ms",
            "dashboard_update_scope",
        ),
    )
    per_seed = _require_csv(
        results_dir / "per_seed_results.csv",
        ("algorithm", "scenario", "seed", "mean_error", "rmse", "p90_error"),
    )

    if config is None:
        config_path = results_dir / "config_resolved.yaml"
        if not config_path.is_file():
            raise FileNotFoundError(f"Resolved config is missing: {config_path}")
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    parameters = _flatten_parameters(config)
    # The headline tables aggregate the configured evaluation seeds.  The
    # scenario-wide metrics file remains the source for visual comparisons,
    # while per_seed_results.csv prevents a single favourable draw from
    # determining the report narrative.
    main = _main_results(per_seed)
    los_nlos = _los_nlos_table(per_seed)

    table_frames = {
        "simulation_parameters.csv": parameters,
        "main_results.csv": main,
        "los_nlos_results.csv": los_nlos,
        "robustness_results.csv": robustness,
        "ablation_results.csv": ablation,
        "trajectory_filtering_results.csv": ablation.loc[
            ablation["evaluation_split"].astype(str).eq("trajectories")
        ].copy(),
        "runtime_results.csv": runtime,
        "per_seed_results.csv": per_seed,
    }
    generated: list[Path] = []
    for name, frame in table_frames.items():
        path = tables_dir / name
        _copy_csv(frame, path)
        generated.append(path)

    for source in (
        "metrics.csv",
        "per_sample_predictions.csv",
        "per_seed_results.csv",
        "robustness_results.csv",
        "ablation_results.csv",
        "runtime_results.csv",
    ):
        destination = data_dir / source
        shutil.copy2(results_dir / source, destination)
        generated.append(destination)

    _write_latex_assets(
        latex_dir, parameters, main, los_nlos, robustness, ablation, runtime
    )
    generated.extend(sorted(latex_dir.glob("table_*.tex")))
    _narrative_assets(report_assets_dir, main, los_nlos, robustness, runtime)
    generated.extend(
        [
            report_assets_dir / "REPORT_RESULTS_DRAFT.md",
            report_assets_dir / "RESULTS_SUMMARY.md",
            latex_dir / "preliminary_results_snippet.tex",
        ]
    )
    return generated


def build_manifest(
    run_dir: Path,
    *,
    profile: str,
    seed: int,
    config_path: str,
    data_counts: Mapping[str, int],
    algorithms: Sequence[str],
    started_at: str,
    finished_at: str,
    duration_s: float,
    test_status: Mapping[str, Any],
    success: bool,
) -> dict[str, Any]:
    """Create a recursive, size-aware experiment manifest."""

    import platform
    import subprocess
    import sys
    from importlib.metadata import PackageNotFoundError, version

    packages: dict[str, str] = {}
    for package in (
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "matplotlib",
        "streamlit",
        "PyYAML",
        "joblib",
    ):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = "not-installed"
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=run_dir.parents[0],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    files = []
    for path in sorted(run_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            files.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "size_bytes": path.stat().st_size,
                }
            )
    return {
        "success": success,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_s": duration_s,
        "profile": profile,
        "seed": seed,
        "config_file": config_path,
        "git_commit": git_commit,
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "data_counts": dict(data_counts),
        "algorithms": list(algorithms),
        "tests": dict(test_status),
        "files": files,
    }
