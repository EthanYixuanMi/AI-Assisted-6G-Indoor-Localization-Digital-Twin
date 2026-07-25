"""Professional Streamlit replay dashboard for saved localization experiments.

The app intentionally performs no model training.  It loads a completed run
from ``outputs/latest`` and replays its per-sample predictions.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from localization_twin.visualization.data import (  # noqa: E402
    ResultBundle,
    anchor_telemetry,
    infer_anchor_columns,
    select_scenario,
)
from localization_twin.visualization.style import (  # noqa: E402
    ALGORITHM_LINESTYLES,
    DARK,
    OKABE_ITO,
    SCENARIO_LABELS,
    SCENARIO_ORDER,
    algorithm_color,
    canonical_scenario,
    ordered_algorithms,
)


APP_TITLE = "AI-Assisted 6G Indoor Localization Digital Twin"
DEFAULT_RESULTS = ROOT / "outputs" / "latest"


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


def _query_scenario(st: Any) -> str | None:
    try:
        value = st.query_params.get("scenario")
    except AttributeError:
        values = st.experimental_get_query_params().get("scenario", [])
        value = values[0] if values else None
    if isinstance(value, list):
        value = value[0] if value else None
    return canonical_scenario(value) if value else None


def _current_rows(frame: pd.DataFrame, timestep: float) -> pd.DataFrame:
    records = []
    for _, rows in frame.groupby("algorithm", sort=False):
        index = (rows["timestep"] - timestep).abs().idxmin()
        records.append(rows.loc[index])
    return pd.DataFrame(records)


def _precomputed_numeric_control(
    st: Any,
    frame: pd.DataFrame,
    *,
    label: str,
    candidates: tuple[str, ...],
    unit: str,
    integer: bool = False,
) -> tuple[pd.DataFrame, float | int | None]:
    column = next((name for name in candidates if name in frame), None)
    if column is None:
        st.sidebar.text_input(
            label,
            value="not recorded in replay",
            disabled=True,
            help=(
                "This dashboard never fabricates or re-simulates controls. "
                "Record this field in per_sample_predictions.csv to filter it."
            ),
        )
        return frame, None
    values = sorted(
        pd.to_numeric(frame[column], errors="coerce").dropna().unique().tolist()
    )
    if not values:
        st.sidebar.text_input(label, value="no saved values", disabled=True)
        return frame, None
    display_values = [int(value) if integer else float(value) for value in values]
    if len(display_values) == 1:
        value = display_values[0]
        st.sidebar.number_input(
            f"{label} ({unit})" if unit else label,
            value=value,
            disabled=True,
        )
    else:
        value = st.sidebar.select_slider(
            f"{label} ({unit})" if unit else label,
            options=display_values,
            value=display_values[0],
            help="Filters saved, precomputed replay conditions; no retraining.",
        )
    numeric = pd.to_numeric(frame[column], errors="coerce")
    return frame[np.isclose(numeric, float(value))].copy(), value


def _display_line_style(algorithm: str) -> str:
    return {
        "-": "solid",
        "--": "dash",
        "-.": "dashdot",
        ":": "dot",
    }.get(ALGORITHM_LINESTYLES.get(algorithm, "-"), "solid")


def _map_figure(
    go: Any,
    replay: pd.DataFrame,
    current: pd.DataFrame,
    environment: dict[str, Any],
    telemetry: pd.DataFrame,
    timestep: float,
    *,
    show_signals: bool,
    show_heatmap: bool,
    show_history: bool,
) -> Any:
    figure = go.Figure()
    width = float(environment["width"])
    height = float(environment["height"])

    if show_heatmap:
        focus = current["algorithm"].iloc[-1]
        heat_rows = replay[replay["algorithm"] == focus]
        figure.add_trace(
            go.Scatter(
                x=heat_rows["true_x"],
                y=heat_rows["true_y"],
                mode="markers",
                marker={
                    "size": 18,
                    "color": heat_rows["error"],
                    "colorscale": "Viridis",
                    "opacity": 0.34,
                    "showscale": True,
                    "colorbar": {
                        "title": {"text": "Error<br>(m)"},
                        "thickness": 12,
                        "len": 0.42,
                    },
                },
                name=f"{focus} spatial error",
                hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<extra></extra>",
            )
        )

    for obstacle in environment.get("obstacles", []):
        if "vertices" in obstacle:
            x_values = [point[0] for point in obstacle["vertices"]]
            y_values = [point[1] for point in obstacle["vertices"]]
            x_values.append(x_values[0])
            y_values.append(y_values[0])
            figure.add_trace(
                go.Scatter(
                    x=x_values,
                    y=y_values,
                    fill="toself",
                    fillcolor="rgba(70, 96, 117, 0.55)",
                    line={"color": "#7892A5", "width": 1},
                    mode="lines",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
        else:
            x0 = float(obstacle["x"])
            y0 = float(obstacle["y"])
            x1 = x0 + float(obstacle["width"])
            y1 = y0 + float(obstacle["height"])
            figure.add_shape(
                type="rect",
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                fillcolor="rgba(70, 96, 117, 0.55)",
                line={"color": "#7892A5", "width": 1},
                layer="below",
            )
    figure.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=width,
        y1=height,
        line={"color": "#A9C4D8", "width": 3},
        layer="below",
    )
    for wall in environment.get("walls", []):
        figure.add_shape(
            type="line",
            x0=wall["start"][0],
            y0=wall["start"][1],
            x1=wall["end"][0],
            y1=wall["end"][1],
            line={"color": "#A9C4D8", "width": 4},
            layer="below",
        )

    truth = (
        replay.sort_values("timestep")
        .drop_duplicates("timestep")
        [["timestep", "true_x", "true_y"]]
    )
    if show_history:
        history = truth[truth["timestep"] <= timestep]
        figure.add_trace(
            go.Scatter(
                x=truth["true_x"],
                y=truth["true_y"],
                mode="lines",
                line={"color": DARK["muted"], "width": 1, "dash": "dot"},
                name="Full ground-truth path",
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=history["true_x"],
                y=history["true_y"],
                mode="lines",
                line={"color": DARK["text"], "width": 3},
                name="Ground truth",
                hovertemplate="x=%{x:.2f} m<br>y=%{y:.2f} m<extra></extra>",
            )
        )

    target = current.iloc[0]
    if show_signals:
        for _, row in telemetry.iterrows():
            if not _bool_value(row["available"], default=True):
                continue
            is_los = _bool_value(row["los"], default=True)
            figure.add_trace(
                go.Scatter(
                    x=[float(row["x"]), float(target["true_x"])],
                    y=[float(row["y"]), float(target["true_y"])],
                    mode="lines",
                    line={
                        "color": (
                            OKABE_ITO["sky_blue"]
                            if is_los
                            else OKABE_ITO["vermillion"]
                        ),
                        "width": 1.3,
                        "dash": "solid" if is_los else "dash",
                    },
                    opacity=0.58,
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

    anchor_x = []
    anchor_y = []
    anchor_text = []
    anchor_symbols = []
    anchor_colors = []
    telemetry_by_id = telemetry.set_index("anchor_id")
    for anchor in environment["anchors"]:
        anchor_id = str(anchor["anchor_id"])
        row = telemetry_by_id.loc[anchor_id]
        online = _bool_value(row["available"], default=True)
        anchor_x.append(float(anchor["x"]))
        anchor_y.append(float(anchor["y"]))
        anchor_symbols.append("triangle-up" if online else "x")
        anchor_colors.append(
            OKABE_ITO["sky_blue"] if online else "#718596"
        )
        rss = row["rss_dbm"]
        rss_text = f"{float(rss):.1f} dBm" if pd.notna(rss) else "RSS N/A"
        anchor_text.append(f"<b>{anchor_id}</b><br>{rss_text}")
    figure.add_trace(
        go.Scatter(
            x=anchor_x,
            y=anchor_y,
            mode="markers+text",
            text=[str(anchor["anchor_id"]) for anchor in environment["anchors"]],
            textposition="top right",
            marker={
                "size": 14,
                "symbol": anchor_symbols,
                "color": anchor_colors,
                "line": {"color": DARK["text"], "width": 1},
            },
            customdata=anchor_text,
            hovertemplate="%{customdata}<extra></extra>",
            name="Anchors",
        )
    )

    for algorithm in ordered_algorithms(replay["algorithm"]):
        rows = replay[
            (replay["algorithm"] == algorithm)
            & (replay["timestep"] <= timestep)
        ].sort_values("timestep")
        if rows.empty:
            continue
        figure.add_trace(
            go.Scatter(
                x=rows["pred_x"],
                y=rows["pred_y"],
                mode="lines",
                line={
                    "color": algorithm_color(algorithm),
                    "width": 2.2,
                    "dash": _display_line_style(algorithm),
                },
                name=algorithm,
                hovertemplate=(
                    f"{algorithm}<br>x=%{{x:.2f}} m<br>y=%{{y:.2f}} m"
                    "<extra></extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[rows["pred_x"].iloc[-1]],
                y=[rows["pred_y"].iloc[-1]],
                mode="markers",
                marker={
                    "size": 10,
                    "symbol": "diamond",
                    "color": algorithm_color(algorithm),
                    "line": {"color": DARK["background"], "width": 1},
                },
                name=f"{algorithm} current",
                showlegend=False,
                hovertemplate=(
                    f"{algorithm}<br>error={rows['error'].iloc[-1]:.2f} m"
                    "<extra></extra>"
                ),
            )
        )

    figure.add_trace(
        go.Scatter(
            x=[float(target["true_x"])],
            y=[float(target["true_y"])],
            mode="markers",
            marker={
                "size": 15,
                "color": OKABE_ITO["yellow"],
                "line": {"color": DARK["background"], "width": 2},
            },
            name="True position",
            hovertemplate="True position<br>x=%{x:.2f} m<br>y=%{y:.2f} m<extra></extra>",
        )
    )
    figure.update_layout(
        height=640,
        margin={"l": 20, "r": 20, "t": 35, "b": 20},
        paper_bgcolor=DARK["panel"],
        plot_bgcolor=DARK["panel"],
        font={"color": DARK["text"], "family": "Inter, Arial, sans-serif"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": -0.17,
            "xanchor": "left",
            "x": 0,
            "font": {"size": 11},
        },
        hovermode="closest",
        xaxis={
            "title": "x position (m)",
            "range": [-0.5, width + 0.5],
            "gridcolor": DARK["grid"],
            "zeroline": False,
            "scaleanchor": "y",
            "scaleratio": 1,
        },
        yaxis={
            "title": "y position (m)",
            "range": [-0.5, height + 0.5],
            "gridcolor": DARK["grid"],
            "zeroline": False,
        },
    )
    return figure


def _error_figure(
    go: Any, replay: pd.DataFrame, timestep: float
) -> Any:
    figure = go.Figure()
    for algorithm in ordered_algorithms(replay["algorithm"]):
        rows = replay[
            (replay["algorithm"] == algorithm)
            & (replay["timestep"] <= timestep)
        ].sort_values("timestep")
        figure.add_trace(
            go.Scatter(
                x=rows["timestep"],
                y=rows["error"],
                mode="lines",
                line={
                    "color": algorithm_color(algorithm),
                    "width": 2,
                    "dash": _display_line_style(algorithm),
                },
                name=algorithm,
                hovertemplate="t=%{x}<br>error=%{y:.2f} m<extra></extra>",
            )
        )
    figure.update_layout(
        height=315,
        margin={"l": 15, "r": 10, "t": 15, "b": 10},
        paper_bgcolor=DARK["panel"],
        plot_bgcolor=DARK["panel"],
        font={"color": DARK["text"]},
        xaxis={"title": "Timestep", "gridcolor": DARK["grid"]},
        yaxis={"title": "Error (m)", "gridcolor": DARK["grid"]},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0,
            "font": {"size": 10},
        },
    )
    return figure


def _comparison_figure(
    go: Any,
    metrics: pd.DataFrame | None,
    replay: pd.DataFrame,
    scenario: str,
) -> Any:
    comparison = pd.DataFrame()
    if metrics is not None:
        metrics_rows = select_scenario(metrics, scenario, require_rows=False)
        if not metrics_rows.empty:
            comparison = (
                metrics_rows.groupby("algorithm", as_index=False)[
                    ["mean_error", "p90_error"]
                ]
                .mean(numeric_only=True)
                .dropna(subset=["mean_error"])
            )
    if comparison.empty:
        comparison = (
            replay.groupby("algorithm", as_index=False)["error"]
            .agg(mean_error="mean", p90_error=lambda values: values.quantile(0.9))
        )
    algorithms = ordered_algorithms(comparison["algorithm"])
    comparison = (
        comparison.set_index("algorithm").reindex(algorithms).reset_index()
    )
    figure = go.Figure(
        data=[
            go.Bar(
                x=comparison["algorithm"],
                y=comparison["mean_error"],
                marker_color=[
                    algorithm_color(name) for name in comparison["algorithm"]
                ],
                text=[f"{value:.2f}" for value in comparison["mean_error"]],
                textposition="outside",
                customdata=comparison[["p90_error"]],
                hovertemplate=(
                    "%{x}<br>mean=%{y:.2f} m<br>P90=%{customdata[0]:.2f} m"
                    "<extra></extra>"
                ),
            )
        ]
    )
    figure.update_layout(
        height=315,
        margin={"l": 15, "r": 10, "t": 15, "b": 10},
        paper_bgcolor=DARK["panel"],
        plot_bgcolor=DARK["panel"],
        font={"color": DARK["text"]},
        xaxis={"title": None, "gridcolor": DARK["grid"]},
        yaxis={"title": "Mean error (m)", "gridcolor": DARK["grid"]},
        showlegend=False,
    )
    return figure


def _metric(
    metrics: pd.DataFrame | None,
    replay: pd.DataFrame,
    scenario: str,
    algorithm: str,
    column: str,
) -> float:
    if metrics is not None and column in metrics:
        rows = metrics[
            (metrics["scenario"] == scenario)
            & (metrics["algorithm"] == algorithm)
        ]
        values = pd.to_numeric(rows[column], errors="coerce").dropna()
        if not values.empty:
            return float(values.mean())
    errors = replay.loc[replay["algorithm"] == algorithm, "error"].dropna()
    if errors.empty:
        return float("nan")
    functions = {
        "mean_error": errors.mean,
        "median_error": errors.median,
        "p90_error": lambda: errors.quantile(0.9),
    }
    if column in functions:
        return float(functions[column]())
    if column == "nlos_mean_error" and "nlos_anchor_count" in replay:
        count = pd.to_numeric(replay["nlos_anchor_count"], errors="coerce")
        subset = replay.loc[
            (replay["algorithm"] == algorithm) & (count > 0), "error"
        ].dropna()
        return float(subset.mean()) if not subset.empty else float("nan")
    return float("nan")


def _format(value: float, unit: str, digits: int = 2) -> str:
    if not np.isfinite(value):
        return "N/A"
    return f"{value:.{digits}f} {unit}".strip()


def _runtime_inference(
    runtime: pd.DataFrame | None, algorithm: str, fallback: float
) -> float:
    """Use the dedicated runtime table when a filtered method lacks metrics."""

    if np.isfinite(fallback) or runtime is None:
        return fallback
    values = pd.to_numeric(
        runtime.loc[
            runtime["algorithm"] == algorithm, "inference_time_ms"
        ],
        errors="coerce",
    ).dropna()
    return float(values.mean()) if not values.empty else fallback


def _render_missing_state(st: Any, bundle: ResultBundle, results_dir: Path) -> None:
    st.markdown(
        """
        <div class="hero-card">
          <div class="eyebrow">ARTIFACT-ONLY REPLAY</div>
          <h1>AI-Assisted 6G Indoor Localization Digital Twin</h1>
          <p>The dashboard is ready, but no complete saved experiment is
          available. It will not invent measurements or retrain models.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.error("A completed experiment run is required.")
    st.code(
        "python scripts/run_pipeline.py --profile quick\n"
        "streamlit run app.py",
        language="powershell",
    )
    st.caption(f"Expected saved artifacts under: {results_dir}")
    with st.expander("Missing or invalid artifacts", expanded=True):
        for message in bundle.missing:
            st.write(f"• {message}")


def _css() -> str:
    return """
    <style>
      :root { --bg:#07111F; --panel:#0D1B2A; --line:#27445D;
              --text:#E8F1F8; --muted:#91A9BC; --cyan:#56B4E9; }
      .stApp { background:
        radial-gradient(circle at 82% 7%, rgba(86,180,233,.11), transparent 27%),
        linear-gradient(180deg, #07111F 0%, #091522 100%); }
      [data-testid="stSidebar"] {
        background: linear-gradient(180deg,#0B1928,#0D2031);
        border-right:1px solid var(--line); }
      [data-testid="stHeader"] { background:rgba(7,17,31,.85); }
      h1,h2,h3,p,label,span { font-family:Inter,Segoe UI,Arial,sans-serif; }
      .hero-card { padding:1.1rem 1.35rem; border:1px solid var(--line);
        border-radius:16px; background:linear-gradient(135deg,#0D1B2A,#102B42);
        box-shadow:0 14px 40px rgba(0,0,0,.18); margin-bottom:.8rem; }
      .hero-card h1 { color:var(--text); font-size:1.8rem; margin:.15rem 0 .35rem; }
      .hero-card p { color:var(--muted); margin:0; max-width:900px; }
      .eyebrow { color:var(--cyan); font-weight:700; letter-spacing:.16em;
        font-size:.72rem; }
      .status-pill { display:inline-block; padding:.28rem .62rem;
        border:1px solid #31506A; border-radius:999px; color:#BBD2E2;
        background:#10263B; font-size:.75rem; margin-right:.35rem; }
      [data-testid="stMetric"] { background:linear-gradient(145deg,#0D1B2A,#10263B);
        border:1px solid var(--line); padding:.75rem .8rem; border-radius:12px; }
      [data-testid="stMetricLabel"] { color:var(--muted); }
      [data-testid="stMetricValue"] { color:var(--text); font-size:1.34rem; }
      [data-testid="stVerticalBlockBorderWrapper"] {
        border-color:var(--line) !important; background:rgba(13,27,42,.72);
        border-radius:14px; }
      .simulation-note { color:#E69F00; font-size:.78rem; font-weight:700;
        letter-spacing:.04em; text-align:right; }
      .small-muted { color:var(--muted); font-size:.78rem; }
      div[data-testid="stDataFrame"] { border:1px solid var(--line);
        border-radius:10px; overflow:hidden; }
      .stButton button, .stDownloadButton button {
        border:1px solid #3A607D; border-radius:9px; background:#102B42;
        color:var(--text); }
      .stButton button:hover, .stDownloadButton button:hover {
        border-color:var(--cyan); color:white; }
    </style>
    """


def main() -> None:
    try:
        import streamlit as st
    except ImportError as exc:
        raise SystemExit(
            "Streamlit is not installed. Install project requirements, then run "
            "`streamlit run app.py`."
        ) from exc
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        st.error(
            "Plotly is required for the interactive dashboard. Install project "
            "requirements and restart Streamlit."
        )
        raise SystemExit(1) from exc

    st.set_page_config(
        page_title=APP_TITLE,
        page_icon="📡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_css(), unsafe_allow_html=True)

    results_dir = Path(
        os.environ.get("LOCALIZATION_TWIN_RESULTS", str(DEFAULT_RESULTS))
    ).expanduser()

    @st.cache_data(show_spinner=False)
    def load_saved_results(path: str) -> ResultBundle:
        return ResultBundle.load(path, strict=False)

    bundle = load_saved_results(str(results_dir))
    if bundle.predictions is None or bundle.environment is None:
        _render_missing_state(st, bundle, results_dir)
        return

    predictions = bundle.predictions
    available_scenarios = [
        scenario
        for scenario in SCENARIO_ORDER
        if scenario in set(predictions["scenario"])
    ]
    available_scenarios.extend(
        sorted(
            set(predictions["scenario"]) - set(available_scenarios)
        )
    )
    if not available_scenarios:
        _render_missing_state(st, bundle, results_dir)
        return

    query_scenario = _query_scenario(st)
    initial_scenario = (
        query_scenario
        if query_scenario in available_scenarios
        else available_scenarios[0]
    )
    st.sidebar.markdown("## Replay Control")
    st.sidebar.caption("Saved artifacts only · no training at app launch")
    scenario = st.sidebar.selectbox(
        "Scenario",
        available_scenarios,
        index=available_scenarios.index(initial_scenario),
        format_func=lambda value: SCENARIO_LABELS.get(
            value, value.replace("_", " ").title()
        ),
    )
    scenario_rows = select_scenario(predictions, scenario)

    seeds = (
        sorted(scenario_rows["seed"].dropna().unique().tolist())
        if "seed" in scenario_rows
        else []
    )
    seed = st.sidebar.selectbox(
        "Random seed",
        seeds or ["not recorded"],
        disabled=not seeds,
    )
    if seeds:
        scenario_rows = scenario_rows[scenario_rows["seed"] == seed]

    st.sidebar.markdown("#### Saved condition filters")
    scenario_rows, noise = _precomputed_numeric_control(
        st,
        scenario_rows,
        label="Noise strength",
        candidates=("noise_std_db", "noise_sigma_db"),
        unit="dB",
    )
    scenario_rows, wall_loss = _precomputed_numeric_control(
        st,
        scenario_rows,
        label="Wall attenuation",
        candidates=("wall_attenuation_db", "wall_loss_db"),
        unit="dB",
    )
    scenario_rows, nlos_bias = _precomputed_numeric_control(
        st,
        scenario_rows,
        label="NLoS bias",
        candidates=("nlos_bias_db",),
        unit="dB",
    )
    scenario_rows, anchors_dropped = _precomputed_numeric_control(
        st,
        scenario_rows,
        label="Anchor dropout",
        candidates=("dropped_anchor_count", "anchors_dropped"),
        unit="count",
        integer=True,
    )
    if scenario_rows.empty:
        st.error(
            "The selected combination has no saved replay rows. Choose another "
            "precomputed condition or rerun the pipeline."
        )
        return

    trajectories = (
        scenario_rows["trajectory_id"].dropna().unique().tolist()
        if "trajectory_id" in scenario_rows
        else []
    )
    trajectory = st.sidebar.selectbox(
        "Trajectory",
        trajectories or ["not recorded"],
        disabled=not trajectories,
    )
    if trajectories:
        scenario_rows = scenario_rows[
            scenario_rows["trajectory_id"] == trajectory
        ]

    algorithms = ordered_algorithms(scenario_rows["algorithm"])
    selected_algorithms = st.sidebar.multiselect(
        "Algorithms",
        algorithms,
        default=algorithms,
        help="All curves are loaded from saved per-sample predictions.",
    )
    if not selected_algorithms:
        st.warning("Select at least one saved algorithm.")
        return
    replay = scenario_rows[
        scenario_rows["algorithm"].isin(selected_algorithms)
    ].sort_values(["algorithm", "timestep"])

    playback_speed = st.sidebar.slider(
        "Playback speed (frames/s)", 0.5, 5.0, 1.5, 0.5
    )
    show_signal_lines = st.sidebar.checkbox("Show signal lines", value=True)
    show_heatmap = st.sidebar.checkbox("Show error heatmap", value=False)
    show_history = st.sidebar.checkbox("Show trajectory history", value=True)

    timesteps = np.sort(replay["timestep"].dropna().unique())
    state_key = (
        f"{scenario}|{seed}|{trajectory}|"
        + ",".join(selected_algorithms)
    )
    if st.session_state.get("_replay_key") != state_key:
        st.session_state["_replay_key"] = state_key
        st.session_state["_pending_frame"] = 0
        st.session_state["playing"] = False
    frame_max = max(0, len(timesteps) - 1)
    pending_frame = st.session_state.pop("_pending_frame", None)
    if pending_frame is not None:
        st.session_state["frame_index"] = min(
            max(0, int(pending_frame)), frame_max
        )
    st.session_state["frame_index"] = min(
        int(st.session_state.get("frame_index", 0)), frame_max
    )
    frame_index = st.sidebar.slider(
        "Trajectory frame",
        0,
        frame_max,
        key="frame_index",
        format="%d",
    )
    control_columns = st.sidebar.columns(3)
    if control_columns[0].button("◀", help="Previous frame"):
        st.session_state["_pending_frame"] = max(0, frame_index - 1)
        st.rerun()
    if control_columns[1].button(
        "▶" if not st.session_state.get("playing") else "Ⅱ",
        help="Play or pause saved frames",
    ):
        st.session_state["playing"] = not st.session_state.get("playing", False)
        st.rerun()
    if control_columns[2].button("↺", help="Reset replay"):
        st.session_state["_pending_frame"] = 0
        st.session_state["playing"] = False
        st.rerun()

    export_rows = replay.copy()
    st.sidebar.download_button(
        "⬇ Export Current Scenario",
        data=export_rows.to_csv(index=False).encode("utf-8"),
        file_name=f"{scenario}_replay.csv",
        mime="text/csv",
        use_container_width=True,
    )
    model_files = sorted((ROOT / "models").glob("*.joblib"))
    st.sidebar.caption(
        f"Artifact status: {len(replay):,} replay rows · "
        f"{len(model_files)} saved model file(s) · no retraining"
    )

    timestep = float(timesteps[frame_index])
    current = _current_rows(replay, timestep)
    preferred = next(
        (
            name
            for name in (
                "Residual AI + Kalman",
                "Residual AI",
                "Direct AI",
                "KNN",
                "Geometric LS",
            )
            if name in selected_algorithms
        ),
        selected_algorithms[-1],
    )
    preferred_row = current[current["algorithm"] == preferred].iloc[0]
    anchor_columns = infer_anchor_columns(
        replay, bundle.environment["anchors"]
    )
    telemetry = anchor_telemetry(
        preferred_row, bundle.environment, anchor_columns
    )
    available_count = int(
        sum(_bool_value(value, default=True) for value in telemetry["available"])
    )

    st.markdown(
        f"""
        <div class="hero-card">
          <div class="eyebrow">SAVED SIMULATION REPLAY</div>
          <h1>{APP_TITLE}</h1>
          <p>{SCENARIO_LABELS.get(scenario, scenario.title())} · frame
          {frame_index + 1}/{len(timesteps)} · focus method {preferred}</p>
        </div>
        <div class="simulation-note">SIMULATION — NOT A REAL 6G DEPLOYMENT</div>
        """,
        unsafe_allow_html=True,
    )
    card_columns = st.columns(7)
    current_error = float(preferred_row["error"])
    card_columns[0].metric("Current Error", _format(current_error, "m"))
    card_columns[1].metric(
        "Mean Error",
        _format(
            _metric(bundle.metrics, replay, scenario, preferred, "mean_error"),
            "m",
        ),
    )
    card_columns[2].metric(
        "Median Error",
        _format(
            _metric(
                bundle.metrics, replay, scenario, preferred, "median_error"
            ),
            "m",
        ),
    )
    card_columns[3].metric(
        "P90 Error",
        _format(
            _metric(bundle.metrics, replay, scenario, preferred, "p90_error"),
            "m",
        ),
    )
    card_columns[4].metric(
        "NLoS Error",
        _format(
            _metric(
                bundle.metrics, replay, scenario, preferred, "nlos_mean_error"
            ),
            "m",
        ),
    )
    card_columns[5].metric(
        "Available Anchors",
        f"{available_count}/{len(telemetry)}",
    )
    card_columns[6].metric(
        "Inference Time",
        _format(
            _runtime_inference(
                bundle.runtime,
                preferred,
                _metric(
                    bundle.metrics,
                    replay,
                    scenario,
                    preferred,
                    "inference_time_ms",
                ),
            ),
            "ms",
            digits=3,
        ),
    )

    map_column, side_column = st.columns([1.72, 1.0], gap="large")
    with map_column:
        with st.container(border=True):
            st.markdown("### Indoor Digital Twin Map")
            st.plotly_chart(
                _map_figure(
                    go,
                    replay,
                    current,
                    bundle.environment,
                    telemetry,
                    timestep,
                    show_signals=show_signal_lines,
                    show_heatmap=show_heatmap,
                    show_history=show_history,
                ),
                use_container_width=True,
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                },
            )
    with side_column:
        with st.container(border=True):
            st.markdown("### Live Error Curve")
            st.plotly_chart(
                _error_figure(go, replay, timestep),
                use_container_width=True,
                config={"displayModeBar": False},
            )
        with st.container(border=True):
            st.markdown("### Anchor Signal Panel")
            panel = telemetry.copy()
            panel["RSS (dBm)"] = panel["rss_dbm"].map(
                lambda value: f"{float(value):.1f}" if pd.notna(value) else "N/A"
            )
            panel["Distance (m)"] = panel["estimated_distance_m"].map(
                lambda value: f"{float(value):.2f}" if pd.notna(value) else "N/A"
            )
            panel["Path"] = panel["los"].map(
                lambda value: "LoS" if _bool_value(value, True) else "NLoS"
            )
            panel["Status"] = panel["available"].map(
                lambda value: (
                    "Online" if _bool_value(value, True) else "Offline"
                )
            )
            st.dataframe(
                panel[
                    ["anchor_id", "RSS (dBm)", "Distance (m)", "Path", "Status"]
                ].rename(columns={"anchor_id": "Anchor"}),
                hide_index=True,
                use_container_width=True,
            )
            if panel["rss_dbm"].isna().all():
                st.caption(
                    "Per-anchor telemetry was not saved. Add rss_A1, "
                    "est_distance_A1, los_A1, and available_A1 columns."
                )

    summary_column, comparison_column = st.columns([0.72, 1.28], gap="large")
    with summary_column:
        with st.container(border=True):
            st.markdown("### Scenario Summary")
            condition_rows = [
                ("Scenario", SCENARIO_LABELS.get(scenario, scenario.title())),
                ("Seed", seed),
                ("Trajectory", trajectory),
                ("Frame / timestep", f"{frame_index + 1} / {timestep:g}"),
                ("Focus method", preferred),
                ("Noise", f"{noise} dB" if noise is not None else "not recorded"),
                (
                    "Wall attenuation",
                    f"{wall_loss} dB"
                    if wall_loss is not None
                    else "not recorded",
                ),
                (
                    "NLoS bias",
                    f"{nlos_bias} dB"
                    if nlos_bias is not None
                    else "not recorded",
                ),
                (
                    "Dropped anchors",
                    anchors_dropped
                    if anchors_dropped is not None
                    else "not recorded",
                ),
            ]
            summary_frame = pd.DataFrame(
                condition_rows, columns=["Condition", "Value"]
            ).astype(str)
            st.dataframe(
                summary_frame,
                hide_index=True,
                use_container_width=True,
            )
    with comparison_column:
        with st.container(border=True):
            st.markdown("### Method Comparison")
            st.plotly_chart(
                _comparison_figure(go, bundle.metrics, replay, scenario),
                use_container_width=True,
                config={"displayModeBar": False},
            )
    st.caption(
        "Every value shown above comes from saved simulation artifacts. "
        "Changing a control filters a precomputed replay; the app never "
        "re-trains models or claims real-world 6G performance."
    )

    capture_mode = False
    try:
        capture_mode = str(st.query_params.get("capture", "0")) == "1"
    except AttributeError:
        capture_mode = (
            st.experimental_get_query_params().get("capture", ["0"])[0] == "1"
        )
    if (
        st.session_state.get("playing", False)
        and not capture_mode
        and frame_index < frame_max
    ):
        time.sleep(max(0.05, 1.0 / playback_speed))
        st.session_state["_pending_frame"] = frame_index + 1
        st.rerun()
    elif frame_index >= frame_max:
        st.session_state["playing"] = False


if __name__ == "__main__":
    main()
