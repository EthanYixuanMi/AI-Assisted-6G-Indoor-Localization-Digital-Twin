"""Validation and normalization for saved experiment visualization artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .style import canonical_algorithm, canonical_scenario


class VisualDataError(RuntimeError):
    """Raised when evidence required for a visualization is absent or invalid."""


RESULT_FILES = {
    "metrics": "metrics.csv",
    "predictions": "per_sample_predictions.csv",
    "robustness": "robustness_results.csv",
    "ablation": "ablation_results.csv",
    "runtime": "runtime_results.csv",
}

REQUIRED_COLUMNS = {
    "metrics": {
        "algorithm",
        "scenario",
        "mean_error",
        "rmse",
        "median_error",
        "p90_error",
    },
    "predictions": {
        "algorithm",
        "scenario",
        "timestep",
        "true_x",
        "true_y",
        "pred_x",
        "pred_y",
    },
    "robustness": {"experiment", "algorithm", "level", "mean_error"},
    "ablation": {"variant", "mean_error"},
    "runtime": {"algorithm", "inference_time_ms"},
}

COLUMN_ALIASES = {
    "method": "algorithm",
    "model": "algorithm",
    "estimator": "algorithm",
    "scenario_name": "scenario",
    "scenario_id": "scenario",
    "random_seed": "seed",
    "time_step": "timestep",
    "step": "timestep",
    "frame": "timestep",
    "target_x": "true_x",
    "target_y": "true_y",
    "ground_truth_x": "true_x",
    "ground_truth_y": "true_y",
    "gt_x": "true_x",
    "gt_y": "true_y",
    "estimated_x": "pred_x",
    "estimated_y": "pred_y",
    "estimate_x": "pred_x",
    "estimate_y": "pred_y",
    "prediction_x": "pred_x",
    "prediction_y": "pred_y",
    "error_m": "error",
    "euclidean_error": "error",
    "euclidean_error_m": "error",
    "mean_error_m": "mean_error",
    "rmse_m": "rmse",
    "median_error_m": "median_error",
    "p90_error_m": "p90_error",
    "max_error_m": "max_error",
    "los_mean_error_m": "los_mean_error",
    "nlos_mean_error_m": "nlos_mean_error",
    "inference_ms": "inference_time_ms",
    "latency_ms": "inference_time_ms",
    "training_seconds": "training_time_s",
    "model_size": "model_size_mb",
    "ablation": "variant",
    "ablation_name": "variant",
    "condition": "variant",
    "experiment_type": "experiment",
    "perturbation": "experiment",
    "noise_std_db": "level",
    "noise_sigma_db": "level",
    "noise_level": "level",
    "anchors_dropped": "level",
    "dropped_anchor_count": "level",
    "anchor_dropout_count": "level",
}


def _normalized_column(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def _read_nonempty_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise VisualDataError(
            f"Missing required result file: {path}. Run the experiment pipeline "
            "before exporting visual assets."
        )
    if path.stat().st_size == 0:
        raise VisualDataError(f"Required result file is empty: {path}")
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        raise VisualDataError(f"Could not read {path}: {exc}") from exc
    if frame.empty:
        raise VisualDataError(f"Required result table has no rows: {path}")
    return frame


def _normalize_frame(frame: pd.DataFrame, kind: str) -> pd.DataFrame:
    normalized = frame.copy()
    renames: dict[object, str] = {}
    seen: set[str] = set()
    for column in normalized.columns:
        key = _normalized_column(column)
        target = COLUMN_ALIASES.get(key, key)
        if target in seen:
            continue
        renames[column] = target
        seen.add(target)
    normalized = normalized.rename(columns=renames)

    # Robustness files often encode the perturbation in a scenario-like column.
    if kind == "robustness":
        if "experiment" not in normalized:
            if any(
                name in frame.columns
                for name in (
                    "noise_std_db",
                    "noise_sigma_db",
                    "noise_level",
                )
            ):
                normalized["experiment"] = "noise"
            elif any(
                name in frame.columns
                for name in (
                    "anchors_dropped",
                    "dropped_anchor_count",
                    "anchor_dropout_count",
                )
            ):
                normalized["experiment"] = "anchor_failure"
        if "level" not in normalized:
            for candidate in (
                "noise_std_db",
                "noise_sigma_db",
                "noise_level",
                "anchors_dropped",
                "dropped_anchor_count",
                "anchor_dropout_count",
            ):
                candidate_key = _normalized_column(candidate)
                if candidate_key in normalized:
                    normalized["level"] = normalized[candidate_key]
                    break

    missing = sorted(REQUIRED_COLUMNS[kind] - set(normalized.columns))
    if missing:
        raise VisualDataError(
            f"{RESULT_FILES[kind]} is missing required column(s): "
            f"{', '.join(missing)}"
        )

    if "algorithm" in normalized:
        normalized["algorithm"] = normalized["algorithm"].map(
            canonical_algorithm
        )
    if "scenario" in normalized:
        normalized["scenario"] = normalized["scenario"].map(canonical_scenario)
    if "experiment" in normalized:
        normalized["experiment"] = normalized["experiment"].map(canonical_scenario)
        normalized["experiment"] = normalized["experiment"].replace(
            {
                "high_noise": "noise",
                "noise_robustness": "noise",
                "anchor_dropout": "anchor_failure",
            }
        )

    if kind == "predictions":
        for column in ("true_x", "true_y", "pred_x", "pred_y", "timestep"):
            normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
        if "error" not in normalized:
            normalized["error"] = np.hypot(
                normalized["pred_x"] - normalized["true_x"],
                normalized["pred_y"] - normalized["true_y"],
            )
        else:
            normalized["error"] = pd.to_numeric(
                normalized["error"], errors="coerce"
            )
        invalid = normalized[
            ["true_x", "true_y", "pred_x", "pred_y", "error"]
        ].isna()
        if bool(invalid.any(axis=None)):
            count = int(invalid.any(axis=1).sum())
            raise VisualDataError(
                f"{RESULT_FILES[kind]} contains {count} row(s) with invalid "
                "coordinates or localization error."
            )

    numeric_candidates = {
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
        "batch_inference_time_ms",
        "dashboard_update_ms",
        "level",
    }
    for column in numeric_candidates & set(normalized.columns):
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")

    required_numeric = (
        REQUIRED_COLUMNS[kind]
        - {"algorithm", "scenario", "experiment", "variant"}
    )
    unusable = [
        column
        for column in sorted(required_numeric)
        if normalized[column].isna().all()
    ]
    if unusable:
        raise VisualDataError(
            f"{RESULT_FILES[kind]} has no numeric values for: "
            f"{', '.join(unusable)}"
        )
    return normalized


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VisualDataError(
            f"Missing required resolved configuration: {path}. "
            "Run the experiment pipeline first."
        )
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisualDataError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VisualDataError(f"Resolved configuration is not a mapping: {path}")
    return value


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise VisualDataError(
            f"Missing required environment description: {path}. "
            "Run the experiment pipeline first."
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise VisualDataError(f"Could not parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise VisualDataError(f"Environment description is not a mapping: {path}")
    return value


def _environment_from_config(config: dict[str, Any]) -> dict[str, Any] | None:
    candidate = config.get("environment")
    return candidate if isinstance(candidate, dict) else None


def normalize_environment(
    environment: dict[str, Any],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize common environment JSON variants without creating geometry."""

    source = dict(environment)
    if "environment" in source and isinstance(source["environment"], dict):
        nested = dict(source["environment"])
        nested.update({key: value for key, value in source.items() if key != "environment"})
        source = nested

    dimensions = source.get("dimensions", {})
    if not isinstance(dimensions, dict):
        dimensions = {}
    width = source.get(
        "width",
        source.get("width_m", dimensions.get("width", dimensions.get("width_m"))),
    )
    height = source.get(
        "height",
        source.get("height_m", dimensions.get("height", dimensions.get("height_m"))),
    )
    if width is None or height is None:
        config_environment = _environment_from_config(config or {}) or {}
        width = width or config_environment.get("width") or config_environment.get("width_m")
        height = (
            height
            or config_environment.get("height")
            or config_environment.get("height_m")
        )
    try:
        width_value = float(width)
        height_value = float(height)
    except (TypeError, ValueError) as exc:
        raise VisualDataError(
            "environment.json must provide numeric width/height (metres)."
        ) from exc
    if width_value <= 0 or height_value <= 0:
        raise VisualDataError("Environment width and height must be positive.")

    anchors_raw = source.get("anchors", [])
    if isinstance(anchors_raw, dict):
        anchors_raw = [
            {"anchor_id": anchor_id, **(value if isinstance(value, dict) else {})}
            for anchor_id, value in anchors_raw.items()
        ]
    anchors: list[dict[str, Any]] = []
    for index, anchor in enumerate(anchors_raw):
        if not isinstance(anchor, dict):
            continue
        position = anchor.get("position", [])
        x = anchor.get("x", position[0] if len(position) >= 1 else None)
        y = anchor.get("y", position[1] if len(position) >= 2 else None)
        try:
            anchors.append(
                {
                    **anchor,
                    "anchor_id": str(
                        anchor.get("anchor_id", anchor.get("id", f"A{index + 1}"))
                    ),
                    "x": float(x),
                    "y": float(y),
                    "online": bool(anchor.get("online", True)),
                }
            )
        except (TypeError, ValueError):
            continue
    if not anchors:
        raise VisualDataError("environment.json contains no usable anchors.")

    walls_raw = source.get("walls", [])
    walls: list[dict[str, Any]] = []
    for wall in walls_raw:
        if not isinstance(wall, dict):
            continue
        start = wall.get("start", wall.get("p1"))
        end = wall.get("end", wall.get("p2"))
        if start is None and {"x1", "y1", "x2", "y2"} <= set(wall):
            start = [wall["x1"], wall["y1"]]
            end = [wall["x2"], wall["y2"]]
        try:
            walls.append(
                {
                    **wall,
                    "start": [float(start[0]), float(start[1])],
                    "end": [float(end[0]), float(end[1])],
                }
            )
        except (TypeError, ValueError, IndexError):
            continue

    obstacles_raw = source.get("obstacles", [])
    obstacles: list[dict[str, Any]] = []
    for obstacle in obstacles_raw:
        if not isinstance(obstacle, dict):
            continue
        if "vertices" in obstacle:
            try:
                vertices = [
                    [float(point[0]), float(point[1])]
                    for point in obstacle["vertices"]
                ]
                obstacles.append({**obstacle, "vertices": vertices})
            except (TypeError, ValueError, IndexError):
                continue
            continue
        x = obstacle.get("x", obstacle.get("x_min"))
        y = obstacle.get("y", obstacle.get("y_min"))
        width_obstacle = obstacle.get(
            "width",
            (
                float(obstacle["x_max"]) - float(x)
                if x is not None and "x_max" in obstacle
                else None
            ),
        )
        height_obstacle = obstacle.get(
            "height",
            (
                float(obstacle["y_max"]) - float(y)
                if y is not None and "y_max" in obstacle
                else None
            ),
        )
        try:
            obstacles.append(
                {
                    **obstacle,
                    "x": float(x),
                    "y": float(y),
                    "width": float(width_obstacle),
                    "height": float(height_obstacle),
                }
            )
        except (TypeError, ValueError):
            continue

    return {
        **source,
        "width": width_value,
        "height": height_value,
        "anchors": anchors,
        "walls": walls,
        "obstacles": obstacles,
    }


@dataclass(slots=True)
class ResultBundle:
    """Validated experiment tables plus the exact resolved environment."""

    results_dir: Path
    metrics: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None
    robustness: pd.DataFrame | None = None
    ablation: pd.DataFrame | None = None
    runtime: pd.DataFrame | None = None
    config: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    missing: list[str] = field(default_factory=list)

    @classmethod
    def load(cls, results_dir: str | Path, strict: bool = True) -> "ResultBundle":
        """Load saved outputs.

        In strict mode every contracted artifact must exist and validate.  The
        dashboard uses non-strict mode so it can display actionable guidance
        rather than crashing when a pipeline has not been run.
        """

        root = Path(results_dir).expanduser().resolve()
        bundle = cls(results_dir=root)
        for attribute, filename in RESULT_FILES.items():
            path = root / filename
            try:
                raw = _read_nonempty_csv(path)
                setattr(bundle, attribute, _normalize_frame(raw, attribute))
            except VisualDataError as exc:
                if strict:
                    raise
                bundle.missing.append(str(exc))

        config_path = root / "config_resolved.yaml"
        try:
            bundle.config = _load_yaml(config_path)
        except VisualDataError as exc:
            if strict:
                raise
            bundle.missing.append(str(exc))

        environment_path = root / "environment.json"
        try:
            raw_environment = _load_json(environment_path)
            bundle.environment = normalize_environment(
                raw_environment, config=bundle.config
            )
        except VisualDataError as exc:
            if strict:
                raise
            bundle.missing.append(str(exc))
            if bundle.config:
                config_environment = _environment_from_config(bundle.config)
                if config_environment:
                    try:
                        bundle.environment = normalize_environment(
                            config_environment, config=bundle.config
                        )
                    except VisualDataError as config_exc:
                        bundle.missing.append(
                            "Resolved-config environment is also unusable: "
                            f"{config_exc}"
                        )
        return bundle

    def require(self, *attributes: str) -> None:
        """Raise a concise error when a requested table is unavailable."""

        missing = [
            RESULT_FILES.get(attribute, attribute)
            for attribute in attributes
            if getattr(self, attribute, None) is None
        ]
        if missing:
            raise VisualDataError(
                "Missing visualization source artifact(s): "
                f"{', '.join(missing)}. Run `python scripts/run_pipeline.py "
                "--profile quick` and retry."
            )


def select_scenario(
    frame: pd.DataFrame,
    scenario: str,
    *,
    require_rows: bool = True,
) -> pd.DataFrame:
    """Select a canonical scenario with an explicit empty-result error."""

    wanted = canonical_scenario(scenario)
    result = frame[frame["scenario"].map(canonical_scenario) == wanted].copy()
    if require_rows and result.empty:
        available = ", ".join(sorted(frame["scenario"].astype(str).unique()))
        raise VisualDataError(
            f"No saved rows for scenario '{wanted}'. Available scenarios: "
            f"{available or '(none)'}."
        )
    return result


def select_replay(
    predictions: pd.DataFrame,
    scenario: str,
    seed: int | float | None = None,
    trajectory_id: str | int | None = None,
) -> pd.DataFrame:
    """Select one reproducible seed/trajectory replay shared by all methods."""

    selected = select_scenario(predictions, scenario)
    if "seed" in selected:
        available_seeds = selected["seed"].dropna().sort_values().unique()
        if len(available_seeds):
            chosen_seed = seed if seed in available_seeds else available_seeds[0]
            selected = selected[selected["seed"] == chosen_seed]
    # Runner outputs contain both spatial test points and continuous replay
    # trajectories.  Prefer the latter for maps instead of accidentally
    # connecting a static holdout set in CSV row order.
    if "split" in selected:
        split_keys = selected["split"].astype(str).map(_normalized_column)
        trajectory_rows = selected[
            split_keys.isin({"trajectory", "trajectories", "replay"})
        ]
        if not trajectory_rows.empty:
            selected = trajectory_rows
    if "trajectory_id" in selected:
        available_trajectories = selected["trajectory_id"].dropna().unique()
        if len(available_trajectories):
            dynamic = [
                value
                for value in available_trajectories
                if _normalized_column(value)
                not in {
                    "static",
                    "test",
                    "spatial_holdout",
                    "in_domain",
                    "domain_shift",
                    "anchor_failure",
                }
            ]
            if dynamic:
                available_trajectories = np.asarray(dynamic, dtype=object)
            chosen_trajectory = (
                trajectory_id
                if trajectory_id in available_trajectories
                else available_trajectories[0]
            )
            selected = selected[
                selected["trajectory_id"] == chosen_trajectory
            ]
    if selected.empty:
        raise VisualDataError(
            f"No replay rows remain for scenario '{canonical_scenario(scenario)}'."
        )
    return selected.sort_values(["algorithm", "timestep"]).reset_index(drop=True)


def infer_anchor_columns(
    frame: pd.DataFrame, anchors: list[dict[str, Any]]
) -> dict[str, dict[str, str]]:
    """Map environment anchor IDs to optional wide telemetry columns."""

    normalized_to_original = {
        _normalized_column(column): str(column) for column in frame.columns
    }
    result: dict[str, dict[str, str]] = {}
    patterns = {
        "rss": ("rss_{id}", "{id}_rss"),
        "estimated_distance": (
            "est_distance_{id}",
            "estimated_distance_{id}",
            "distance_est_{id}",
            "{id}_est_distance",
        ),
        "los": ("los_{id}", "is_los_{id}", "{id}_los"),
        "available": (
            "available_{id}",
            "availability_{id}",
            "mask_{id}",
            "{id}_available",
        ),
    }
    for anchor in anchors:
        anchor_id = str(anchor["anchor_id"])
        anchor_key = _normalized_column(anchor_id)
        fields: dict[str, str] = {}
        for field_name, templates in patterns.items():
            for template in templates:
                candidate = _normalized_column(template.format(id=anchor_key))
                if candidate in normalized_to_original:
                    fields[field_name] = normalized_to_original[candidate]
                    break
        result[anchor_id] = fields
    return result


def anchor_telemetry(
    row: pd.Series,
    environment: dict[str, Any],
    columns: dict[str, dict[str, str]],
) -> pd.DataFrame:
    """Extract one frame of per-anchor telemetry without inventing values."""

    records: list[dict[str, Any]] = []
    for anchor in environment["anchors"]:
        anchor_id = str(anchor["anchor_id"])
        mapping = columns.get(anchor_id, {})
        available_value = (
            row.get(mapping["available"])
            if "available" in mapping
            else anchor.get("online")
        )
        los_value = row.get(mapping["los"]) if "los" in mapping else np.nan
        records.append(
            {
                "anchor_id": anchor_id,
                "x": anchor["x"],
                "y": anchor["y"],
                "rss_dbm": (
                    pd.to_numeric(row.get(mapping["rss"]), errors="coerce")
                    if "rss" in mapping
                    else np.nan
                ),
                "estimated_distance_m": (
                    pd.to_numeric(
                        row.get(mapping["estimated_distance"]), errors="coerce"
                    )
                    if "estimated_distance" in mapping
                    else np.nan
                ),
                "los": los_value,
                "available": available_value,
            }
        )
    return pd.DataFrame.from_records(records)
