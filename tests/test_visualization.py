"""Visualization contract tests use explicit synthetic fixtures only.

The fixture data are confined to pytest's temporary directory.  Production
exporters still reject missing or empty experiment evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
import yaml

from localization_twin.visualization.report_assets import (
    VisualAssetError,
    export_all_visual_assets,
    export_report_figures,
)


ALGORITHMS = ("Geometric LS", "KNN", "Direct AI", "Residual AI")
SCENARIOS = (
    "normal",
    "high_noise",
    "strong_blockage",
    "anchor_failure",
    "domain_shift",
)
ANCHORS = (
    ("A1", 1.0, 1.0),
    ("A2", 19.0, 1.0),
    ("A3", 19.0, 11.0),
    ("A4", 1.0, 11.0),
)


def _write_visual_fixture(root: Path) -> Path:
    results = root / "results"
    results.mkdir()
    config = {
        "profile": "test",
        "seed": 17,
        "environment": {"width": 20.0, "height": 12.0},
    }
    (results / "config_resolved.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    environment = {
        "width": 20.0,
        "height": 12.0,
        "anchors": [
            {"anchor_id": anchor_id, "x": x, "y": y, "online": True}
            for anchor_id, x, y in ANCHORS
        ],
        "walls": [
            {"start": [7.0, 0.0], "end": [7.0, 7.5], "attenuation_db": 5.0},
            {"start": [13.0, 4.5], "end": [13.0, 12.0], "attenuation_db": 5.0},
        ],
        "obstacles": [
            {"x": 9.0, "y": 4.5, "width": 2.0, "height": 2.5}
        ],
    }
    (results / "environment.json").write_text(
        json.dumps(environment, indent=2), encoding="utf-8"
    )

    prediction_records = []
    scales = {
        "Geometric LS": 0.85,
        "KNN": 0.62,
        "Direct AI": 0.52,
        "Residual AI": 0.38,
    }
    scenario_multiplier = {
        "normal": 1.0,
        "high_noise": 1.45,
        "strong_blockage": 1.8,
        "anchor_failure": 1.65,
        "domain_shift": 1.5,
    }
    for scenario_index, scenario in enumerate(SCENARIOS):
        for algorithm_index, algorithm in enumerate(ALGORITHMS):
            for timestep in range(24):
                fraction = timestep / 23
                true_x = 2.0 + 16.0 * fraction
                true_y = 2.0 + 7.5 * fraction + 1.1 * np.sin(fraction * 2 * np.pi)
                amplitude = scales[algorithm] * scenario_multiplier[scenario]
                dx = amplitude * np.sin(0.45 * timestep + algorithm_index * 0.3)
                dy = amplitude * np.cos(0.37 * timestep + scenario_index * 0.2)
                record = {
                    "algorithm": algorithm,
                    "scenario": scenario,
                    "seed": 17,
                    "trajectory_id": "static",
                    "timestep": timestep,
                    "split": "spatial_holdout",
                    "true_x": true_x,
                    "true_y": true_y,
                    "pred_x": true_x + dx,
                    "pred_y": true_y + dy,
                    "error": float(np.hypot(dx, dy)),
                    "nlos_anchor_count": 0 if timestep < 8 else 2,
                    "noise_std_db": 2.0 + scenario_index * 0.5,
                    "wall_attenuation_db": 5.0,
                    "nlos_bias_db": 3.0,
                    "dropped_anchor_count": 1 if scenario == "anchor_failure" else 0,
                }
                for anchor_index, (anchor_id, anchor_x, anchor_y) in enumerate(ANCHORS):
                    distance = float(np.hypot(true_x - anchor_x, true_y - anchor_y))
                    available = not (
                        scenario == "anchor_failure" and anchor_id == "A2"
                    )
                    los = not (timestep >= 8 and anchor_index % 2 == 0)
                    record[f"rss_{anchor_id}"] = (
                        -38.0 - 20.0 * np.log10(max(distance, 1.0))
                        - (6.0 if not los else 0.0)
                        if available
                        else np.nan
                    )
                    record[f"est_distance_{anchor_id}"] = (
                        distance * (1.15 if not los else 1.0)
                        if available
                        else np.nan
                    )
                    record[f"los_{anchor_id}"] = los
                    record[f"available_{anchor_id}"] = available
                prediction_records.append(record)
    predictions = pd.DataFrame(prediction_records)
    trajectory_rows = predictions.copy()
    trajectory_rows["split"] = "trajectories"
    trajectory_rows["trajectory_id"] = "turning_path"
    filtered = trajectory_rows[
        trajectory_rows["algorithm"] == "Residual AI"
    ].copy()
    filtered["algorithm"] = "Residual AI + Kalman"
    filtered["pred_x"] = (
        0.75 * filtered["pred_x"] + 0.25 * filtered["true_x"]
    )
    filtered["pred_y"] = (
        0.75 * filtered["pred_y"] + 0.25 * filtered["true_y"]
    )
    filtered["error"] = np.hypot(
        filtered["pred_x"] - filtered["true_x"],
        filtered["pred_y"] - filtered["true_y"],
    )
    predictions = pd.concat(
        [predictions, trajectory_rows, filtered],
        ignore_index=True,
    )
    predictions.to_csv(results / "per_sample_predictions.csv", index=False)

    metric_records = []
    for scenario in SCENARIOS:
        for algorithm in ALGORITHMS:
            values = predictions.loc[
                (predictions["scenario"] == scenario)
                & (predictions["algorithm"] == algorithm),
                "error",
            ]
            los_values = values.iloc[:8]
            nlos_values = values.iloc[8:]
            metric_records.append(
                {
                    "algorithm": algorithm,
                    "scenario": scenario,
                    "seed": 17,
                    "sample_count": len(values),
                    "mean_error": values.mean(),
                    "rmse": np.sqrt(np.mean(values**2)),
                    "median_error": values.median(),
                    "p90_error": values.quantile(0.9),
                    "max_error": values.max(),
                    "los_mean_error": los_values.mean(),
                    "nlos_mean_error": nlos_values.mean(),
                    "inference_time_ms": 0.25 + ALGORITHMS.index(algorithm) * 0.2,
                    "training_time_s": 0.0
                    if algorithm == "Geometric LS"
                    else 1.2 + ALGORITHMS.index(algorithm),
                    "model_size_mb": 0.0
                    if algorithm == "Geometric LS"
                    else 0.4 + ALGORITHMS.index(algorithm) * 0.2,
                }
            )
    pd.DataFrame(metric_records).to_csv(results / "metrics.csv", index=False)

    robustness_records = []
    for seed in (17, 23):
        for algorithm in ALGORITHMS:
            base = scales[algorithm]
            for level in (1.0, 3.0, 5.0):
                robustness_records.append(
                    {
                        "experiment": "noise",
                        "algorithm": algorithm,
                        "scenario": "high_noise",
                        "seed": seed,
                        "level": level,
                        "mean_error": base * (1.0 + 0.18 * level) + seed * 0.0005,
                        "p90_error": base * (1.3 + 0.22 * level),
                    }
                )
            for level in (0, 1, 2):
                robustness_records.append(
                    {
                        "experiment": "anchor_failure",
                        "algorithm": algorithm,
                        "scenario": "anchor_failure",
                        "seed": seed,
                        "level": level,
                        "mean_error": base * (1.0 + 0.4 * level) + seed * 0.0005,
                        "p90_error": base * (1.35 + 0.5 * level),
                    }
                )
    pd.DataFrame(robustness_records).to_csv(
        results / "robustness_results.csv", index=False
    )

    ablation_records = []
    variants = (
        "Full Residual AI",
        "Without LoS/NLoS features",
        "Without geometric residual",
        "Without anchor mask",
        "Without spatial bias training",
    )
    for seed in (17, 23):
        for index, variant in enumerate(variants):
            ablation_records.append(
                {
                    "variant": variant,
                    "seed": seed,
                    "sample_count": 80,
                    "mean_error": 0.42 + 0.11 * index + seed * 0.0003,
                    "rmse": 0.5 + 0.13 * index,
                    "p90_error": 0.7 + 0.17 * index,
                }
            )
    pd.DataFrame(ablation_records).to_csv(
        results / "ablation_results.csv", index=False
    )

    runtime_records = []
    for index, algorithm in enumerate(ALGORITHMS):
        runtime_records.append(
            {
                "algorithm": algorithm,
                "training_time_s": 0.0 if index == 0 else 1.0 + index,
                "inference_time_ms": 0.2 + 0.17 * index,
                "batch_inference_time_ms": 3.0 + index,
                "model_size_mb": 0.0 if index == 0 else 0.3 + index * 0.2,
                "dashboard_update_ms": 25.0 + index,
            }
        )
    runtime_records.append(
        {
            "algorithm": "Residual AI + Kalman",
            "training_time_s": 4.0,
            "inference_time_ms": 0.95,
            "batch_inference_time_ms": 7.5,
            "model_size_mb": 1.0,
            "dashboard_update_ms": 29.0,
        }
    )
    pd.DataFrame(runtime_records).to_csv(
        results / "runtime_results.csv", index=False
    )
    return results


def test_complete_visual_asset_export(tmp_path: Path) -> None:
    results = _write_visual_fixture(tmp_path)
    output = tmp_path / "report_assets"
    paths = export_all_visual_assets(results, output)

    required_pairs = {
        "trajectory_comparison",
        "error_cdf",
        "spatial_error_heatmaps",
        "robustness_noise",
        "robustness_anchor_failure",
        "robustness_results",
        "los_nlos_comparison",
        "ablation_results",
        "runtime_comparison",
    }
    for stem in required_pairs:
        for extension in ("pdf", "png"):
            path = output / "figures" / f"{stem}.{extension}"
            assert path in paths
            assert path.stat().st_size > 500
    overview = output / "figures" / "dashboard_overview.png"
    image = mpimg.imread(overview)
    assert image.shape[1] >= 3000
    assert image.shape[0] >= 1700
    for scenario in SCENARIOS:
        screenshot = output / "screenshots" / f"dashboard_{scenario}.png"
        assert screenshot.stat().st_size > 10_000
    svg = output / "figures" / "system_architecture.svg"
    assert "<svg" in svg.read_text(encoding="utf-8")
    manifest = json.loads(
        (output / "data" / "visual_asset_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["no_fabricated_fallback_data"] is True
    assert "metrics.csv" in manifest["source_files"]
    assert manifest["figure_protocols"]["error_cdf"]["filters"] == {
        "scenario": "normal",
        "split": "spatial_holdout",
        "algorithms": list(ALGORITHMS),
    }
    assert (
        manifest["figure_protocols"]["ablation_results"]["filters"][
            "exclude_variant_substring_case_insensitive"
        ]
        == "kalman"
    )

    cdf_source = pd.read_csv(output / "data" / "error_cdf_source.csv")
    assert set(cdf_source["algorithm"]) == set(ALGORITHMS)
    assert set(cdf_source["split"]) == {"spatial_holdout"}
    assert cdf_source.groupby("algorithm").size().nunique() == 1

    ablation_source = pd.read_csv(
        output / "data" / "ablation_results_source.csv"
    )
    assert not ablation_source["variant"].str.contains(
        "kalman", case=False
    ).any()
    assert set(ablation_source["sample_count"]) == {80}


def test_missing_results_are_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(VisualAssetError, match="Missing required result file"):
        export_report_figures(empty, tmp_path / "assets")
