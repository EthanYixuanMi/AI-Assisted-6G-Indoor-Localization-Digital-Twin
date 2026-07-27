"""Small end-to-end check for the public pipeline command."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from localization_twin.models import ResidualAILocator


ROOT = Path(__file__).resolve().parents[1]


def test_quick_pipeline_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "pipeline-output"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--profile",
        "quick",
        "--config",
        str(ROOT / "tests" / "fixtures" / "smoke.yaml"),
        "--output-dir",
        str(output_dir),
        "--skip-robustness",
        "--force",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "figures" / "error_cdf.pdf").stat().st_size > 0
    assert (output_dir / "screenshots" / "dashboard_normal.png").stat().st_size > 0
    assert (output_dir / "latex" / "table_main_results.tex").stat().st_size > 0
    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert set(("Geometric LS", "KNN", "Direct AI", "Residual AI")).issubset(
        set(metrics["algorithm"])
    )

    residual_model = ResidualAILocator.load(
        output_dir / "models" / "residual_ai.joblib"
    )
    expected_feature_groups = (
        "rss",
        "distance",
        "residual",
        "available_count",
        "mask",
        "nlos",
        "los",
    )
    assert residual_model.estimator == "random_forest"
    assert residual_model.selected_estimator_ == "random_forest"
    assert residual_model.feature_groups == expected_feature_groups
    assert residual_model.n_estimators == 7
    assert residual_model.max_depth == 6
    assert residual_model.min_samples_leaf == 3
    assert residual_model.max_features == pytest.approx(0.5)
    assert residual_model.correction_scale == 0.0
    assert residual_model.correction_cap_quantile == pytest.approx(0.5)
    assert residual_model.correction_cap_ is not None
    assert np.isfinite(residual_model.correction_cap_)
    assert residual_model.correction_cap_ > 0.0
    assert "geometric_x" not in residual_model.feature_names_
    assert "geometric_y" not in residual_model.feature_names_

    training_log = json.loads(
        (output_dir / "training_log.json").read_text(encoding="utf-8")
    )
    recorded = training_log["residual_ai_config"]
    assert recorded["configured_estimator"] == "random_forest"
    assert recorded["n_estimators"] == 7
    assert recorded["max_depth"] == 6
    assert recorded["min_samples_leaf"] == 3
    assert recorded["max_features"] == pytest.approx(0.5)
    assert tuple(recorded["feature_groups"]) == expected_feature_groups
    assert recorded["correction_scale"] == 0.0
    assert recorded["correction_cap_quantile"] == pytest.approx(0.5)
    assert recorded["learned_correction_cap"] == pytest.approx(
        residual_model.correction_cap_
    )
    assert recorded["feature_names"] == residual_model.feature_names_

    predictions = pd.read_csv(output_dir / "per_sample_predictions.csv")
    selected = predictions[
        predictions["algorithm"].isin(["Geometric LS", "Residual AI"])
    ].copy()
    selected["sample_index"] = selected.groupby(
        ["algorithm", "scenario", "seed"],
        sort=False,
    ).cumcount()
    keys = ["scenario", "seed", "sample_index"]
    geometric = selected[selected["algorithm"] == "Geometric LS"][
        keys + ["pred_x", "pred_y"]
    ]
    residual = selected[selected["algorithm"] == "Residual AI"][
        keys + ["pred_x", "pred_y"]
    ]
    paired = residual.merge(
        geometric,
        on=keys,
        suffixes=("_residual", "_geometric"),
        validate="one_to_one",
    )
    assert len(paired) == len(residual)
    np.testing.assert_array_equal(
        paired[["pred_x_residual", "pred_y_residual"]].to_numpy(),
        paired[["pred_x_geometric", "pred_y_geometric"]].to_numpy(),
    )

    ablation = pd.read_csv(output_dir / "ablation_results.csv")
    spatial_residual = ablation[
        (ablation["evaluation_split"] == "spatial_holdout")
        & (ablation["algorithm"] == "Residual AI")
    ]
    assert set(spatial_residual["variant"]) == {
        "full",
        "without_los_nlos",
        "without_geometric_residual",
        "without_anchor_mask",
        "without_spatial_bias_training",
    }
    np.testing.assert_allclose(
        spatial_residual["mean_error_change_vs_geometric"].to_numpy(dtype=float),
        np.zeros(len(spatial_residual), dtype=float),
        rtol=0.0,
        atol=1e-12,
    )
