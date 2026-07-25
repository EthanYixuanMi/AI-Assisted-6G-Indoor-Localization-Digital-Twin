"""Dataset generation, schema, and leakage tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from localization_twin.config import load_config
from localization_twin.dataset import generate_dataset
from localization_twin.environment import Environment
from localization_twin.features import build_feature_matrix, target_matrix


def _small_config() -> dict[str, object]:
    return load_config(
        "quick",
        "normal",
        overrides={
            "sampling": {
                "train_count": 80,
                "validation_count": 24,
                "test_count": 30,
                "spatial_holdout_count": 30,
                "domain_shift_count": 24,
                "anchor_failure_count": 24,
                "grid_fraction": 0.25,
                "trajectories": {"points_per_trajectory": 16},
            }
        },
    )


def test_dataset_has_required_splits_schema_and_scenarios() -> None:
    config = _small_config()
    splits, metadata = generate_dataset(config, seed=19)
    assert set(splits) == {
        "train",
        "validation",
        "test",
        "spatial_holdout",
        "domain_shift",
        "anchor_failure",
        "trajectories",
    }
    assert metadata["split_sizes"] == {
        "train": 80,
        "validation": 24,
        "test": 30,
        "spatial_holdout": 30,
        "domain_shift": 24,
        "anchor_failure": 24,
        "trajectories": 48,
    }
    anchor_ids = metadata["anchor_ids"]
    required = {
        "true_x",
        "true_y",
        "nlos_anchor_count",
        "anchor_availability_mask",
        "sample_class",
        "scenario_name",
        "trajectory_id",
        "timestep",
        "random_seed",
        "split",
    }
    for anchor_id in anchor_ids:
        required.update(
            {
                f"rss_{anchor_id}",
                f"true_distance_{anchor_id}",
                f"estimated_distance_{anchor_id}",
                f"los_{anchor_id}",
                f"available_{anchor_id}",
            }
        )
    assert required.issubset(splits["train"].columns)
    assert set(splits["domain_shift"]["scenario_name"]) == {"domain_shift"}
    assert set(splits["anchor_failure"]["scenario_name"]) == {"anchor_failure"}
    assert splits["anchor_failure"]["available_A2"].sum() == 0
    assert splits["anchor_failure"]["available_A5"].sum() == 0


def test_dataset_sampling_avoids_obstacles_and_spatial_leakage() -> None:
    config = _small_config()
    splits, metadata = generate_dataset(config, seed=23)
    environment = Environment.from_config(config)
    for frame in splits.values():
        assert all(
            environment.obstacle_at(point) is None
            for point in frame[["true_x", "true_y"]].to_numpy()
        )

    development = pd.concat(
        [splits["train"], splits["validation"], splits["test"]],
        ignore_index=True,
    )
    holdout = splits["spatial_holdout"]
    threshold = metadata["spatial_holdout"]["threshold"]
    buffer = metadata["spatial_holdout"]["buffer"]
    assert development["true_x"].max() <= threshold - buffer + 1e-12
    assert holdout["true_x"].min() >= threshold + buffer - 1e-12
    assert holdout["true_x"].min() - development["true_x"].max() >= 2.0 * buffer


def test_dataset_reproducibility_and_different_seed() -> None:
    config = _small_config()
    first, first_metadata = generate_dataset(config, seed=31)
    second, second_metadata = generate_dataset(config, seed=31)
    changed, _ = generate_dataset(config, seed=32)
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    assert first_metadata["config_sha256"] == second_metadata["config_sha256"]
    assert not np.allclose(
        first["train"][["true_x", "true_y"]].to_numpy(),
        changed["train"][["true_x", "true_y"]].to_numpy(),
    )


def test_feature_matrices_have_finite_values_and_expected_shape() -> None:
    splits, metadata = generate_dataset(_small_config(), seed=41)
    train = splits["train"]
    direct, direct_names = build_feature_matrix(train, "direct")
    knn, knn_names = build_feature_matrix(train, "knn")
    residual, residual_names = build_feature_matrix(
        train,
        "residual",
        geometric=np.zeros((len(train), 2)),
    )
    target = target_matrix(train)
    anchor_count = len(metadata["anchor_ids"])
    assert direct.shape == (len(train), anchor_count * 3 + 1)
    assert knn.shape == (len(train), anchor_count * 2)
    assert residual.shape[0] == len(train)
    assert residual.shape[1] == anchor_count * 4 + 3
    assert target.shape == (len(train), 2)
    assert len(direct_names) == direct.shape[1]
    assert len(knn_names) == knn.shape[1]
    assert len(residual_names) == residual.shape[1]
    assert np.isfinite(direct).all()
    assert np.isfinite(knn).all()
    assert np.isfinite(residual).all()

