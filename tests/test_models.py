from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from localization_twin.models import (
    DirectAILocator,
    GeometricLocator,
    KNNFingerprintLocator,
    KalmanFilter2D,
    ResidualAILocator,
    extract_residual_features,
)


ANCHORS = {
    "a0": (0.0, 0.0),
    "a1": (10.0, 0.0),
    "a2": (0.0, 10.0),
    "a3": (10.0, 10.0),
}
ANCHOR_IDS = list(ANCHORS)


def artifact_path(name: str) -> Path:
    directory = Path("models")
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"_pytest_{name}"


def synthetic_frame(
    count: int = 100,
    *,
    seed: int = 7,
    distance_bias: float = 0.0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.5, 9.5, size=(count, 2))
    data: dict[str, np.ndarray] = {
        "true_x": xy[:, 0],
        "true_y": xy[:, 1],
    }
    nlos_columns: list[np.ndarray] = []
    for index, (anchor_id, position) in enumerate(ANCHORS.items()):
        distance = np.linalg.norm(xy - np.asarray(position), axis=1)
        nlos = (xy[:, 0] > 5.0) & (index % 2 == 0)
        los = ~nlos
        biased_distance = distance + distance_bias * nlos.astype(float)
        rss = -40.0 - 20.0 * np.log10(np.maximum(distance, 0.1))
        rss -= 3.0 * nlos
        data[f"rss_{anchor_id}"] = rss
        data[f"estimated_distance_{anchor_id}"] = biased_distance
        data[f"los_{anchor_id}"] = los.astype(int)
        data[f"available_{anchor_id}"] = np.ones(count, dtype=int)
        nlos_columns.append(nlos.astype(int))
    data["nlos_anchor_count"] = np.column_stack(nlos_columns).sum(axis=1)
    return pd.DataFrame(data)


def test_geometric_locator_is_accurate_with_ideal_distances() -> None:
    frame = synthetic_frame(30)
    locator = GeometricLocator(ANCHORS, bounds=((0.0, 0.0), (10.0, 10.0)))
    details = locator.predict(frame, return_details=True)
    prediction = details[["geometric_x", "geometric_y"]].to_numpy()
    truth = frame[["true_x", "true_y"]].to_numpy()
    assert np.max(np.linalg.norm(prediction - truth, axis=1)) < 1e-5
    assert (details["available_count"] == 4).all()
    assert (details["residual_cost"] < 1e-9).all()


def test_geometric_locator_falls_back_when_anchors_are_insufficient() -> None:
    frame = synthetic_frame(2)
    for anchor_id in ("a2", "a3"):
        frame.loc[:, f"available_{anchor_id}"] = 0
        frame.loc[:, f"estimated_distance_{anchor_id}"] = np.nan
    locator = GeometricLocator(ANCHORS, bounds=((0.0, 0.0), (10.0, 10.0)))
    details = locator.predict_details(frame)
    assert np.isfinite(details[["geometric_x", "geometric_y", "residual_cost"]]).all().all()
    assert (details["available_count"] == 2).all()
    assert details["used_fallback"].all()
    assert np.all((details[["geometric_x", "geometric_y"]] >= 0.0).to_numpy())
    assert np.all((details[["geometric_x", "geometric_y"]] <= 10.0).to_numpy())


def test_locator_constructed_with_two_collinear_anchors_is_bounded() -> None:
    frame = synthetic_frame(3)
    locator = GeometricLocator({"a0": ANCHORS["a0"], "a1": ANCHORS["a1"]})
    prediction = locator.predict(frame)
    assert prediction.shape == (3, 2)
    assert np.isfinite(prediction).all()


def test_feature_extraction_and_residual_ablation_groups() -> None:
    frame = synthetic_frame(4)
    geometric = GeometricLocator(ANCHORS).predict_details(frame)
    full, names = extract_residual_features(
        frame, ANCHOR_IDS, geometric=geometric, return_names=True
    )
    without_mask, names_without_mask = extract_residual_features(
        frame,
        ANCHOR_IDS,
        geometric=geometric,
        feature_groups=(
            "rss",
            "distance",
            "geometric",
            "residual",
            "available_count",
            "nlos",
            "los",
        ),
        return_names=True,
    )
    assert full.shape[0] == len(frame)
    assert "residual_cost" in names
    anchor_masks = {f"available_{anchor_id}" for anchor_id in ANCHOR_IDS}
    assert anchor_masks.issubset(names)
    assert anchor_masks.isdisjoint(names_without_mask)
    assert without_mask.shape[1] == full.shape[1] - len(ANCHOR_IDS)


def test_knn_selects_validation_k_and_round_trips() -> None:
    train = synthetic_frame(120, seed=1)
    validation = synthetic_frame(35, seed=2)
    test = synthetic_frame(20, seed=3)
    # Exercise dropout/imputation.
    test.loc[test.index[:3], ["rss_a0", "estimated_distance_a0"]] = np.nan
    test.loc[test.index[:3], "available_a0"] = 0
    model = KNNFingerprintLocator(ANCHOR_IDS, k_candidates=(1, 3, 5))
    model.fit(train, validation_frame=validation)
    prediction = model.predict(test)
    assert prediction.shape == (len(test), 2)
    assert model.best_k in {1, 3, 5}
    path = model.save(artifact_path("knn.joblib"))
    try:
        loaded = KNNFingerprintLocator.load(path)
        np.testing.assert_allclose(
            loaded.predict(test), prediction, rtol=0.0, atol=0.0
        )
    finally:
        path.unlink(missing_ok=True)


def test_direct_ai_shape_and_round_trip() -> None:
    train = synthetic_frame(100, seed=10)
    validation = synthetic_frame(25, seed=11)
    test = synthetic_frame(12, seed=12)
    model = DirectAILocator(
        ANCHOR_IDS,
        configs=(
            {
                "hidden_layer_sizes": (16,),
                "alpha": 1e-3,
                "learning_rate_init": 2e-3,
            },
        ),
        max_iter=60,
        random_state=9,
        n_iter_no_change=8,
    )
    model.fit(train, validation_frame=validation)
    prediction = model.predict(test)
    assert prediction.shape == (len(test), 2)
    assert model.best_config_["hidden_layer_sizes"] == (16,)
    path = model.save(artifact_path("direct.joblib"))
    try:
        loaded = DirectAILocator.load(path)
        np.testing.assert_allclose(
            loaded.predict(test), prediction, rtol=0.0, atol=0.0
        )
    finally:
        path.unlink(missing_ok=True)


@pytest.mark.parametrize("estimator", ["extra_trees", "random_forest"])
def test_residual_ai_corrects_and_round_trips(estimator: str) -> None:
    train = synthetic_frame(140, seed=20, distance_bias=2.0)
    validation = synthetic_frame(35, seed=21, distance_bias=2.0)
    test = synthetic_frame(25, seed=22, distance_bias=2.0)
    geometric_locator = GeometricLocator(ANCHORS)
    geometric_train = geometric_locator.predict_details(train)
    geometric_validation = geometric_locator.predict_details(validation)
    geometric_test = geometric_locator.predict_details(test)
    model = ResidualAILocator(
        ANCHOR_IDS,
        estimator=estimator,
        n_estimators=24,
        max_depth=8,
        correction_scale=0.5,
        correction_cap_quantile=0.9,
        random_state=4,
    )
    model.fit(
        train,
        geometric_train,
        validation_frame=validation,
        validation_geometric=geometric_validation,
    )
    prediction = model.predict(test, geometric_test)
    assert prediction.shape == (len(test), 2)
    assert np.isfinite(prediction).all()
    path = model.save(artifact_path(f"{estimator}.joblib"))
    try:
        loaded = ResidualAILocator.load(path)
        assert loaded.correction_scale == 0.5
        assert loaded.correction_cap_quantile == 0.9
        assert loaded.correction_cap_ == model.correction_cap_
        np.testing.assert_allclose(
            loaded.predict(test, geometric_test), prediction, rtol=0.0, atol=0.0
        )
    finally:
        path.unlink(missing_ok=True)


def test_residual_ai_zero_scale_exactly_falls_back_to_geometric() -> None:
    train = synthetic_frame(80, seed=30, distance_bias=2.0)
    test = synthetic_frame(20, seed=31, distance_bias=2.0)
    geometric_locator = GeometricLocator(ANCHORS)
    geometric_train = geometric_locator.predict_details(train)
    geometric_test = geometric_locator.predict_details(test)
    model = ResidualAILocator(
        ANCHOR_IDS,
        n_estimators=16,
        max_depth=6,
        correction_scale=0.0,
        correction_cap_quantile=0.9,
        random_state=5,
    )
    model.fit(train, geometric_train)

    expected = geometric_test[["geometric_x", "geometric_y"]].to_numpy()
    np.testing.assert_array_equal(model.predict(test, geometric_test), expected)


def test_residual_ai_cap_uses_training_targets_only() -> None:
    train = synthetic_frame(4, seed=32)
    validation = synthetic_frame(2, seed=33)
    geometric = pd.DataFrame(
        {
            "geometric_x": np.zeros(4),
            "geometric_y": np.zeros(4),
        }
    )
    geometric_validation = pd.DataFrame(
        {
            "geometric_x": np.zeros(2),
            "geometric_y": np.zeros(2),
        }
    )
    residual_targets = np.asarray(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]
    )
    model = ResidualAILocator(
        ANCHOR_IDS,
        feature_groups=("geometric",),
        n_estimators=8,
        max_depth=4,
        correction_scale=0.4,
        correction_cap_quantile=0.5,
        random_state=6,
    )
    model.fit(
        train,
        geometric,
        y=residual_targets,
        validation_frame=validation,
        validation_geometric=geometric_validation,
        y_validation=np.asarray([[100.0, 0.0], [200.0, 0.0]]),
    )

    assert model.correction_cap_ == pytest.approx(2.5)
    bounded = model._apply_correction_policy([[100.0, 0.0]])
    assert np.linalg.norm(bounded[0]) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"correction_scale": -0.01},
        {"correction_scale": 1.01},
        {"correction_cap_quantile": 0.0},
        {"correction_cap_quantile": 1.01},
    ),
)
def test_residual_ai_rejects_invalid_correction_policy(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        ResidualAILocator(ANCHOR_IDS, **kwargs)


def test_geometric_save_load_is_identical() -> None:
    frame = synthetic_frame(8)
    model = GeometricLocator(ANCHORS)
    expected = model.predict(frame)
    path = model.save(artifact_path("geometric.joblib"))
    try:
        loaded = GeometricLocator.load(path)
        np.testing.assert_allclose(
            loaded.predict(frame), expected, rtol=0.0, atol=0.0
        )
    finally:
        path.unlink(missing_ok=True)


def test_kalman_filter_shape_smoothing_and_round_trip() -> None:
    rng = np.random.default_rng(123)
    truth = np.column_stack((np.linspace(0.0, 20.0, 80), np.linspace(2.0, 10.0, 80)))
    noisy = truth + rng.normal(0.0, 0.8, size=truth.shape)
    model = KalmanFilter2D(process_noise=0.02, measurement_noise=0.64)
    filtered = model.filter(noisy)
    assert filtered.shape == noisy.shape
    assert np.mean(np.linalg.norm(filtered - truth, axis=1)) < np.mean(
        np.linalg.norm(noisy - truth, axis=1)
    )
    path = model.save(artifact_path("kalman.joblib"))
    try:
        loaded = KalmanFilter2D.load(path)
        np.testing.assert_allclose(
            loaded.filter(noisy, reset=True),
            model.filter(noisy, reset=True),
            rtol=0.0,
            atol=0.0,
        )
    finally:
        path.unlink(missing_ok=True)
