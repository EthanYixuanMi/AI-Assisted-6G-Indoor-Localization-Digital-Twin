from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from localization_twin.evaluation import (
    add_baseline_improvements,
    add_robustness_degradation,
    baseline_improvement,
    euclidean_errors,
    evaluate_predictions,
    localization_metrics,
    los_nlos_metrics,
    measure_inference_time,
)


def test_euclidean_metrics_use_distance_not_coordinate_rmse() -> None:
    truth = np.asarray([[0.0, 0.0], [0.0, 0.0]])
    prediction = np.asarray([[0.0, 0.0], [3.0, 4.0]])
    np.testing.assert_allclose(euclidean_errors(truth, prediction), [0.0, 5.0])
    metrics = localization_metrics(truth, prediction)
    assert metrics["sample_count"] == 2
    assert metrics["mean_error"] == pytest.approx(2.5)
    assert metrics["rmse"] == pytest.approx(math.sqrt(12.5))
    assert metrics["median_error"] == pytest.approx(2.5)
    assert metrics["p90_error"] == pytest.approx(4.5)
    assert metrics["max_error"] == pytest.approx(5.0)


def test_los_nlos_metrics_are_stratified_from_nlos_count() -> None:
    truth = np.zeros((4, 2))
    prediction = np.asarray([[1.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 4.0]])
    frame = pd.DataFrame({"nlos_anchor_count": [0, 1, 0, 2]})
    result = los_nlos_metrics(truth, prediction, frame)
    assert result["los_sample_count"] == 2
    assert result["nlos_sample_count"] == 2
    assert result["los_mean_error"] == pytest.approx(2.0)
    assert result["los_median_error"] == pytest.approx(2.0)
    assert result["nlos_mean_error"] == pytest.approx(3.0)
    assert result["nlos_median_error"] == pytest.approx(3.0)
    assert result["nlos_p90_error"] == pytest.approx(3.8)
    assert result["nlos_los_mean_degradation_percent"] == pytest.approx(50.0)
    combined = evaluate_predictions(truth, prediction, nlos=frame)
    assert combined["mean_error"] == pytest.approx(2.5)
    assert combined["nlos_mean_error"] == pytest.approx(3.0)


def test_empty_los_stratum_is_explicit_nan() -> None:
    truth = np.zeros((2, 2))
    prediction = np.ones((2, 2))
    result = los_nlos_metrics(truth, prediction, [1, 2])
    assert result["los_sample_count"] == 0
    assert math.isnan(result["los_mean_error"])
    assert result["nlos_sample_count"] == 2


def test_baseline_improvement_signs_and_table_grouping() -> None:
    improvement = baseline_improvement(method_error=1.5, baseline_error=2.0)
    assert improvement["error_change"] == pytest.approx(-0.5)
    assert improvement["absolute_improvement"] == pytest.approx(0.5)
    assert improvement["percent_improvement"] == pytest.approx(25.0)
    worse = baseline_improvement(method_error=2.5, baseline_error=2.0)
    assert worse["percent_improvement"] == pytest.approx(-25.0)

    results = pd.DataFrame(
        {
            "scenario": ["normal", "normal", "shift", "shift"],
            "seed": [1, 1, 1, 1],
            "algorithm": ["Geometric LS", "Residual AI"] * 2,
            "mean_error": [2.0, 1.5, 3.0, 3.6],
        }
    )
    augmented = add_baseline_improvements(results)
    residual = augmented[augmented["algorithm"] == "Residual AI"].reset_index(drop=True)
    assert residual.loc[0, "percent_improvement"] == pytest.approx(25.0)
    assert residual.loc[1, "percent_improvement"] == pytest.approx(-20.0)
    assert residual.loc[0, "absolute_error_change"] == pytest.approx(-0.5)


def test_inference_timing_performs_warmup() -> None:
    calls = {"count": 0}

    def predictor(values: np.ndarray) -> np.ndarray:
        calls["count"] += 1
        return values * 2.0

    values = np.ones((10, 2))
    timing = measure_inference_time(
        predictor,
        values,
        warmup_runs=2,
        repeats=3,
    )
    assert calls["count"] == 5
    assert timing["sample_count"] == 10
    assert timing["batch_inference_time_ms"] >= 0.0
    assert timing["inference_time_ms"] == pytest.approx(
        timing["batch_inference_time_ms"] / 10
    )


def test_robustness_degradation_uses_cleanest_level() -> None:
    rows = pd.DataFrame(
        {
            "algorithm": ["A", "A", "B", "B"],
            "robustness_dimension": ["noise"] * 4,
            "seed": [1] * 4,
            "robustness_level": [1.0, 5.0, 1.0, 5.0],
            "mean_error": [1.0, 2.0, 2.0, 3.0],
        }
    )
    result = add_robustness_degradation(rows)
    high_a = result[(result["algorithm"] == "A") & (result["robustness_level"] == 5.0)]
    assert high_a.iloc[0]["mean_error_degradation"] == pytest.approx(1.0)
    assert high_a.iloc[0]["mean_error_degradation_percent"] == pytest.approx(100.0)


def test_metrics_reject_mismatched_shapes_and_nonfinite_values() -> None:
    with pytest.raises(ValueError):
        euclidean_errors(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        localization_metrics(np.zeros((1, 2)), np.asarray([[np.nan, 0.0]]))
