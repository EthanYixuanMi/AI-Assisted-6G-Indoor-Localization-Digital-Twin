"""RSS propagation model tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from localization_twin.environment import Anchor, Environment, Wall
from localization_twin.propagation import PropagationModel


def _propagation_config(**overrides: object) -> dict[str, object]:
    propagation: dict[str, object] = {
        "reference_distance": 1.0,
        "minimum_distance": 0.1,
        "noise_std": 0.0,
        "nlos_bias_mean": 3.0,
        "nlos_bias_std": 0.0,
        "wall_loss_multiplier": 1.0,
        "obstacle_loss_multiplier": 1.0,
        "dropout_probability": 0.0,
        "fixed_offline_anchors": [],
        "spatial_bias": {
            "enabled": False,
            "amplitude": 0.0,
            "grid_shape": [5, 5],
            "correlation_sigma": 1.0,
        },
    }
    propagation.update(overrides)
    return {"propagation": propagation}


def test_rss_decreases_with_distance_in_ideal_los() -> None:
    environment = Environment(
        20.0,
        10.0,
        anchors=(Anchor("A1", 0.0, 0.0, -40.0, 2.0),),
    )
    model = PropagationModel(environment, _propagation_config(), seed=1)
    near = model.measure((1.0, 0.0))
    far = model.measure((10.0, 0.0))
    assert near["rss_A1"] == -40.0
    assert far["rss_A1"] == -60.0
    assert near["rss_A1"] > far["rss_A1"]
    assert np.isclose(far["estimated_distance_A1"], 10.0)


def test_wall_adds_attenuation_and_positive_nlos_bias() -> None:
    anchor = Anchor("A1", 1.0, 5.0, -40.0, 2.0)
    clear_environment = Environment(10.0, 10.0, anchors=(anchor,))
    blocked_environment = Environment(
        10.0,
        10.0,
        anchors=(anchor,),
        walls=(Wall("W1", (5.0, 0.0), (5.0, 10.0), 6.0),),
    )
    clear = PropagationModel(
        clear_environment, _propagation_config(), seed=2
    ).measure((9.0, 5.0))
    blocked = PropagationModel(
        blocked_environment, _propagation_config(), seed=2
    ).measure((9.0, 5.0))
    assert clear["los_A1"] is True
    assert blocked["los_A1"] is False
    assert blocked["sample_class"] == "NLoS"
    assert np.isclose(clear["rss_A1"] - blocked["rss_A1"], 9.0)
    assert blocked["estimated_distance_A1"] > clear["estimated_distance_A1"]


def test_seed_reproducibility_and_seed_variation() -> None:
    environment = Environment(
        10.0,
        10.0,
        anchors=(
            Anchor("A1", 1.0, 1.0, -40.0, 2.0),
            Anchor("A2", 9.0, 9.0, -40.0, 2.0),
        ),
    )
    config = _propagation_config(
        noise_std=2.0,
        dropout_probability=0.2,
        spatial_bias={
            "enabled": True,
            "amplitude": 2.0,
            "grid_shape": [7, 7],
            "correlation_sigma": 1.0,
            "seed_offset": 33,
        },
    )
    positions = np.asarray([[2.0, 2.0], [5.0, 5.0], [8.0, 3.0]])
    first = PropagationModel(environment, config, seed=77).simulate_positions(
        positions, split="test", random_seed=77
    )
    second = PropagationModel(environment, config, seed=77).simulate_positions(
        positions, split="test", random_seed=77
    )
    changed = PropagationModel(environment, config, seed=78).simulate_positions(
        positions, split="test", random_seed=78
    )
    pd.testing.assert_frame_equal(first, second)
    assert not np.allclose(
        first[["rss_A1", "rss_A2"]].fillna(-200.0),
        changed[["rss_A1", "rss_A2"]].fillna(-200.0),
    )


def test_fixed_anchor_failure_is_encoded_in_mask() -> None:
    environment = Environment(
        10.0,
        10.0,
        anchors=(
            Anchor("A1", 1.0, 1.0),
            Anchor("A2", 9.0, 1.0),
            Anchor("A3", 5.0, 9.0),
        ),
    )
    config = _propagation_config(fixed_offline_anchors=["A2"])
    measurement = PropagationModel(environment, config, seed=3).measure((5.0, 5.0))
    assert measurement["available_A1"] is True
    assert measurement["available_A2"] is False
    assert np.isnan(measurement["rss_A2"])
    assert np.isnan(measurement["estimated_distance_A2"])
    assert measurement["anchor_availability_mask"] == "[1,0,1]"
