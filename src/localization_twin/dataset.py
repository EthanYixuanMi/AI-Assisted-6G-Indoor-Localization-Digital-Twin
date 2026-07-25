"""End-to-end generation of reproducible localization datasets."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .config import CONFIG_DIR, deep_merge, load_config, load_yaml, validate_config
from .environment import Environment
from .propagation import PropagationModel
from .splits import SpatialHoldoutSpec, make_spatial_predicates
from .trajectory import generate_trajectories, sample_static_positions

LOGGER = logging.getLogger(__name__)

SPLIT_SEED_OFFSETS = {
    "train": 101,
    "validation": 211,
    "test": 307,
    "spatial_holdout": 401,
    "domain_shift": 503,
    "anchor_failure": 601,
    "trajectories": 701,
}


def _scenario_overlay(config: Mapping[str, Any], scenario_name: str) -> dict[str, Any]:
    path = CONFIG_DIR / "scenarios" / f"{scenario_name}.yaml"
    merged = deep_merge(config, load_yaml(path))
    validate_config(merged)
    return merged


def _simulate_static_split(
    config: Mapping[str, Any],
    environment: Environment,
    *,
    split: str,
    count: int,
    seed: int,
    root_seed: int,
    scenario_name: str,
    predicate: Any = None,
) -> pd.DataFrame:
    sampling = config["sampling"]
    positions = sample_static_positions(
        environment,
        count,
        seed=seed,
        grid_fraction=float(sampling.get("grid_fraction", 0.35)),
        clearance=float(sampling.get("obstacle_clearance", 0.0)),
        predicate=predicate,
    )
    simulator = PropagationModel(environment, config, seed=seed + 10_000)
    return simulator.simulate_positions(
        positions,
        split=split,
        scenario_name=scenario_name,
        random_seed=root_seed,
    )


def generate_dataset(
    config: Mapping[str, Any] | None = None,
    profile: str = "quick",
    seed: int = 42,
    scenario: str = "normal",
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    """Generate development, robustness, and trajectory datasets.

    The spatial holdout occupies a disjoint map region with a configured guard
    band, preventing nearby fingerprints from leaking into training. Domain
    shift and anchor failure are simulated with their dedicated YAML overlays.

    Returns:
        A ``(splits, metadata)`` tuple. Split keys are ``train``,
        ``validation``, ``test``, ``spatial_holdout``, ``domain_shift``,
        ``anchor_failure``, and ``trajectories``.
    """

    root_seed = int(seed)
    resolved = (
        load_config(profile=profile, scenario=scenario)
        if config is None
        else deepcopy(dict(config))
    )
    validate_config(resolved)
    scenario_name = str(resolved.get("scenario", {}).get("name", scenario))
    sampling = resolved["sampling"]
    environment = Environment.from_config(resolved)
    holdout_spec = SpatialHoldoutSpec.from_config(resolved)
    development_predicate, holdout_predicate = make_spatial_predicates(holdout_spec)

    splits: dict[str, pd.DataFrame] = {}
    for split, count_key in (
        ("train", "train_count"),
        ("validation", "validation_count"),
        ("test", "test_count"),
    ):
        split_seed = root_seed + SPLIT_SEED_OFFSETS[split]
        splits[split] = _simulate_static_split(
            resolved,
            environment,
            split=split,
            count=int(sampling[count_key]),
            seed=split_seed,
            root_seed=root_seed,
            scenario_name=scenario_name,
            predicate=development_predicate,
        )

    spatial_seed = root_seed + SPLIT_SEED_OFFSETS["spatial_holdout"]
    splits["spatial_holdout"] = _simulate_static_split(
        resolved,
        environment,
        split="spatial_holdout",
        count=int(sampling["spatial_holdout_count"]),
        seed=spatial_seed,
        root_seed=root_seed,
        scenario_name=scenario_name,
        predicate=holdout_predicate,
    )

    domain_config = _scenario_overlay(resolved, "domain_shift")
    domain_environment = Environment.from_config(domain_config)
    domain_seed = root_seed + SPLIT_SEED_OFFSETS["domain_shift"]
    splits["domain_shift"] = _simulate_static_split(
        domain_config,
        domain_environment,
        split="domain_shift",
        count=int(sampling["domain_shift_count"]),
        seed=domain_seed,
        root_seed=root_seed,
        scenario_name="domain_shift",
    )

    failure_config = _scenario_overlay(resolved, "anchor_failure")
    failure_environment = Environment.from_config(failure_config)
    failure_seed = root_seed + SPLIT_SEED_OFFSETS["anchor_failure"]
    splits["anchor_failure"] = _simulate_static_split(
        failure_config,
        failure_environment,
        split="anchor_failure",
        count=int(sampling["anchor_failure_count"]),
        seed=failure_seed,
        root_seed=root_seed,
        scenario_name="anchor_failure",
    )

    trajectory_seed = root_seed + SPLIT_SEED_OFFSETS["trajectories"]
    trajectory_points = generate_trajectories(
        environment, resolved, seed=trajectory_seed
    )
    trajectory_simulator = PropagationModel(
        environment, resolved, seed=trajectory_seed + 10_000
    )
    positions = trajectory_points[["true_x", "true_y"]].to_numpy(dtype=float)
    splits["trajectories"] = trajectory_simulator.simulate_positions(
        positions,
        split="trajectories",
        scenario_name=scenario_name,
        trajectory_ids=trajectory_points["trajectory_id"].astype(str).tolist(),
        timesteps=trajectory_points["timestep"].tolist(),
        random_seed=root_seed,
    )

    portable_config = {
        key: value for key, value in resolved.items() if key != "_meta"
    }
    config_json = json.dumps(
        portable_config, sort_keys=True, default=str
    ).encode("utf-8")
    metadata: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile": str(resolved.get("profile", {}).get("name", profile)),
        "base_scenario": scenario_name,
        "random_seed": root_seed,
        "split_seeds": {
            name: root_seed + offset for name, offset in SPLIT_SEED_OFFSETS.items()
        },
        "split_sizes": {name: int(len(frame)) for name, frame in splits.items()},
        "anchor_ids": list(environment.anchor_ids),
        "environment": environment.to_dict(),
        "spatial_holdout": {
            "axis": holdout_spec.axis,
            "threshold": holdout_spec.threshold,
            "buffer": holdout_spec.buffer,
            "side": holdout_spec.side,
            "minimum_axis_gap": 2.0 * holdout_spec.buffer,
        },
        "config_sha256": hashlib.sha256(config_json).hexdigest(),
        "simulated_data": True,
        "split_descriptions": {
            "train": "Nominal development region; model fitting only.",
            "validation": "Nominal development region; model selection only.",
            "test": "Nominal in-domain test positions.",
            "spatial_holdout": "Disjoint map region beyond a guard band.",
            "domain_shift": "Changed path loss, wall loss, spatial field, and hardware bias.",
            "anchor_failure": "Two fixed anchors offline plus random dropout.",
            "trajectories": "Configured continuous paths for replay and filtering.",
        },
    }
    LOGGER.info(
        "Generated dataset profile=%s seed=%d with split sizes %s",
        metadata["profile"],
        root_seed,
        metadata["split_sizes"],
    )
    return splits, metadata


def save_dataset(
    splits: Mapping[str, pd.DataFrame],
    metadata: Mapping[str, Any],
    output_dir: str | Path,
) -> dict[str, Path]:
    """Save inspectable CSVs, compact NumPy arrays, and JSON metadata."""

    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, frame in splits.items():
        path = destination / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[name] = path
    anchor_ids = [str(value) for value in metadata.get("anchor_ids", [])]
    array_payload: dict[str, np.ndarray] = {}
    for name, frame in splits.items():
        if {"true_x", "true_y"}.issubset(frame.columns):
            array_payload[f"{name}__true_position"] = frame[
                ["true_x", "true_y"]
            ].to_numpy(dtype=float)
        for prefix, dtype in (
            ("rss", float),
            ("true_distance", float),
            ("estimated_distance", float),
            ("los", np.uint8),
            ("available", np.uint8),
        ):
            columns = [
                f"{prefix}_{anchor_id}"
                for anchor_id in anchor_ids
                if f"{prefix}_{anchor_id}" in frame
            ]
            if columns:
                values = frame[columns]
                if prefix in {"los", "available"}:
                    values = values.fillna(False)
                array_payload[f"{name}__{prefix}"] = values.to_numpy(dtype=dtype)
    arrays_path = destination / "dataset_arrays.npz"
    np.savez_compressed(arrays_path, **array_payload)
    paths["npz"] = arrays_path
    metadata_path = destination / "metadata.json"
    with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(dict(metadata), handle, indent=2, ensure_ascii=False, default=str)
    paths["metadata"] = metadata_path
    return paths


generate_datasets = generate_dataset
