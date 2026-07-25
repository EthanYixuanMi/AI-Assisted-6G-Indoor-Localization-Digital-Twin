"""End-to-end experiment orchestration for the localization digital twin."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from localization_twin.config import load_config
from localization_twin.dataset import (
    SPLIT_SEED_OFFSETS,
    generate_dataset,
    save_dataset,
)
from localization_twin.environment import Environment
from localization_twin.evaluation.metrics import (
    add_baseline_improvements,
    evaluate_frame,
    euclidean_errors,
    measure_inference_time,
)
from localization_twin.models import (
    ABLATION_FEATURE_GROUPS,
    DirectAILocator,
    GeometricLocator,
    KNNFingerprintLocator,
    KalmanFilter2D,
    ResidualAILocator,
)
from localization_twin.propagation import PropagationModel
from localization_twin.reporting import (
    build_manifest,
    export_report_tables_and_text,
)


LOGGER = logging.getLogger(__name__)
ALGORITHM_ORDER = ("Geometric LS", "KNN", "Direct AI", "Residual AI")
SCENARIOS = (
    "normal",
    "high_noise",
    "strong_blockage",
    "anchor_failure",
    "domain_shift",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _config_sha256(config: Mapping[str, Any]) -> str:
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _timestamped_run_dir(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = root / "outputs" / f"run_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / "outputs" / f"run_{stamp}_{suffix:02d}"
        suffix += 1
    return candidate


def _validate_removal_target(target: Path, workspace: Path) -> None:
    resolved = target.resolve()
    workspace_resolved = workspace.resolve()
    if resolved in {workspace_resolved, workspace_resolved.parent, Path(resolved.anchor)}:
        raise ValueError(f"Unsafe directory target: {resolved}")
    if workspace_resolved not in resolved.parents:
        raise ValueError(f"Refusing to replace a directory outside the workspace: {resolved}")


def _prepare_run_dir(path: Path, *, root: Path, force: bool) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        if not force:
            raise FileExistsError(
                f"Output directory is not empty: {destination}. Use --force to replace it."
            )
        _validate_removal_target(destination, root)
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for child in ("figures", "tables", "screenshots", "latex", "logs", "data", "models"):
        (destination / child).mkdir(parents=True, exist_ok=True)
    return destination


def _configure_logging(log_path: Path) -> None:
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        handler.close()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(stream)
    root_logger.addHandler(file_handler)


def _anchors_for_geometric(environment: Environment) -> dict[str, tuple[float, float]]:
    return {anchor.anchor_id: anchor.position for anchor in environment.anchors}


def _geometric_details(model: GeometricLocator, frame: pd.DataFrame) -> pd.DataFrame:
    details = model.predict_details(frame)
    if not isinstance(details, pd.DataFrame):
        raise TypeError("GeometricLocator.predict_details must return a DataFrame")
    return details


def _prediction_array(details: pd.DataFrame) -> np.ndarray:
    return details.loc[:, ["geometric_x", "geometric_y"]].to_numpy(dtype=float)


def _train_models(
    config: Mapping[str, Any],
    splits: Mapping[str, pd.DataFrame],
    environment: Environment,
    model_dir: Path,
    seed: int,
) -> tuple[dict[str, Any], dict[str, float], dict[str, float], dict[str, Any]]:
    """Fit all point estimators and serialize them with training metadata."""

    train = splits["train"]
    validation = splits["validation"]
    anchors = _anchors_for_geometric(environment)
    bounds = ([0.0, 0.0], [environment.width, environment.height])
    geometric = GeometricLocator(
        anchors,
        anchor_ids=environment.anchor_ids,
        bounds=bounds,
        use_rss_weights=True,
        max_nfev=60,
    )
    started = time.perf_counter()
    geometric.fit(train)
    geometric_training = time.perf_counter() - started
    LOGGER.info("Computing geometric training features for %d samples", len(train))
    geometric_train = _geometric_details(geometric, train)
    geometric_validation = _geometric_details(geometric, validation)

    knn = KNNFingerprintLocator(
        environment.anchor_ids,
        k_candidates=(3, 5, 7, 11),
        weights="distance",
        n_jobs=1,
    )
    knn.fit(train, validation_frame=validation)

    model_config = config.get("models", {})
    direct_config = model_config.get("direct_ai", {})
    hidden = tuple(direct_config.get("hidden_layer_sizes", (64, 32)))
    direct_candidates = (
        {
            "hidden_layer_sizes": (48,),
            "alpha": 5e-4,
            "learning_rate_init": 2e-3,
        },
        {
            "hidden_layer_sizes": hidden,
            "alpha": 1e-3,
            "learning_rate_init": 1e-3,
        },
    )
    direct = DirectAILocator(
        environment.anchor_ids,
        include_los=True,
        configs=direct_candidates,
        random_state=seed,
        max_iter=int(direct_config.get("max_iter", 220)),
    )
    direct.fit(train, validation_frame=validation)

    residual_config = model_config.get("residual_ai", {})
    residual = ResidualAILocator(
        environment.anchor_ids,
        estimator="extra_trees",
        n_estimators=int(residual_config.get("n_estimators", 120)),
        max_depth=int(residual_config.get("max_depth", 18)),
        random_state=seed,
        n_jobs=1,
    )
    residual.fit(
        train,
        geometric_train,
        validation_frame=validation,
        validation_geometric=geometric_validation,
    )

    kalman = KalmanFilter2D(dt=1.0, process_noise=0.15, measurement_noise=1.5)
    models: dict[str, Any] = {
        "Geometric LS": geometric,
        "KNN": knn,
        "Direct AI": direct,
        "Residual AI": residual,
        "Residual AI + Kalman": kalman,
    }
    file_names = {
        "Geometric LS": "geometric_ls.joblib",
        "KNN": "knn.joblib",
        "Direct AI": "direct_ai.joblib",
        "Residual AI": "residual_ai.joblib",
        "Residual AI + Kalman": "kalman.joblib",
    }
    training_times = {
        "Geometric LS": geometric_training,
        "KNN": float(knn.training_time_s_ or 0.0),
        "Direct AI": float(direct.training_time_s_ or 0.0),
        "Residual AI": float(residual.training_time_s_ or 0.0),
        "Residual AI + Kalman": float(residual.training_time_s_ or 0.0),
    }
    model_sizes: dict[str, float] = {}
    for name, model in models.items():
        model_path = model_dir / file_names[name]
        model.save(model_path)
        model_sizes[name] = model_path.stat().st_size / (1024.0 * 1024.0)
    model_sizes["Residual AI + Kalman"] += model_sizes["Residual AI"]

    metadata = {
        "model_schema_version": 1,
        "config_sha256": _config_sha256(config),
        "seed": seed,
        "anchor_ids": list(environment.anchor_ids),
        "anchor_positions": environment.anchor_positions.tolist(),
        "training_columns": list(train.columns),
        "knn_best_k": knn.best_k,
        "knn_validation_scores": knn.validation_scores_,
        "direct_ai_best_config": direct.best_config_,
        "direct_ai_validation_scores": direct.validation_scores_,
        "residual_estimator": residual.selected_estimator_,
        "residual_validation_scores": residual.validation_scores_,
        "training_time_s": training_times,
        "model_size_mb": model_sizes,
        "model_files": file_names,
    }
    return models, training_times, model_sizes, metadata


def _load_models(
    environment: Environment,
    source_dir: Path,
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, float], dict[str, float], dict[str, Any]]:
    metadata_path = source_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            "--skip-training requires models/metadata.json for compatibility checks"
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    expected_hash = _config_sha256(config)
    if metadata.get("config_sha256") != expected_hash:
        raise ValueError(
            "--skip-training model/config mismatch: resolved configuration hash "
            "differs from models/metadata.json"
        )
    if metadata.get("anchor_ids") != list(environment.anchor_ids):
        raise ValueError("--skip-training model anchor IDs do not match the environment")
    recorded_positions = np.asarray(metadata.get("anchor_positions"), dtype=float)
    if (
        recorded_positions.shape != environment.anchor_positions.shape
        or not np.allclose(recorded_positions, environment.anchor_positions)
    ):
        raise ValueError(
            "--skip-training model anchor coordinates do not match the environment"
        )
    required = {
        "Geometric LS": (GeometricLocator, "geometric_ls.joblib"),
        "KNN": (KNNFingerprintLocator, "knn.joblib"),
        "Direct AI": (DirectAILocator, "direct_ai.joblib"),
        "Residual AI": (ResidualAILocator, "residual_ai.joblib"),
        "Residual AI + Kalman": (KalmanFilter2D, "kalman.joblib"),
    }
    models: dict[str, Any] = {}
    sizes: dict[str, float] = {}
    for name, (model_type, file_name) in required.items():
        path = source_dir / file_name
        if not path.is_file():
            raise FileNotFoundError(
                f"--skip-training requested, but model is missing: {path}"
            )
        models[name] = model_type.load(path)
        sizes[name] = path.stat().st_size / (1024.0 * 1024.0)
    sizes["Residual AI + Kalman"] += sizes["Residual AI"]
    training = {
        name: float(metadata.get("training_time_s", {}).get(name, 0.0))
        for name in models
    }
    return models, training, sizes, metadata


def _copy_models_to_repository(run_model_dir: Path, repository_model_dir: Path) -> None:
    repository_model_dir.mkdir(parents=True, exist_ok=True)
    for path in run_model_dir.iterdir():
        if path.is_file():
            shutil.copy2(path, repository_model_dir / path.name)


def _predict_all(
    frame: pd.DataFrame,
    models: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    geometric_details = _geometric_details(models["Geometric LS"], frame)
    predictions = {
        "Geometric LS": _prediction_array(geometric_details),
        "KNN": models["KNN"].predict(frame),
        "Direct AI": models["Direct AI"].predict(frame),
        "Residual AI": models["Residual AI"].predict(frame, geometric_details),
    }
    return predictions, geometric_details


def _filter_by_trajectory(
    frame: pd.DataFrame,
    predictions: np.ndarray,
    kalman: KalmanFilter2D,
) -> np.ndarray:
    filtered = np.empty_like(predictions, dtype=float)
    if "trajectory_id" not in frame:
        return kalman.filter(predictions, reset=True)
    trajectory_ids = frame["trajectory_id"].astype(str).to_numpy()
    for trajectory_id in pd.unique(trajectory_ids):
        indices = np.flatnonzero(trajectory_ids == trajectory_id)
        order = indices
        if "timestep" in frame:
            order = indices[
                np.argsort(
                    pd.to_numeric(frame.iloc[indices]["timestep"], errors="coerce")
                    .fillna(0)
                    .to_numpy()
                )
            ]
        filtered[order] = kalman.filter(predictions[order], reset=True)
    return filtered


def _measure_prediction_times(
    frame: pd.DataFrame,
    models: Mapping[str, Any],
) -> dict[str, dict[str, float | int]]:
    geometric = models["Geometric LS"]
    geometric_details = _geometric_details(geometric, frame)
    timings: dict[str, dict[str, float | int]] = {}

    predictors: dict[str, Callable[[pd.DataFrame], np.ndarray]] = {
        "Geometric LS": lambda data: geometric.predict(data),
        "KNN": lambda data: models["KNN"].predict(data),
        "Direct AI": lambda data: models["Direct AI"].predict(data),
        "Residual AI": lambda data: models["Residual AI"].predict(
            data, _geometric_details(geometric, data)
        ),
        "Residual AI + Kalman": lambda data: models[
            "Residual AI + Kalman"
        ].filter(
            models["Residual AI"].predict(
                data, _geometric_details(geometric, data)
            ),
            reset=True,
        ),
    }
    one_row = frame.iloc[:1].copy()
    for algorithm, predictor in predictors.items():
        single = measure_inference_time(
            predictor,
            one_row,
            warmup_runs=3,
            repeats=15,
        )
        batch = measure_inference_time(
            predictor,
            frame,
            warmup_runs=1,
            repeats=3,
        )
        timings[algorithm] = {
            "sample_count": int(batch["sample_count"]),
            "warmup_runs": int(single["warmup_runs"]),
            "timing_repeats": int(single["timing_repeats"]),
            "inference_time_ms": float(single["batch_inference_time_ms"]),
            "batch_inference_time_ms": float(batch["batch_inference_time_ms"]),
            "batch_amortized_time_ms": float(batch["inference_time_ms"]),
        }
    return timings


def _scenario_frame(
    base_config: Mapping[str, Any],
    *,
    profile: str,
    scenario: str,
    positions: np.ndarray,
    seed: int,
    split: str = "spatial_holdout",
    trajectory_ids: Sequence[str] | None = None,
    timesteps: Sequence[int | float] | None = None,
    config_path: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any], Environment]:
    if scenario == "normal":
        config = deepcopy(dict(base_config))
    else:
        # Re-resolve with the same caller overlay used by the root pipeline.
        # This preserves custom maps and model/propagation overrides across all
        # scenario evaluations instead of silently reverting to repository
        # defaults outside the Normal case.
        config = load_config(
            profile=profile,
            scenario=scenario,
            config_path=config_path,
        )
    environment = Environment.from_config(config)
    simulator = PropagationModel(environment, config, seed=seed + 20_000)
    frame = simulator.simulate_positions(
        positions,
        split=split,
        scenario_name=scenario,
        trajectory_ids=trajectory_ids,
        timesteps=timesteps,
        random_seed=seed,
    )
    return frame, config, environment


def _telemetry_aliases(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in list(output.columns):
        if column.startswith("estimated_distance_"):
            alias = column.replace("estimated_distance_", "est_distance_", 1)
            output[alias] = output[column]
    return output


def _prediction_records(
    frame: pd.DataFrame,
    predictions: Mapping[str, np.ndarray],
    *,
    scenario: str,
    seed: int,
) -> pd.DataFrame:
    base = _telemetry_aliases(frame).reset_index(drop=True)
    # The simulator retains provenance as scenario_name/random_seed.  The
    # visualization contract uses scenario/seed, so remove the aliases before
    # adding canonical columns and avoid duplicate column names after loading.
    base = base.drop(
        columns=[
            column
            for column in ("scenario_name", "random_seed")
            if column in base.columns
        ]
    )
    records: list[pd.DataFrame] = []
    truth = base[["true_x", "true_y"]].to_numpy(dtype=float)
    for algorithm, estimate in predictions.items():
        if estimate.shape != (len(base), 2):
            raise ValueError(f"Unexpected prediction shape for {algorithm}: {estimate.shape}")
        current = base.copy()
        current.insert(0, "algorithm", algorithm)
        current["scenario"] = scenario
        current["seed"] = int(seed)
        current["pred_x"] = estimate[:, 0]
        current["pred_y"] = estimate[:, 1]
        current["error"] = euclidean_errors(truth, estimate)
        if "trajectory_id" not in current:
            current["trajectory_id"] = current.get("split", "static")
        if "timestep" not in current:
            current["timestep"] = np.arange(len(current))
        records.append(current)
    return pd.concat(records, ignore_index=True, sort=False)


def _metric_row(
    frame: pd.DataFrame,
    prediction: np.ndarray,
    *,
    algorithm: str,
    scenario: str,
    seed: int,
    timing: Mapping[str, float | int],
    training_time: float,
    model_size: float,
    evaluation_split: str = "spatial_holdout",
) -> dict[str, Any]:
    result = evaluate_frame(
        frame,
        prediction,
        algorithm=algorithm,
        scenario=scenario,
        seed=seed,
    )
    result.update(
        {
            "evaluation_split": evaluation_split,
            "inference_time_ms": float(timing["inference_time_ms"]),
            "training_time_s": float(training_time),
            "model_size_mb": float(model_size),
        }
    )
    return result


def _main_evaluation(
    config: Mapping[str, Any],
    profile: str,
    splits: Mapping[str, pd.DataFrame],
    models: Mapping[str, Any],
    seed: int,
    training_times: Mapping[str, float],
    model_sizes: Mapping[str, float],
    config_path: str | Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, float | int]], dict[str, pd.DataFrame]]:
    base_holdout = splits["spatial_holdout"]
    positions = base_holdout[["true_x", "true_y"]].to_numpy(dtype=float)
    trajectory_base = splits["trajectories"]
    trajectory_positions = trajectory_base[["true_x", "true_y"]].to_numpy(dtype=float)
    trajectory_ids = trajectory_base["trajectory_id"].astype(str).tolist()
    timesteps = trajectory_base["timestep"].tolist()

    normal_timings = _measure_prediction_times(base_holdout, models)
    metric_rows: list[dict[str, Any]] = []
    prediction_frames: list[pd.DataFrame] = []
    scenario_frames: dict[str, pd.DataFrame] = {}

    # Keep the ordinary in-domain test distinct from the stricter spatial
    # holdout used for the report headline.
    in_domain = splits["test"]
    in_domain_predictions, _ = _predict_all(in_domain, models)
    for algorithm in ALGORITHM_ORDER:
        metric_rows.append(
            _metric_row(
                in_domain,
                in_domain_predictions[algorithm],
                algorithm=algorithm,
                scenario="in_domain",
                seed=seed,
                timing=normal_timings[algorithm],
                training_time=training_times[algorithm],
                model_size=model_sizes[algorithm],
                evaluation_split="in_domain_test",
            )
        )

    for scenario in SCENARIOS:
        frame, _, _ = _scenario_frame(
            config,
            profile=profile,
            scenario=scenario,
            positions=positions,
            seed=seed,
            config_path=config_path,
        )
        scenario_frames[scenario] = frame
        estimates, _ = _predict_all(frame, models)
        for algorithm in ALGORITHM_ORDER:
            metric_rows.append(
                _metric_row(
                    frame,
                    estimates[algorithm],
                    algorithm=algorithm,
                    scenario=scenario,
                    seed=seed,
                    timing=normal_timings[algorithm],
                    training_time=training_times[algorithm],
                    model_size=model_sizes[algorithm],
                )
            )
        prediction_frames.append(
            _prediction_records(frame, estimates, scenario=scenario, seed=seed)
        )

        trajectory_frame, _, _ = _scenario_frame(
            config,
            profile=profile,
            scenario=scenario,
            positions=trajectory_positions,
            seed=seed + 5_000,
            split="trajectories",
            trajectory_ids=trajectory_ids,
            timesteps=timesteps,
            config_path=config_path,
        )
        trajectory_estimates, _ = _predict_all(trajectory_frame, models)
        trajectory_estimates["Residual AI + Kalman"] = _filter_by_trajectory(
            trajectory_frame,
            trajectory_estimates["Residual AI"],
            models["Residual AI + Kalman"],
        )
        prediction_frames.append(
            _prediction_records(
                trajectory_frame,
                trajectory_estimates,
                scenario=scenario,
                seed=seed,
            )
        )

    metrics = pd.DataFrame(metric_rows)
    metrics = add_baseline_improvements(
        metrics,
        baseline_algorithm="Geometric LS",
        group_columns=("scenario", "seed"),
    )
    per_sample = pd.concat(prediction_frames, ignore_index=True, sort=False)
    return metrics, per_sample, normal_timings, scenario_frames


def _seed_evaluation(
    config: Mapping[str, Any],
    profile: str,
    root_splits: Mapping[str, pd.DataFrame],
    root_models: Mapping[str, Any],
    root_seed: int,
    seeds: Sequence[int],
    timings: Mapping[str, Mapping[str, float | int]],
    root_training_times: Mapping[str, float],
    root_model_sizes: Mapping[str, float],
    environment: Environment,
    run_model_dir: Path,
    *,
    skip_training: bool,
) -> pd.DataFrame:
    """Run independent train/validation/holdout repetitions by seed.

    When training is enabled, each seed receives its own sampled splits,
    model initialization, validation selection, and serialized model bundle.
    ``--skip-training`` intentionally changes the scope to a fixed saved model
    under new simulated measurements and labels that scope in the CSV.
    """

    rows: list[dict[str, Any]] = []
    for evaluation_seed in seeds:
        current_seed = int(evaluation_seed)
        if current_seed == root_seed:
            current_splits = root_splits
            current_models = root_models
            current_training_times = root_training_times
            current_model_sizes = root_model_sizes
            seed_scope = (
                f"fixed_model_seed_{root_seed}"
                if skip_training
                else "independent_training"
            )
        elif skip_training:
            current_splits, _ = generate_dataset(
                config=config,
                profile=profile,
                seed=current_seed,
                scenario="normal",
            )
            current_models = root_models
            current_training_times = root_training_times
            current_model_sizes = root_model_sizes
            seed_scope = f"fixed_model_seed_{root_seed}"
        else:
            current_splits, current_metadata = generate_dataset(
                config=config,
                profile=profile,
                seed=current_seed,
                scenario="normal",
            )
            seed_model_dir = run_model_dir / f"seed_{current_seed}"
            seed_model_dir.mkdir(parents=True, exist_ok=True)
            (
                current_models,
                current_training_times,
                current_model_sizes,
                seed_training_metadata,
            ) = _train_models(
                config,
                current_splits,
                environment,
                seed_model_dir,
                current_seed,
            )
            _write_json(seed_model_dir / "metadata.json", seed_training_metadata)
            _write_json(seed_model_dir / "dataset_metadata.json", current_metadata)
            seed_scope = "independent_training"

        frame = current_splits["spatial_holdout"]
        predictions, _ = _predict_all(frame, current_models)
        for algorithm in ALGORITHM_ORDER:
            row = _metric_row(
                frame,
                predictions[algorithm],
                algorithm=algorithm,
                scenario="normal",
                seed=current_seed,
                timing=timings[algorithm],
                training_time=current_training_times[algorithm],
                model_size=current_model_sizes[algorithm],
            )
            row["evaluation_split"] = "spatial_holdout"
            row["seed_scope"] = seed_scope
            rows.append(row)
    result = pd.DataFrame(rows)
    return add_baseline_improvements(
        result,
        baseline_algorithm="Geometric LS",
        group_columns=("scenario", "seed"),
    )


def _simulate_modified(
    config: Mapping[str, Any],
    positions: np.ndarray,
    *,
    seed: int,
    scenario_name: str,
) -> pd.DataFrame:
    environment = Environment.from_config(config)
    simulator = PropagationModel(environment, config, seed=seed + 30_000)
    return simulator.simulate_positions(
        positions,
        split="robustness",
        scenario_name=scenario_name,
        random_seed=seed,
    )


def _robustness_evaluation(
    config: Mapping[str, Any],
    positions: np.ndarray,
    models: Mapping[str, Any],
    seeds: Sequence[int],
    training_seed: int,
    *,
    skip_robustness: bool,
) -> pd.DataFrame:
    positions = positions[: min(len(positions), 450)]
    experiment_config = config.get("experiment", {})
    noise_levels = (
        [float(config["propagation"]["noise_std"])]
        if skip_robustness
        else [
            float(value)
            for value in experiment_config.get(
                "noise_levels_db", (0.5, 1, 2, 3, 5, 7)
            )
        ]
    )
    random_anchor_levels = (
        [0]
        if skip_robustness
        else [
            int(value)
            for value in experiment_config.get(
                "random_anchor_failure_counts", (0, 1, 2, 3)
            )
        ]
    )
    wall_levels = (
        [1.0]
        if skip_robustness
        else [
            float(value)
            for value in experiment_config.get(
                "wall_loss_multipliers", (0.75, 1.0, 1.35, 1.65)
            )
        ]
    )
    active_seeds = list(seeds[:1] if skip_robustness else seeds)
    rows: list[dict[str, Any]] = []

    def evaluate_condition(
        frame: pd.DataFrame,
        *,
        experiment: str,
        scenario: str,
        level: float | int,
        evaluation_seed: int,
        failure_mode: str = "not_applicable",
    ) -> None:
        predictions, _ = _predict_all(frame, models)
        for algorithm, prediction in predictions.items():
            metrics = evaluate_frame(frame, prediction)
            rows.append(
                {
                    "experiment": experiment,
                    "algorithm": algorithm,
                    "scenario": scenario,
                    "seed": int(evaluation_seed),
                    "measurement_seed": int(evaluation_seed),
                    "model_training_seed": int(training_seed),
                    "seed_scope": "fixed_root_model_across_measurement_seeds",
                    "level": level,
                    "failure_mode": failure_mode,
                    "sample_count": metrics["sample_count"],
                    "mean_error": metrics["mean_error"],
                    "rmse": metrics["rmse"],
                    "median_error": metrics["median_error"],
                    "p90_error": metrics["p90_error"],
                    "max_error": metrics["max_error"],
                    "los_mean_error": metrics["los_mean_error"],
                    "nlos_mean_error": metrics["nlos_mean_error"],
                }
            )

    for evaluation_seed in active_seeds:
        for level in noise_levels:
            modified = deepcopy(dict(config))
            modified["propagation"]["noise_std"] = float(level)
            frame = _simulate_modified(
                modified,
                positions,
                seed=int(evaluation_seed),
                scenario_name="noise",
            )
            evaluate_condition(
                frame,
                experiment="noise",
                scenario="RSS noise",
                level=level,
                evaluation_seed=int(evaluation_seed),
            )

        for level in wall_levels:
            modified = deepcopy(dict(config))
            modified["propagation"]["wall_loss_multiplier"] = float(level)
            modified["propagation"]["nlos_bias_mean"] = (
                float(config["propagation"]["nlos_bias_mean"]) * float(level)
            )
            frame = _simulate_modified(
                modified,
                positions,
                seed=int(evaluation_seed),
                scenario_name="wall_attenuation",
            )
            evaluate_condition(
                frame,
                experiment="wall_attenuation",
                scenario="Wall/NLoS severity",
                level=level,
                evaluation_seed=int(evaluation_seed),
            )

        anchor_ids = list(Environment.from_config(config).anchor_ids)
        rng = np.random.default_rng(int(evaluation_seed) + 81)
        random_order = list(rng.permutation(anchor_ids))
        configured_critical = [
            str(value)
            for value in experiment_config.get(
                "fixed_anchor_failure_ids", anchor_ids[:2]
            )
        ]
        unknown_critical = set(configured_critical).difference(anchor_ids)
        if unknown_critical:
            raise ValueError(
                "experiment.fixed_anchor_failure_ids contains unknown anchors: "
                f"{sorted(unknown_critical)}"
            )
        modified = deepcopy(dict(config))
        modified["propagation"]["dropout_probability"] = 0.0
        modified["propagation"]["fixed_offline_anchors"] = []
        clean_frame = _simulate_modified(
            modified,
            positions,
            seed=int(evaluation_seed),
            scenario_name="anchor_failure",
        )
        failure_specs = (
            ("random", random_order, random_anchor_levels),
            (
                "fixed_critical",
                configured_critical,
                [0]
                if skip_robustness
                else list(range(len(configured_critical) + 1)),
            ),
        )
        for failure_mode, order, levels in failure_specs:
            for level in levels:
                if int(level) > len(order):
                    raise ValueError(
                        f"Anchor failure level {level} exceeds {len(order)} available IDs"
                    )
                frame = _apply_anchor_failures(
                    clean_frame,
                    anchor_ids,
                    order[: int(level)],
                )
                evaluate_condition(
                    frame,
                    experiment="anchor_failure",
                    scenario=f"{failure_mode} anchor failure",
                    level=int(level),
                    evaluation_seed=int(evaluation_seed),
                    failure_mode=failure_mode,
                )
    result = pd.DataFrame(rows)
    return add_baseline_improvements(
        result,
        baseline_algorithm="Geometric LS",
        group_columns=(
            "experiment",
            "scenario",
            "level",
            "failure_mode",
            "seed",
        ),
    )


def _apply_anchor_failures(
    frame: pd.DataFrame,
    anchor_ids: Sequence[str],
    offline_ids: Sequence[str],
) -> pd.DataFrame:
    """Mask selected anchors without perturbing measurements on other links."""

    output = frame.copy()
    offline = set(offline_ids)
    for anchor_id in anchor_ids:
        if anchor_id not in offline:
            continue
        output[f"available_{anchor_id}"] = False
        output[f"rss_{anchor_id}"] = np.nan
        output[f"estimated_distance_{anchor_id}"] = np.nan
        output[f"los_{anchor_id}"] = False
    availability = np.column_stack(
        [
            output[f"available_{anchor_id}"].astype(bool).to_numpy()
            for anchor_id in anchor_ids
        ]
    )
    los = np.column_stack(
        [
            output[f"los_{anchor_id}"].astype(bool).to_numpy()
            for anchor_id in anchor_ids
        ]
    )
    output["nlos_anchor_count"] = np.sum(availability & ~los, axis=1)
    output["sample_class"] = np.where(
        output["nlos_anchor_count"].to_numpy() > 0, "NLoS", "LoS"
    )
    output["anchor_availability_mask"] = [
        json.dumps(row.astype(int).tolist(), separators=(",", ":"))
        for row in availability
    ]
    return output


def _ablation_evaluation(
    config: Mapping[str, Any],
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    trajectories: pd.DataFrame,
    models: Mapping[str, Any],
    seed: int,
) -> pd.DataFrame:
    geometric = models["Geometric LS"]
    geo_train = _geometric_details(geometric, train)
    geo_validation = _geometric_details(geometric, validation)
    geo_test = _geometric_details(geometric, test)
    rows: list[dict[str, Any]] = []

    full_prediction = models["Residual AI"].predict(test, geo_test)
    full_metrics = evaluate_frame(test, full_prediction)
    rows.append(
        {
            "variant": "full",
            "algorithm": "Residual AI",
            "scenario": "normal",
            "evaluation_split": "spatial_holdout",
            "seed": seed,
            **full_metrics,
        }
    )

    residual_config = config.get("models", {}).get("residual_ai", {})
    ablation_trees = int(residual_config.get("n_estimators", 120))
    ablation_depth = int(residual_config.get("max_depth", 18))
    for variant in (
        "without_los_nlos",
        "without_geometric_residual",
        "without_anchor_mask",
    ):
        model = ResidualAILocator(
            Environment.from_config(config).anchor_ids,
            estimator="extra_trees",
            feature_groups=ABLATION_FEATURE_GROUPS[variant],
            n_estimators=ablation_trees,
            max_depth=ablation_depth,
            random_state=seed,
            n_jobs=1,
        )
        model.fit(
            train,
            geo_train,
            validation_frame=validation,
            validation_geometric=geo_validation,
        )
        prediction = model.predict(test, geo_test)
        rows.append(
            {
                "variant": variant,
                "algorithm": "Residual AI",
                "scenario": "normal",
                "evaluation_split": "spatial_holdout",
                "seed": seed,
                **evaluate_frame(test, prediction),
            }
        )

    no_bias_config = deepcopy(dict(config))
    no_bias_config["propagation"]["spatial_bias"]["enabled"] = False
    no_bias_environment = Environment.from_config(no_bias_config)
    no_bias_train = PropagationModel(
        no_bias_environment,
        no_bias_config,
        seed=seed + SPLIT_SEED_OFFSETS["train"] + 10_000,
    ).simulate_positions(
        train[["true_x", "true_y"]].to_numpy(dtype=float),
        split="train",
        scenario_name="training_without_spatial_bias",
        random_seed=seed,
    )
    no_bias_validation = PropagationModel(
        no_bias_environment,
        no_bias_config,
        seed=seed + SPLIT_SEED_OFFSETS["validation"] + 10_000,
    ).simulate_positions(
        validation[["true_x", "true_y"]].to_numpy(dtype=float),
        split="validation",
        scenario_name="validation_without_spatial_bias",
        random_seed=seed,
    )
    no_bias_geo_train = _geometric_details(geometric, no_bias_train)
    no_bias_geo_validation = _geometric_details(geometric, no_bias_validation)
    no_bias_model = ResidualAILocator(
        Environment.from_config(config).anchor_ids,
        estimator="extra_trees",
        n_estimators=ablation_trees,
        max_depth=ablation_depth,
        random_state=seed,
        n_jobs=1,
    )
    no_bias_model.fit(
        no_bias_train,
        no_bias_geo_train,
        validation_frame=no_bias_validation,
        validation_geometric=no_bias_geo_validation,
    )
    no_bias_prediction = no_bias_model.predict(test, geo_test)
    rows.append(
        {
            "variant": "without_spatial_bias_training",
            "algorithm": "Residual AI",
            "scenario": "normal",
            "evaluation_split": "spatial_holdout",
            "seed": seed,
            **evaluate_frame(test, no_bias_prediction),
        }
    )

    trajectory_predictions, _ = _predict_all(trajectories, models)
    residual_trajectory = trajectory_predictions["Residual AI"]
    filtered = _filter_by_trajectory(
        trajectories, residual_trajectory, models["Residual AI + Kalman"]
    )
    rows.append(
        {
            "variant": "without_kalman_filter",
            "algorithm": "Residual AI",
            "scenario": "normal",
            "evaluation_split": "trajectories",
            "seed": seed,
            **evaluate_frame(trajectories, residual_trajectory),
        }
    )
    rows.append(
        {
            "variant": "with_kalman_filter",
            "algorithm": "Residual AI + Kalman",
            "scenario": "normal",
            "evaluation_split": "trajectories",
            "seed": seed,
            **evaluate_frame(trajectories, filtered),
        }
    )
    result = pd.DataFrame(rows)
    geometric_baselines = {
        "spatial_holdout": float(
            evaluate_frame(test, _prediction_array(geo_test))["mean_error"]
        ),
        "trajectories": float(
            evaluate_frame(
                trajectories, trajectory_predictions["Geometric LS"]
            )["mean_error"]
        ),
    }
    result["mean_error_geometric_baseline"] = result["evaluation_split"].map(
        geometric_baselines
    )
    result["mean_error_change_vs_geometric"] = (
        result["mean_error"] - result["mean_error_geometric_baseline"]
    )
    result["mean_error_absolute_improvement"] = (
        result["mean_error_geometric_baseline"] - result["mean_error"]
    )
    result["mean_error_percent_improvement"] = (
        100.0
        * result["mean_error_absolute_improvement"]
        / result["mean_error_geometric_baseline"].replace(0.0, np.nan)
    )
    return result


def _dashboard_data_prepare_time(per_sample: pd.DataFrame, algorithm: str) -> float:
    """Measure saved-frame selection and KPI preparation, excluding UI render."""

    subset = per_sample.loc[
        (per_sample["algorithm"] == algorithm)
        & (per_sample["scenario"] == "normal")
        & (per_sample["split"] == "trajectories")
    ].head(100)
    if subset.empty:
        subset = per_sample.loc[per_sample["algorithm"] == algorithm].head(100)
    repeats = 100
    elapsed: list[float] = []
    availability_columns = [
        column for column in subset.columns if column.startswith("available_")
    ]
    for index in range(repeats):
        started = time.perf_counter()
        row = subset.iloc[index % len(subset)]
        current_timestep = row["timestep"]
        current = per_sample.loc[
            (per_sample["scenario"] == row["scenario"])
            & (per_sample["split"] == row["split"])
            & (per_sample["timestep"] == current_timestep)
        ]
        errors = pd.to_numeric(current["error"], errors="coerce").dropna()
        _ = {
            "x": float(row["pred_x"]),
            "y": float(row["pred_y"]),
            "error": float(row["error"]),
            "mean_error": float(errors.mean()),
            "median_error": float(errors.median()),
            "p90_error": float(errors.quantile(0.90)),
            "available": int(
                sum(bool(row[column]) for column in availability_columns)
            ),
        }
        elapsed.append((time.perf_counter() - started) * 1000.0)
    return float(np.median(elapsed))


def _runtime_table(
    timings: Mapping[str, Mapping[str, float | int]],
    training_times: Mapping[str, float],
    model_sizes: Mapping[str, float],
    per_sample: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for algorithm in (*ALGORITHM_ORDER, "Residual AI + Kalman"):
        timing = timings[algorithm]
        rows.append(
            {
                "algorithm": algorithm,
                "training_time_s": float(training_times[algorithm]),
                "inference_time_ms": float(timing["inference_time_ms"]),
                "batch_inference_time_ms": float(timing["batch_inference_time_ms"]),
                "batch_amortized_time_ms": float(timing["batch_amortized_time_ms"]),
                "model_size_mb": float(model_sizes[algorithm]),
                "dashboard_update_ms": _dashboard_data_prepare_time(
                    per_sample, algorithm
                ),
                "dashboard_update_scope": (
                    "saved-frame selection and KPI preparation; excludes "
                    "Streamlit and Plotly rendering"
                ),
            }
        )
    return pd.DataFrame(rows)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )


def _sync_directory(source: Path, destination: Path, workspace: Path) -> None:
    if destination.exists():
        _validate_removal_target(destination, workspace)
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def _sync_report_assets(run_dir: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for directory_name in ("figures", "screenshots", "tables", "latex", "data"):
        source = run_dir / directory_name
        target = destination / directory_name
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    for file_name in (
        "REPORT_RESULTS_DRAFT.md",
        "RESULTS_SUMMARY.md",
    ):
        shutil.copy2(run_dir / file_name, destination / file_name)


def _run_internal_output_checks(run_dir: Path) -> dict[str, Any]:
    required_csv = (
        "metrics.csv",
        "per_sample_predictions.csv",
        "per_seed_results.csv",
        "robustness_results.csv",
        "ablation_results.csv",
        "runtime_results.csv",
    )
    required_figure_stems = (
        "trajectory_comparison.png",
        "error_cdf.png",
        "spatial_error_heatmaps.png",
        "robustness_noise.png",
        "robustness_anchor_failure.png",
        "robustness_results.png",
        "los_nlos_comparison.png",
        "ablation_results.png",
        "runtime_comparison.png",
    )
    required_figures = tuple(
        name
        for png_name in required_figure_stems
        for name in (png_name, png_name.removesuffix(".png") + ".pdf")
    )
    required_screenshots = tuple(f"dashboard_{name}.png" for name in SCENARIOS)
    required_latex = (
        "table_simulation_parameters.tex",
        "table_main_results.tex",
        "table_los_nlos.tex",
        "table_robustness.tex",
        "table_ablation.tex",
        "table_trajectory_filtering.tex",
        "table_runtime.tex",
        "preliminary_results_snippet.tex",
    )
    required_misc = (
        Path("figures") / "dashboard_overview.png",
        Path("figures") / "system_architecture.svg",
        Path("data") / "visual_asset_manifest.json",
        Path("REPORT_RESULTS_DRAFT.md"),
        Path("RESULTS_SUMMARY.md"),
    )
    required_table_csv = (
        "simulation_parameters.csv",
        "main_results.csv",
        "los_nlos_results.csv",
        "robustness_results.csv",
        "ablation_results.csv",
        "trajectory_filtering_results.csv",
        "runtime_results.csv",
        "per_seed_results.csv",
    )
    checked: list[str] = []
    for relative in (
        *(Path(name) for name in required_csv),
        *(Path("figures") / name for name in required_figures),
        *(Path("screenshots") / name for name in required_screenshots),
        *(Path("latex") / name for name in required_latex),
        *(Path("tables") / name for name in required_table_csv),
        *required_misc,
    ):
        path = run_dir / relative
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Required output is missing or empty: {path}")
        checked.append(relative.as_posix())
    for name in required_csv:
        if pd.read_csv(run_dir / name).empty:
            raise RuntimeError(f"Required CSV contains no rows: {name}")
    return {"status": "passed", "checked_files": len(checked)}


def run_pipeline(
    *,
    profile: str = "quick",
    seed: int | None = None,
    output_dir: str | Path | None = None,
    config_path: str | Path | None = None,
    skip_training: bool = False,
    skip_robustness: bool = False,
    force: bool = False,
) -> Path:
    """Execute data generation, training, evaluation, and asset export.

    Raises:
        Any configuration, training, evaluation, or export error. The CLI
        therefore returns a non-zero code and never reports a partial run as
        successful.
    """

    root = _repo_root()
    run_path = (
        Path(output_dir)
        if output_dir is not None
        else _timestamped_run_dir(root)
    )
    run_dir = _prepare_run_dir(run_path, root=root, force=force)
    outputs_root = (root / "outputs").resolve()
    publish_repository_outputs = outputs_root in run_dir.parents
    _configure_logging(run_dir / "logs" / "pipeline.log")
    started_at = _utc_now()
    started = time.perf_counter()
    LOGGER.info("Starting %s pipeline in %s", profile, run_dir)

    try:
        config = load_config(
            profile=profile,
            scenario="normal",
            config_path=config_path,
        )
        selected_seed = int(
            seed
            if seed is not None
            else config.get("profile", {}).get("seeds", [42])[0]
        )
        configured_seeds = [
            int(value) for value in config.get("profile", {}).get("seeds", [selected_seed])
        ]
        if selected_seed not in configured_seeds:
            configured_seeds[0] = selected_seed

        (run_dir / "config_resolved.yaml").write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        _write_json(
            run_dir / "seeds.json",
            {"training_seed": selected_seed, "evaluation_seeds": configured_seeds},
        )

        splits, dataset_metadata = generate_dataset(
            config=config,
            profile=profile,
            seed=selected_seed,
            scenario="normal",
        )
        save_dataset(splits, dataset_metadata, run_dir / "data" / "splits")
        _write_json(run_dir / "environment.json", dataset_metadata["environment"])

        environment = Environment.from_config(config)
        if skip_training:
            models, training_times, model_sizes, training_metadata = _load_models(
                environment, root / "models", config
            )
            for model_path in (root / "models").glob("*"):
                if model_path.is_file():
                    shutil.copy2(model_path, run_dir / "models" / model_path.name)
            training_metadata["loaded_without_training"] = True
        else:
            (
                models,
                training_times,
                model_sizes,
                training_metadata,
            ) = _train_models(
                config,
                splits,
                environment,
                run_dir / "models",
                selected_seed,
            )
            _write_json(run_dir / "models" / "metadata.json", training_metadata)
            if publish_repository_outputs:
                _copy_models_to_repository(run_dir / "models", root / "models")
            else:
                LOGGER.info(
                    "Keeping models isolated because output_dir is outside %s",
                    outputs_root,
                )
        _write_json(run_dir / "training_log.json", training_metadata)

        metrics, per_sample, timings, scenario_frames = _main_evaluation(
            config,
            profile,
            splits,
            models,
            selected_seed,
            training_times,
            model_sizes,
            config_path,
        )
        per_seed = _seed_evaluation(
            config,
            profile,
            splits,
            models,
            selected_seed,
            configured_seeds,
            timings,
            training_times,
            model_sizes,
            environment,
            run_dir / "models",
            skip_training=skip_training,
        )
        robustness = _robustness_evaluation(
            config,
            splits["spatial_holdout"][["true_x", "true_y"]].to_numpy(dtype=float),
            models,
            configured_seeds,
            selected_seed,
            skip_robustness=skip_robustness,
        )
        ablation = _ablation_evaluation(
            config,
            splits["train"],
            splits["validation"],
            splits["spatial_holdout"],
            splits["trajectories"],
            models,
            selected_seed,
        )
        runtime = _runtime_table(
            timings, training_times, model_sizes, per_sample
        )
        runtime_lookup = runtime[
            [
                "algorithm",
                "inference_time_ms",
                "training_time_s",
                "model_size_mb",
            ]
        ]
        robustness["evaluation_split"] = "spatial_holdout_subset"
        robustness = robustness.merge(
            runtime_lookup,
            on="algorithm",
            how="left",
            validate="many_to_one",
        )
        ablation = ablation.merge(
            runtime_lookup,
            on="algorithm",
            how="left",
            validate="many_to_one",
        )

        metrics.to_csv(run_dir / "metrics.csv", index=False)
        per_sample.to_csv(run_dir / "per_sample_predictions.csv", index=False)
        per_seed.to_csv(run_dir / "per_seed_results.csv", index=False)
        robustness.to_csv(run_dir / "robustness_results.csv", index=False)
        ablation.to_csv(run_dir / "ablation_results.csv", index=False)
        runtime.to_csv(run_dir / "runtime_results.csv", index=False)
        _write_json(
            run_dir / "runtime.json",
            {
                "algorithms": runtime.to_dict(orient="records"),
                "timing_policy": (
                    "Median after warm-up; one-row latency and batch throughput "
                    "are separate; dashboard_update_ms is saved-frame data "
                    "preparation and excludes Streamlit/Plotly rendering."
                ),
            },
        )

        export_report_tables_and_text(run_dir, run_dir, config=config)
        from localization_twin.visualization.report_assets import (
            export_all_visual_assets,
        )

        export_all_visual_assets(run_dir, run_dir, strict=True)
        output_checks = _run_internal_output_checks(run_dir)

        finished_at = _utc_now()
        duration_s = time.perf_counter() - started
        manifest = build_manifest(
            run_dir,
            profile=profile,
            seed=selected_seed,
            config_path=str(config_path or f"config/{profile}.yaml"),
            data_counts={name: len(frame) for name, frame in splits.items()},
            algorithms=list(models),
            started_at=started_at,
            finished_at=finished_at,
            duration_s=duration_s,
            test_status={
                "pipeline_output_checks": output_checks,
                "pytest": "run separately; see EXECUTION_REPORT.md",
            },
            success=True,
        )
        manifest["published_to_repository_latest"] = publish_repository_outputs
        _write_json(run_dir / "manifest.json", manifest)

        if publish_repository_outputs:
            latest = root / "outputs" / "latest"
            _sync_directory(run_dir, latest, root)
            _sync_report_assets(run_dir, root / "report_assets")
            LOGGER.info("Latest outputs synchronized to %s", latest)
        else:
            LOGGER.info(
                "Skipping repository latest/report_assets publication for "
                "external output directory %s",
                run_dir,
            )
        LOGGER.info("Pipeline completed successfully in %.2f seconds", duration_s)
        return run_dir
    except Exception:
        LOGGER.exception("Pipeline failed; partial output remains at %s", run_dir)
        failure = {
            "success": False,
            "started_at": started_at,
            "failed_at": _utc_now(),
            "duration_s": time.perf_counter() - started,
        }
        _write_json(run_dir / "manifest.failed.json", failure)
        raise
