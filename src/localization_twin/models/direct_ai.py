"""Lightweight direct coordinate regression using scikit-learn MLPs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import coordinate_targets, extract_features, infer_anchor_ids


DEFAULT_MLP_CONFIGS: tuple[dict[str, Any], ...] = (
    {
        "hidden_layer_sizes": (48,),
        "alpha": 5e-4,
        "learning_rate_init": 2e-3,
    },
    {
        "hidden_layer_sizes": (64, 32),
        "alpha": 1e-3,
        "learning_rate_init": 1e-3,
    },
)


def _coerce_targets(frame: pd.DataFrame, supplied: Any | None) -> np.ndarray:
    targets = (
        coordinate_targets(frame)
        if supplied is None
        else np.asarray(supplied, dtype=float)
    )
    if targets.shape != (len(frame), 2):
        raise ValueError(f"Targets must have shape ({len(frame)}, 2), got {targets.shape}.")
    if np.any(~np.isfinite(targets)):
        raise ValueError("Targets contain NaN or infinite values.")
    return targets


class DirectAILocator:
    """Direct RSS-to-coordinate MLP with a small validation search."""

    def __init__(
        self,
        anchor_ids: Sequence[str] | None = None,
        *,
        include_los: bool = True,
        configs: Sequence[Mapping[str, Any]] | None = None,
        random_state: int = 42,
        max_iter: int = 220,
        missing_value: float = -110.0,
        validation_fraction: float = 0.15,
        n_iter_no_change: int = 15,
    ) -> None:
        self.anchor_ids = list(anchor_ids) if anchor_ids is not None else None
        self.include_los = bool(include_los)
        self.configs = tuple(
            dict(config) for config in (configs if configs is not None else DEFAULT_MLP_CONFIGS)
        )
        if not self.configs:
            raise ValueError("At least one MLP configuration is required.")
        self.random_state = int(random_state)
        self.max_iter = int(max_iter)
        self.missing_value = float(missing_value)
        self.validation_fraction = float(validation_fraction)
        self.n_iter_no_change = int(n_iter_no_change)
        self.preprocessor_: Pipeline | None = None
        self.model_: MLPRegressor | None = None
        self.best_config_: dict[str, Any] | None = None
        self.validation_scores_: list[dict[str, Any]] = []
        self.training_time_s_: float | None = None
        self.feature_names_: list[str] = []

    @property
    def feature_groups(self) -> tuple[str, ...]:
        return ("rss", "mask", "los") if self.include_los else ("rss", "mask")

    def _features(self, frame: pd.DataFrame) -> np.ndarray:
        if self.anchor_ids is None:
            self.anchor_ids = infer_anchor_ids(frame)
        matrix, names = extract_features(
            frame,
            self.anchor_ids,
            self.feature_groups,
            return_names=True,
        )
        if self.feature_names_ and self.feature_names_ != names:
            raise ValueError("Feature columns differ from the fitted model.")
        return matrix

    def _make_model(self, config: Mapping[str, Any]) -> MLPRegressor:
        parameters = dict(config)
        parameters.update(
            {
                "random_state": self.random_state,
                "max_iter": self.max_iter,
                "early_stopping": True,
                "validation_fraction": self.validation_fraction,
                "n_iter_no_change": self.n_iter_no_change,
                "batch_size": "auto",
                "solver": "adam",
            }
        )
        return MLPRegressor(**parameters)

    def fit(
        self,
        train_frame: pd.DataFrame,
        y: Any | None = None,
        *,
        validation_frame: pd.DataFrame | None = None,
        y_validation: Any | None = None,
    ) -> "DirectAILocator":
        """Fit the direct model; external validation is never used for updates."""

        if isinstance(y, pd.DataFrame) and validation_frame is None:
            validation_frame = y
            y = None
        if len(train_frame) < 8:
            raise ValueError("DirectAILocator needs at least eight training samples.")
        started = perf_counter()
        if self.anchor_ids is None:
            self.anchor_ids = infer_anchor_ids(train_frame)
        x_train, names = extract_features(
            train_frame,
            self.anchor_ids,
            self.feature_groups,
            return_names=True,
        )
        self.feature_names_ = names
        targets = _coerce_targets(train_frame, y)
        self.preprocessor_ = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=self.missing_value,
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformed_train = self.preprocessor_.fit_transform(x_train)

        transformed_validation: np.ndarray | None = None
        validation_targets: np.ndarray | None = None
        if validation_frame is not None and len(validation_frame):
            transformed_validation = self.preprocessor_.transform(
                self._features(validation_frame)
            )
            validation_targets = _coerce_targets(validation_frame, y_validation)

        self.validation_scores_ = []
        candidate_models: list[MLPRegressor] = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            for config in self.configs:
                candidate = self._make_model(config)
                candidate.fit(transformed_train, targets)
                candidate_models.append(candidate)
                if transformed_validation is not None and validation_targets is not None:
                    prediction = candidate.predict(transformed_validation)
                    score = float(
                        np.mean(np.linalg.norm(prediction - validation_targets, axis=1))
                    )
                else:
                    score = float(candidate.loss_)
                self.validation_scores_.append(
                    {"config": dict(config), "mean_error": score}
                )

        best_index = min(
            range(len(candidate_models)),
            key=lambda index: (self.validation_scores_[index]["mean_error"], index),
        )
        self.model_ = candidate_models[best_index]
        self.best_config_ = dict(self.configs[best_index])
        self.training_time_s_ = perf_counter() - started
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        if self.preprocessor_ is None or self.model_ is None:
            raise RuntimeError("DirectAILocator must be fit before prediction.")
        transformed = self.preprocessor_.transform(self._features(frame))
        predictions = np.asarray(self.model_.predict(transformed), dtype=float)
        if predictions.shape != (len(frame), 2):
            raise RuntimeError("Unexpected Direct AI prediction shape.")
        return predictions

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "DirectAILocator":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"{path!s} does not contain a {cls.__name__}.")
        return loaded


DirectAIRegressor = DirectAILocator
