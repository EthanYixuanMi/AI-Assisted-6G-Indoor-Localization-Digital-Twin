"""AI correction of the geometric localization residual."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer

from .features import (
    FULL_RESIDUAL_FEATURE_GROUPS,
    coordinate_targets,
    extract_residual_features,
    infer_anchor_ids,
)


class ResidualAILocator:
    """Predict ``true_position - geometric_position`` with tree ensembles."""

    def __init__(
        self,
        anchor_ids: Sequence[str] | None = None,
        *,
        estimator: str = "extra_trees",
        feature_groups: Iterable[str] = FULL_RESIDUAL_FEATURE_GROUPS,
        n_estimators: int = 120,
        max_depth: int | None = 18,
        min_samples_leaf: int = 2,
        max_features: float | str | None = 1.0,
        random_state: int = 42,
        n_jobs: int = 1,
    ) -> None:
        estimator = estimator.lower()
        if estimator not in {"extra_trees", "random_forest", "auto"}:
            raise ValueError(
                "estimator must be 'extra_trees', 'random_forest', or 'auto'."
            )
        self.anchor_ids = list(anchor_ids) if anchor_ids is not None else None
        self.estimator = estimator
        self.feature_groups = tuple(dict.fromkeys(feature_groups))
        if not self.feature_groups:
            raise ValueError("feature_groups cannot be empty.")
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.max_features = max_features
        self.random_state = int(random_state)
        self.n_jobs = int(n_jobs)
        self.imputer_: SimpleImputer | None = None
        self.model_: ExtraTreesRegressor | RandomForestRegressor | None = None
        self.selected_estimator_: str | None = None
        self.validation_scores_: dict[str, float] = {}
        self.feature_names_: list[str] = []
        self.training_time_s_: float | None = None

    @staticmethod
    def _coordinates(
        frame: pd.DataFrame,
        geometric: Any | None,
    ) -> np.ndarray:
        source = frame if geometric is None else geometric
        if isinstance(source, pd.DataFrame):
            for pair in (
                ("geometric_x", "geometric_y"),
                ("pred_x", "pred_y"),
                ("estimated_x", "estimated_y"),
            ):
                if pair[0] in source and pair[1] in source:
                    values = source.loc[:, list(pair)].to_numpy(dtype=float)
                    break
            else:
                raise ValueError(
                    "Geometric data frame needs geometric_x/geometric_y or pred_x/pred_y."
                )
        else:
            values = np.asarray(source, dtype=float)
        if values.shape != (len(frame), 2):
            raise ValueError(
                f"Geometric predictions must have shape ({len(frame)}, 2), "
                f"got {values.shape}."
            )
        if np.any(~np.isfinite(values)):
            raise ValueError("Geometric predictions contain NaN or infinite values.")
        return values

    def _features(
        self,
        frame: pd.DataFrame,
        geometric: Any | None,
    ) -> np.ndarray:
        if self.anchor_ids is None:
            self.anchor_ids = infer_anchor_ids(frame)
        matrix, names = extract_residual_features(
            frame,
            self.anchor_ids,
            geometric=geometric,
            feature_groups=self.feature_groups,
            return_names=True,
        )
        if self.feature_names_ and names != self.feature_names_:
            raise ValueError("Feature columns differ from the fitted residual model.")
        return matrix

    def _make_model(
        self, estimator_name: str
    ) -> ExtraTreesRegressor | RandomForestRegressor:
        common: dict[str, Any] = {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "random_state": self.random_state,
            "n_jobs": self.n_jobs,
        }
        if estimator_name == "extra_trees":
            return ExtraTreesRegressor(**common)
        return RandomForestRegressor(**common)

    def fit(
        self,
        train_frame: pd.DataFrame,
        geometric: Any | None = None,
        y: Any | None = None,
        *,
        validation_frame: pd.DataFrame | None = None,
        validation_geometric: Any | None = None,
        y_validation: Any | None = None,
    ) -> "ResidualAILocator":
        """Fit residual correction, optionally selecting the ensemble family."""

        if len(train_frame) == 0:
            raise ValueError("Cannot fit residual correction on an empty frame.")
        started = perf_counter()
        if self.anchor_ids is None:
            self.anchor_ids = infer_anchor_ids(train_frame)
        x_train, names = extract_residual_features(
            train_frame,
            self.anchor_ids,
            geometric=geometric,
            feature_groups=self.feature_groups,
            return_names=True,
        )
        self.feature_names_ = names
        geometric_train = self._coordinates(train_frame, geometric)
        true_train = (
            coordinate_targets(train_frame)
            if y is None
            else np.asarray(y, dtype=float)
        )
        if true_train.shape != geometric_train.shape:
            raise ValueError("Residual targets must have shape (n_samples, 2).")
        residual_targets = true_train - geometric_train

        self.imputer_ = SimpleImputer(
            strategy="median",
            keep_empty_features=True,
        )
        transformed_train = self.imputer_.fit_transform(x_train)
        estimator_names = (
            ("extra_trees", "random_forest")
            if self.estimator == "auto"
            else (self.estimator,)
        )
        self.validation_scores_ = {}
        candidates: dict[
            str, ExtraTreesRegressor | RandomForestRegressor
        ] = {}
        for estimator_name in estimator_names:
            candidate = self._make_model(estimator_name)
            candidate.fit(transformed_train, residual_targets)
            candidates[estimator_name] = candidate
            if validation_frame is not None and len(validation_frame):
                validation_features = self._features(
                    validation_frame, validation_geometric
                )
                validation_features = self.imputer_.transform(validation_features)
                geometric_validation_values = self._coordinates(
                    validation_frame, validation_geometric
                )
                validation_true = (
                    coordinate_targets(validation_frame)
                    if y_validation is None
                    else np.asarray(y_validation, dtype=float)
                )
                corrected = (
                    geometric_validation_values
                    + candidate.predict(validation_features)
                )
                self.validation_scores_[estimator_name] = float(
                    np.mean(np.linalg.norm(corrected - validation_true, axis=1))
                )

        if self.validation_scores_:
            selected = min(
                estimator_names,
                key=lambda name: (self.validation_scores_[name], name),
            )
        else:
            selected = estimator_names[0]
        self.selected_estimator_ = selected
        self.model_ = candidates[selected]
        self.training_time_s_ = perf_counter() - started
        return self

    def predict_correction(
        self,
        frame: pd.DataFrame,
        geometric: Any | None = None,
    ) -> np.ndarray:
        """Predict the learned ``(dx, dy)`` correction only."""

        if self.model_ is None or self.imputer_ is None:
            raise RuntimeError("ResidualAILocator must be fit before prediction.")
        features = self.imputer_.transform(self._features(frame, geometric))
        correction = np.asarray(self.model_.predict(features), dtype=float)
        if correction.shape != (len(frame), 2):
            raise RuntimeError("Unexpected residual prediction shape.")
        return correction

    def predict(
        self,
        frame: pd.DataFrame,
        geometric: Any | None = None,
    ) -> np.ndarray:
        """Return geometric coordinates plus the learned correction."""

        baseline = self._coordinates(frame, geometric)
        return baseline + self.predict_correction(frame, geometric)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("ResidualAILocator must be fit first.")
        return np.asarray(self.model_.feature_importances_, dtype=float)

    def feature_importance_table(self) -> pd.DataFrame:
        """Return descending tree feature importances for diagnostics."""

        return (
            pd.DataFrame(
                {
                    "feature": self.feature_names_,
                    "importance": self.feature_importances_,
                }
            )
            .sort_values("importance", ascending=False, ignore_index=True)
        )

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "ResidualAILocator":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"{path!s} does not contain a {cls.__name__}.")
        return loaded


ResidualAIRegressor = ResidualAILocator
