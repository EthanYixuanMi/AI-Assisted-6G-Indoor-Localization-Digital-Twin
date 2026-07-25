"""RSS fingerprinting with validation-only neighbour selection."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import coordinate_targets, extract_rss_mask_features, infer_anchor_ids


def _targets(frame: pd.DataFrame, supplied: Any | None) -> np.ndarray:
    values = coordinate_targets(frame) if supplied is None else np.asarray(supplied, dtype=float)
    if values.shape != (len(frame), 2):
        raise ValueError(f"Targets must have shape ({len(frame)}, 2), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("Targets contain NaN or infinite values.")
    return values


class KNNFingerprintLocator:
    """Distance-weighted KNN fingerprint locator.

    The imputer and scaler are fit on the training split only.  Candidate
    neighbour counts are compared solely on the supplied validation split.
    """

    def __init__(
        self,
        anchor_ids: Sequence[str] | None = None,
        *,
        k_candidates: Sequence[int] = (3, 5, 7, 9),
        weights: str = "distance",
        missing_rss: float = -110.0,
        metric: str = "minkowski",
        p: int = 2,
        n_jobs: int = 1,
    ) -> None:
        candidates = sorted({int(value) for value in k_candidates if int(value) > 0})
        if not candidates:
            raise ValueError("k_candidates must contain at least one positive integer.")
        if weights not in {"uniform", "distance"}:
            raise ValueError("weights must be 'uniform' or 'distance'.")
        self.anchor_ids = list(anchor_ids) if anchor_ids is not None else None
        self.k_candidates = tuple(candidates)
        self.weights = weights
        self.missing_rss = float(missing_rss)
        self.metric = metric
        self.p = int(p)
        self.n_jobs = int(n_jobs)
        self.preprocessor_: Pipeline | None = None
        self.model_: KNeighborsRegressor | None = None
        self.best_k_: int | None = None
        self.validation_scores_: dict[int, float] = {}
        self.training_time_s_: float | None = None
        self.feature_names_: list[str] = []

    @property
    def best_k(self) -> int | None:
        """The selected number of neighbours after fitting."""

        return self.best_k_

    def _features(self, frame: pd.DataFrame) -> np.ndarray:
        if self.anchor_ids is None:
            self.anchor_ids = infer_anchor_ids(frame)
        matrix, names = extract_rss_mask_features(
            frame, self.anchor_ids, return_names=True
        )
        if self.feature_names_ and names != self.feature_names_:
            raise ValueError("Feature columns differ from the fitted model.")
        return matrix

    def fit(
        self,
        train_frame: pd.DataFrame,
        y: Any | None = None,
        *,
        validation_frame: pd.DataFrame | None = None,
        y_validation: Any | None = None,
    ) -> "KNNFingerprintLocator":
        """Fit fingerprints and select ``k`` using a validation frame.

        For convenience, ``fit(train, validation_frame)`` is accepted when the
        second positional data frame contains ``true_x`` and ``true_y``.
        """

        if isinstance(y, pd.DataFrame) and validation_frame is None:
            validation_frame = y
            y = None
        started = perf_counter()
        x_train, names = extract_rss_mask_features(
            train_frame,
            self.anchor_ids,
            return_names=True,
        )
        if self.anchor_ids is None:
            self.anchor_ids = infer_anchor_ids(train_frame)
        self.feature_names_ = names
        target_train = _targets(train_frame, y)
        if len(train_frame) == 0:
            raise ValueError("Cannot fit KNN on an empty training frame.")

        self.preprocessor_ = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant",
                        fill_value=self.missing_rss,
                        keep_empty_features=True,
                    ),
                ),
                ("scaler", StandardScaler()),
            ]
        )
        transformed_train = self.preprocessor_.fit_transform(x_train)
        valid_candidates = [
            candidate for candidate in self.k_candidates if candidate <= len(train_frame)
        ]
        if not valid_candidates:
            valid_candidates = [len(train_frame)]

        self.validation_scores_ = {}
        if validation_frame is not None and len(validation_frame):
            x_validation = self._features(validation_frame)
            transformed_validation = self.preprocessor_.transform(x_validation)
            target_validation = _targets(validation_frame, y_validation)
            for candidate in valid_candidates:
                candidate_model = KNeighborsRegressor(
                    n_neighbors=candidate,
                    weights=self.weights,
                    metric=self.metric,
                    p=self.p,
                    n_jobs=self.n_jobs,
                )
                candidate_model.fit(transformed_train, target_train)
                prediction = candidate_model.predict(transformed_validation)
                errors = np.linalg.norm(prediction - target_validation, axis=1)
                self.validation_scores_[candidate] = float(np.mean(errors))
            self.best_k_ = min(
                self.validation_scores_,
                key=lambda candidate: (self.validation_scores_[candidate], candidate),
            )
        else:
            self.best_k_ = valid_candidates[0]

        self.model_ = KNeighborsRegressor(
            n_neighbors=self.best_k_,
            weights=self.weights,
            metric=self.metric,
            p=self.p,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(transformed_train, target_train)
        self.training_time_s_ = perf_counter() - started
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict ``(x, y)`` coordinates for a data frame."""

        if self.model_ is None or self.preprocessor_ is None:
            raise RuntimeError("KNNFingerprintLocator must be fit before prediction.")
        transformed = self.preprocessor_.transform(self._features(frame))
        prediction = np.asarray(self.model_.predict(transformed), dtype=float)
        if prediction.shape != (len(frame), 2):
            raise RuntimeError("Unexpected KNN prediction shape.")
        return prediction

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "KNNFingerprintLocator":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"{path!s} does not contain a {cls.__name__}.")
        return loaded


KNNFingerprinting = KNNFingerprintLocator
