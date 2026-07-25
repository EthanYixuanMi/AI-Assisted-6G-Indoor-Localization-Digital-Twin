"""Two-dimensional constant-velocity Kalman filtering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np


class KalmanFilter2D:
    """Linear Kalman filter with state ``[x, y, vx, vy]``."""

    def __init__(
        self,
        *,
        dt: float = 1.0,
        process_noise: float = 0.1,
        measurement_noise: float | np.ndarray = 1.0,
        initial_covariance: float = 10.0,
    ) -> None:
        self.dt = float(dt)
        self.process_noise = float(process_noise)
        self.measurement_noise = measurement_noise
        self.initial_covariance = float(initial_covariance)
        if self.dt <= 0.0:
            raise ValueError("dt must be positive.")
        if self.process_noise < 0.0:
            raise ValueError("process_noise cannot be negative.")
        if self.initial_covariance <= 0.0:
            raise ValueError("initial_covariance must be positive.")
        self.H = np.asarray([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
        self.R = self._measurement_covariance(measurement_noise)
        self.state_: np.ndarray | None = None
        self.covariance_: np.ndarray | None = None

    @staticmethod
    def _measurement_covariance(value: float | np.ndarray) -> np.ndarray:
        array = np.asarray(value, dtype=float)
        if array.ndim == 0:
            if float(array) <= 0.0:
                raise ValueError("measurement_noise must be positive.")
            return np.eye(2) * float(array)
        if array.shape != (2, 2):
            raise ValueError("measurement_noise must be a scalar or 2x2 covariance.")
        if np.any(~np.isfinite(array)):
            raise ValueError("measurement covariance must be finite.")
        return array

    def _transition(self, dt: float) -> tuple[np.ndarray, np.ndarray]:
        transition = np.asarray(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        noise_map = np.asarray(
            [
                [0.25 * dt**4, 0.0, 0.5 * dt**3, 0.0],
                [0.0, 0.25 * dt**4, 0.0, 0.5 * dt**3],
                [0.5 * dt**3, 0.0, dt**2, 0.0],
                [0.0, 0.5 * dt**3, 0.0, dt**2],
            ]
        )
        return transition, noise_map * self.process_noise

    def reset(
        self,
        initial_position: np.ndarray | list[float] | tuple[float, float] | None = None,
        initial_velocity: np.ndarray | list[float] | tuple[float, float] = (0.0, 0.0),
    ) -> "KalmanFilter2D":
        """Reset filter state, optionally initializing a position."""

        if initial_position is None:
            self.state_ = None
            self.covariance_ = None
            return self
        position = np.asarray(initial_position, dtype=float)
        velocity = np.asarray(initial_velocity, dtype=float)
        if position.shape != (2,) or velocity.shape != (2,):
            raise ValueError("Initial position and velocity must be length-2.")
        if np.any(~np.isfinite(position)) or np.any(~np.isfinite(velocity)):
            raise ValueError("Initial state must be finite.")
        self.state_ = np.concatenate((position, velocity))
        self.covariance_ = np.eye(4) * self.initial_covariance
        return self

    def predict_state(self, *, dt: float | None = None) -> np.ndarray:
        """Advance the state without incorporating a measurement."""

        if self.state_ is None or self.covariance_ is None:
            raise RuntimeError("Filter has not been initialized with a measurement.")
        step = self.dt if dt is None else float(dt)
        if step <= 0.0:
            raise ValueError("dt must be positive.")
        transition, process_covariance = self._transition(step)
        self.state_ = transition @ self.state_
        self.covariance_ = (
            transition @ self.covariance_ @ transition.T + process_covariance
        )
        return self.state_.copy()

    def update(
        self,
        measurement: np.ndarray | list[float] | tuple[float, float],
        *,
        dt: float | None = None,
    ) -> np.ndarray:
        """Advance and update from one position measurement.

        A non-finite measurement triggers prediction-only behavior after
        initialization, which is useful for short localization dropouts.
        """

        observation = np.asarray(measurement, dtype=float)
        if observation.shape != (2,):
            raise ValueError("measurement must be a length-2 coordinate.")
        finite = np.all(np.isfinite(observation))
        if self.state_ is None:
            if not finite:
                raise ValueError("The first Kalman measurement must be finite.")
            self.reset(observation)
            return observation.copy()

        self.predict_state(dt=dt)
        if not finite:
            return self.state_[:2].copy()
        assert self.covariance_ is not None
        innovation = observation - self.H @ self.state_
        innovation_covariance = self.H @ self.covariance_ @ self.H.T + self.R
        gain = np.linalg.solve(
            innovation_covariance.T,
            (self.covariance_ @ self.H.T).T,
        ).T
        self.state_ = self.state_ + gain @ innovation
        identity = np.eye(4)
        # Joseph form protects symmetry/positive semidefiniteness numerically.
        correction = identity - gain @ self.H
        self.covariance_ = (
            correction @ self.covariance_ @ correction.T + gain @ self.R @ gain.T
        )
        return self.state_[:2].copy()

    def filter(
        self,
        positions: Any,
        *,
        dts: Any | None = None,
        reset: bool = True,
    ) -> np.ndarray:
        """Filter an ``(n_samples, 2)`` coordinate sequence."""

        observations = np.asarray(positions, dtype=float)
        if observations.ndim != 2 or observations.shape[1] != 2:
            raise ValueError("positions must have shape (n_samples, 2).")
        if reset:
            self.reset()
        if dts is None:
            steps = np.full(len(observations), self.dt, dtype=float)
        else:
            steps = np.asarray(dts, dtype=float)
            if steps.ndim == 0:
                steps = np.full(len(observations), float(steps), dtype=float)
            if steps.shape != (len(observations),):
                raise ValueError("dts must be scalar or length n_samples.")
        result = np.empty_like(observations, dtype=float)
        for index, (observation, step) in enumerate(zip(observations, steps)):
            result[index] = self.update(observation, dt=float(step))
        return result

    def predict(self, positions: Any, **kwargs: Any) -> np.ndarray:
        """Unified model-interface alias for :meth:`filter`."""

        return self.filter(positions, **kwargs)

    def save(self, path: str | Path) -> Path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, destination)
        return destination

    @classmethod
    def load(cls, path: str | Path) -> "KalmanFilter2D":
        loaded = joblib.load(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"{path!s} does not contain a {cls.__name__}.")
        return loaded


ConstantVelocityKalmanFilter = KalmanFilter2D
