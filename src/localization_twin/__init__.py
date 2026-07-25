"""AI-assisted indoor localization digital-twin simulation package."""

from .config import ConfigError, load_config, resolve_config
from .dataset import generate_dataset, save_dataset
from .environment import (
    Anchor,
    Environment,
    PropagationRegion,
    RectangleObstacle,
    Wall,
)
from .features import build_feature_matrix, infer_anchor_ids, target_matrix
from .propagation import PropagationModel, SpatialBiasField
from .trajectory import generate_trajectories, sample_static_positions

__all__ = [
    "Anchor",
    "ConfigError",
    "Environment",
    "PropagationModel",
    "PropagationRegion",
    "RectangleObstacle",
    "SpatialBiasField",
    "Wall",
    "build_feature_matrix",
    "generate_dataset",
    "generate_trajectories",
    "infer_anchor_ids",
    "load_config",
    "resolve_config",
    "sample_static_positions",
    "save_dataset",
    "target_matrix",
]

__version__ = "0.1.0"
