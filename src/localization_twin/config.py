"""Configuration loading and validation for the localization digital twin."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any, Mapping, MutableMapping

import yaml

LOGGER = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


class ConfigError(ValueError):
    """Raised when a configuration file is missing or internally inconsistent."""


def deep_merge(
    base: Mapping[str, Any], override: Mapping[str, Any]
) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and list values.

    The returned dictionary is independent of both inputs. Lists are replaced
    rather than concatenated so that a scenario can intentionally replace the
    complete anchor or wall definition.
    """

    merged: dict[str, Any] = deepcopy(dict(base))
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], Mapping)
            and isinstance(value, Mapping)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Read a YAML mapping from *path* with actionable validation errors."""

    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise ConfigError(f"Configuration file does not exist: {resolved}")
    try:
        with resolved.open("r", encoding="utf-8") as handle:
            document = yaml.safe_load(handle)
    except yaml.YAMLError as exc:
        raise ConfigError(f"Invalid YAML in {resolved}: {exc}") from exc
    if document is None:
        return {}
    if not isinstance(document, dict):
        raise ConfigError(f"Top-level YAML value must be a mapping: {resolved}")
    return document


def _resolve_profile_path(profile: str | Path) -> Path:
    candidate = Path(profile)
    if candidate.suffix.lower() in {".yaml", ".yml"} or candidate.parent != Path("."):
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return CONFIG_DIR / f"{candidate}.yaml"


def _resolve_scenario_path(scenario: str | Path) -> Path:
    candidate = Path(scenario)
    if candidate.suffix.lower() in {".yaml", ".yml"} or candidate.parent != Path("."):
        return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate
    return CONFIG_DIR / "scenarios" / f"{candidate}.yaml"


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate fields required by the core simulator.

    This intentionally validates structural invariants and ranges, while
    allowing model/evaluation sections to be extended by other modules.
    """

    environment = config.get("environment")
    propagation = config.get("propagation")
    sampling = config.get("sampling")
    if not isinstance(environment, Mapping):
        raise ConfigError("Configuration requires an 'environment' mapping.")
    if not isinstance(propagation, Mapping):
        raise ConfigError("Configuration requires a 'propagation' mapping.")
    if not isinstance(sampling, Mapping):
        raise ConfigError("Configuration requires a 'sampling' mapping.")

    width = float(environment.get("width", 0.0))
    height = float(environment.get("height", 0.0))
    if width <= 0.0 or height <= 0.0:
        raise ConfigError("Environment width and height must be positive.")

    anchors = environment.get("anchors", [])
    if not isinstance(anchors, list) or len(anchors) < 3:
        raise ConfigError("At least three anchor definitions are required.")
    anchor_ids = [str(item.get("anchor_id", "")) for item in anchors]
    if any(not anchor_id for anchor_id in anchor_ids):
        raise ConfigError("Every anchor requires a non-empty anchor_id.")
    if len(set(anchor_ids)) != len(anchor_ids):
        raise ConfigError("Anchor IDs must be unique.")
    for anchor in anchors:
        x = float(anchor.get("x", -1.0))
        y = float(anchor.get("y", -1.0))
        if not 0.0 <= x <= width or not 0.0 <= y <= height:
            raise ConfigError(
                f"Anchor {anchor['anchor_id']} lies outside the environment."
            )
        if float(anchor.get("path_loss_exponent", 2.0)) <= 0.0:
            raise ConfigError("Anchor path-loss exponents must be positive.")

    for key in (
        "train_count",
        "validation_count",
        "test_count",
        "spatial_holdout_count",
        "domain_shift_count",
        "anchor_failure_count",
    ):
        if int(sampling.get(key, 0)) < 0:
            raise ConfigError(f"sampling.{key} cannot be negative.")

    if float(propagation.get("noise_std", 0.0)) < 0.0:
        raise ConfigError("propagation.noise_std cannot be negative.")
    dropout = float(propagation.get("dropout_probability", 0.0))
    if not 0.0 <= dropout <= 1.0:
        raise ConfigError("propagation.dropout_probability must be in [0, 1].")


def load_config(
    profile: str | Path = "quick",
    scenario: str | Path | None = "normal",
    config_path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load the default, profile, scenario, and caller overrides.

    Merge precedence is ``default < profile < scenario < config_path <
    overrides``. Passing a YAML path as *profile* is supported. The optional
    *config_path* is useful for a complete experiment-specific overlay.
    """

    default_path = CONFIG_DIR / "default.yaml"
    config = load_yaml(default_path)
    source_paths = [default_path.resolve()]

    profile_path = _resolve_profile_path(profile)
    if profile_path.resolve() != default_path.resolve():
        config = deep_merge(config, load_yaml(profile_path))
        source_paths.append(profile_path.resolve())

    if scenario is not None:
        scenario_path = _resolve_scenario_path(scenario)
        config = deep_merge(config, load_yaml(scenario_path))
        source_paths.append(scenario_path.resolve())

    if config_path is not None:
        extra_path = Path(config_path)
        if not extra_path.is_absolute():
            extra_path = PROJECT_ROOT / extra_path
        config = deep_merge(config, load_yaml(extra_path))
        source_paths.append(extra_path.resolve())

    if overrides:
        config = deep_merge(config, overrides)

    config.setdefault("_meta", {})
    config["_meta"]["source_files"] = [str(path) for path in source_paths]
    config["_meta"]["project_root"] = str(PROJECT_ROOT)
    validate_config(config)
    LOGGER.info(
        "Loaded profile=%s scenario=%s from %d YAML files",
        config.get("profile", {}).get("name", profile),
        config.get("scenario", {}).get("name", scenario),
        len(source_paths),
    )
    return config


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> Path:
    """Write a resolved configuration to YAML and return its absolute path."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = deepcopy(dict(config))
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
    return destination


def set_nested_value(
    config: MutableMapping[str, Any], dotted_key: str, value: Any
) -> None:
    """Set ``a.b.c`` inside a mutable configuration mapping."""

    keys = dotted_key.split(".")
    if not keys or any(not key for key in keys):
        raise ConfigError(f"Invalid dotted configuration key: {dotted_key!r}")
    cursor: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        child = cursor.setdefault(key, {})
        if not isinstance(child, MutableMapping):
            raise ConfigError(
                f"Cannot set {dotted_key!r}; {key!r} is not a mapping."
            )
        cursor = child
    cursor[keys[-1]] = value


resolve_config = load_config

