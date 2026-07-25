"""High-level, traceable export entry points for report visual assets."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

from .architecture import export_system_architecture_svg
from .dashboard_snapshot import export_dashboard_snapshots
from .data import RESULT_FILES, ResultBundle, VisualDataError
from .plots import export_publication_figures
from .style import canonical_algorithm


class VisualAssetError(RuntimeError):
    """Public exporter error with an actionable, user-facing message."""


CORE_METHODS = {"Geometric LS", "KNN", "Direct AI", "Residual AI"}


def _wrap_error(action: str, exc: Exception) -> VisualAssetError:
    return VisualAssetError(f"{action} failed: {exc}")


def _validate_core_methods(bundle: ResultBundle) -> None:
    bundle.require("metrics", "predictions")
    metric_methods = set(bundle.metrics["algorithm"].map(canonical_algorithm))
    prediction_methods = set(
        bundle.predictions["algorithm"].map(canonical_algorithm)
    )
    missing_metrics = sorted(CORE_METHODS - metric_methods)
    missing_predictions = sorted(CORE_METHODS - prediction_methods)
    messages = []
    if missing_metrics:
        messages.append(
            "metrics.csv missing " + ", ".join(missing_metrics)
        )
    if missing_predictions:
        messages.append(
            "per_sample_predictions.csv missing "
            + ", ".join(missing_predictions)
        )
    if messages:
        raise VisualDataError(
            "Required localization methods were not silently omitted: "
            + "; ".join(messages)
            + ". Re-run the complete experiment evaluation."
        )


def _source_hashes(results_dir: Path) -> dict[str, dict[str, Any]]:
    names = [
        *RESULT_FILES.values(),
        "config_resolved.yaml",
        "environment.json",
    ]
    hashes: dict[str, dict[str, Any]] = {}
    for name in names:
        path = results_dir / name
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        hashes[name] = {
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        }
    return hashes


def _write_manifest(
    results_dir: Path,
    output_root: Path,
    paths: list[Path],
) -> Path:
    data_root = output_root / "data"
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / "visual_asset_manifest.json"
    generated: list[dict[str, Any]] = []
    for path in sorted(set(paths)):
        relative = path.relative_to(output_root).as_posix()
        generated.append({"path": relative, "bytes": path.stat().st_size})
    payload = {
        "artifact_type": "simulation_visual_assets",
        "generated_at_utc": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "source_directory": results_dir.name,
        "source_files": _source_hashes(results_dir),
        "figure_protocols": {
            "error_cdf": {
                "source_file": "per_sample_predictions.csv",
                "source_data_file": "data/error_cdf_source.csv",
                "filters": {
                    "scenario": "normal",
                    "split": "spatial_holdout",
                    "algorithms": [
                        "Geometric LS",
                        "KNN",
                        "Direct AI",
                        "Residual AI",
                    ],
                },
                "excluded_protocols": ["trajectories", "Kalman smoothing"],
            },
            "ablation_results": {
                "source_file": "ablation_results.csv",
                "source_data_file": "data/ablation_results_source.csv",
                "filters": {
                    "comparison": "feature ablations on spatial holdout",
                    "exclude_variant_substring_case_insensitive": "kalman",
                },
                "excluded_protocols": ["trajectory filtering"],
            },
        },
        "generated_files": generated,
        "no_fabricated_fallback_data": True,
    }
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def export_report_figures(
    results_dir: str | Path,
    report_assets_dir: str | Path,
    *,
    strict: bool = True,
) -> list[Path]:
    """Export system architecture and all required paper figures.

    Parameters
    ----------
    results_dir:
        Directory containing the saved experiment contract, usually
        ``outputs/latest``.
    report_assets_dir:
        Root output directory containing ``figures`` and ``data``.
    strict:
        Validate every contracted result table before rendering.  Disabling
        strict loading does not create fallback evidence; individual plots still
        fail if their source is unavailable.
    """

    try:
        bundle = ResultBundle.load(results_dir, strict=strict)
        _validate_core_methods(bundle)
        root = Path(report_assets_dir).expanduser().resolve()
        paths = export_publication_figures(
            bundle,
            root / "figures",
            root / "data",
        )
        paths.append(
            export_system_architecture_svg(
                root / "figures" / "system_architecture.svg"
            )
        )
        return paths
    except (VisualDataError, OSError, ValueError, RuntimeError) as exc:
        raise _wrap_error("Report figure export", exc) from exc


def export_dashboard_assets(
    results_dir: str | Path,
    report_assets_dir: str | Path,
) -> list[Path]:
    """Export the overview and five required browser-free screenshots."""

    try:
        bundle = ResultBundle.load(results_dir, strict=False)
        bundle.require("metrics", "predictions")
        if bundle.environment is None:
            raise VisualDataError(
                "Dashboard export requires a valid environment.json."
            )
        _validate_core_methods(bundle)
        root = Path(report_assets_dir).expanduser().resolve()
        return export_dashboard_snapshots(
            bundle,
            root / "figures",
            root / "screenshots",
            root / "data",
        )
    except (VisualDataError, OSError, ValueError, RuntimeError) as exc:
        raise _wrap_error("Dashboard snapshot export", exc) from exc


def export_all_visual_assets(
    results_dir: str | Path,
    report_assets_dir: str | Path,
    *,
    strict: bool = True,
) -> list[Path]:
    """Export all figures/screenshots and write a traceability manifest."""

    results_root = Path(results_dir).expanduser().resolve()
    output_root = Path(report_assets_dir).expanduser().resolve()
    figure_paths = export_report_figures(
        results_root, output_root, strict=strict
    )
    dashboard_paths = export_dashboard_assets(results_root, output_root)
    paths = figure_paths + dashboard_paths
    manifest = _write_manifest(results_root, output_root, paths)
    return paths + [manifest]
