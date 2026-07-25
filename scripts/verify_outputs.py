"""Validate the latest experiment package without modifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from PIL import Image


REQUIRED_CSV = (
    "metrics.csv",
    "per_sample_predictions.csv",
    "per_seed_results.csv",
    "robustness_results.csv",
    "ablation_results.csv",
    "runtime_results.csv",
)
REQUIRED_FIGURES = (
    "trajectory_comparison",
    "error_cdf",
    "spatial_error_heatmaps",
    "robustness_noise",
    "robustness_anchor_failure",
    "robustness_results",
    "los_nlos_comparison",
    "ablation_results",
    "runtime_comparison",
)
REQUIRED_SCREENSHOTS = (
    "dashboard_normal.png",
    "dashboard_high_noise.png",
    "dashboard_strong_blockage.png",
    "dashboard_anchor_failure.png",
    "dashboard_domain_shift.png",
)
REQUIRED_LATEX = (
    "table_simulation_parameters.tex",
    "table_main_results.tex",
    "table_los_nlos.tex",
    "table_robustness.tex",
    "table_ablation.tex",
    "table_runtime.tex",
    "preliminary_results_snippet.tex",
)


def verify(results_dir: Path) -> dict[str, int | bool]:
    results_dir = results_dir.resolve()
    checked = 0
    for name in REQUIRED_CSV:
        path = results_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Missing or empty CSV: {path}")
        frame = pd.read_csv(path)
        if frame.empty:
            raise RuntimeError(f"CSV has no rows: {path}")
        checked += 1

    for stem in REQUIRED_FIGURES:
        for suffix in (".pdf", ".png"):
            path = results_dir / "figures" / f"{stem}{suffix}"
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Missing or empty figure: {path}")
            if suffix == ".pdf" and not path.read_bytes().startswith(b"%PDF"):
                raise RuntimeError(f"Invalid PDF header: {path}")
            if suffix == ".png":
                with Image.open(path) as image:
                    image.verify()
            checked += 1

    for name in REQUIRED_SCREENSHOTS:
        path = results_dir / "screenshots" / name
        with Image.open(path) as image:
            image.verify()
        checked += 1

    for name in REQUIRED_LATEX:
        path = results_dir / "latex" / name
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            raise RuntimeError(f"Empty LaTeX asset: {path}")
        if name.startswith("table_") and "\\toprule" not in text:
            raise RuntimeError(f"Table does not use booktabs: {path}")
        checked += 1

    manifest_path = results_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("success") is not True:
        raise RuntimeError("manifest.json does not mark the run successful")
    checked += 1
    return {"success": True, "checked_artifacts": checked}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "outputs" / "latest",
    )
    args = parser.parse_args()
    result = verify(args.results_dir)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

