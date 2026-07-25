"""Small end-to-end check for the public pipeline command."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_quick_pipeline_smoke(tmp_path: Path) -> None:
    output_dir = tmp_path / "pipeline-output"
    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_pipeline.py"),
        "--profile",
        "quick",
        "--config",
        str(ROOT / "tests" / "fixtures" / "smoke.yaml"),
        "--output-dir",
        str(output_dir),
        "--skip-robustness",
        "--force",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + "\n" + completed.stderr
    assert (output_dir / "manifest.json").is_file()
    assert (output_dir / "figures" / "error_cdf.pdf").stat().st_size > 0
    assert (output_dir / "screenshots" / "dashboard_normal.png").stat().st_size > 0
    assert (output_dir / "latex" / "table_main_results.tex").stat().st_size > 0
    metrics = pd.read_csv(output_dir / "metrics.csv")
    assert set(("Geometric LS", "KNN", "Direct AI", "Residual AI")).issubset(
        set(metrics["algorithm"])
    )

