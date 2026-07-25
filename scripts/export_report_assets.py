"""Export publication figures from a completed experiment run."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from localization_twin.visualization.report_assets import (  # noqa: E402
    VisualAssetError,
    export_all_visual_assets,
    export_report_figures,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate traceable PDF/PNG report figures from saved simulation "
            "results. Missing or empty result tables are treated as errors."
        )
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "outputs" / "latest",
        help="Saved result directory (default: outputs/latest).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "report_assets",
        help="Report asset root (default: report_assets).",
    )
    parser.add_argument(
        "--include-dashboard",
        action="store_true",
        help="Also generate dashboard overview and all five scenario screenshots.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    try:
        if args.include_dashboard:
            paths = export_all_visual_assets(
                args.results_dir, args.output_dir, strict=True
            )
        else:
            paths = export_report_figures(
                args.results_dir, args.output_dir, strict=True
            )
    except VisualAssetError as exc:
        logging.error("%s", exc)
        return 1
    logging.info(
        "Exported %d visual asset(s) to %s",
        len(paths),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
