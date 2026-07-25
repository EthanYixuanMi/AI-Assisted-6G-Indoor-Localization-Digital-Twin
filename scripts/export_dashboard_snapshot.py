"""Create browser-free dashboard screenshots from saved experiment results."""

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
    export_dashboard_assets,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a static dashboard overview and five scenario screenshots "
            "without starting a browser."
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s: %(message)s"
    )
    try:
        paths = export_dashboard_assets(args.results_dir, args.output_dir)
    except VisualAssetError as exc:
        logging.error("%s", exc)
        return 1
    logging.info(
        "Exported %d static dashboard image(s) to %s",
        len(paths),
        args.output_dir.resolve(),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
