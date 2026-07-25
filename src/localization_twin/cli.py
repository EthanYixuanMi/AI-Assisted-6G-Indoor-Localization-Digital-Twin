"""Command-line entry point for the localization digital twin."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Build the public command-line parser."""

    parser = argparse.ArgumentParser(
        prog="localization-twin",
        description="Run the AI-assisted indoor localization digital twin.",
    )
    parser.add_argument(
        "--profile",
        choices=("quick", "full"),
        default="quick",
        help="Experiment profile to execute.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-robustness", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Execute the pipeline and return a process exit code."""

    args = build_parser().parse_args(argv)
    from localization_twin.evaluation.runner import run_pipeline

    run_pipeline(
        profile=args.profile,
        seed=args.seed,
        output_dir=args.output_dir,
        config_path=args.config,
        skip_training=args.skip_training,
        skip_robustness=args.skip_robustness,
        force=args.force,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

