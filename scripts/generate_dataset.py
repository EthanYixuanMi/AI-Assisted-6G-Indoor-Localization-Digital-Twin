"""Generate and persist only the configured simulated datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from localization_twin.config import load_config
from localization_twin.dataset import generate_dataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "generated")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    config = load_config(profile=args.profile, config_path=args.config)
    seed = int(
        args.seed
        if args.seed is not None
        else config.get("experiment", {}).get(
            "seed", config.get("profile", {}).get("seeds", [42])[0]
        )
    )
    splits, generated_metadata = generate_dataset(
        config=config, profile=args.profile, seed=seed
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, object] = {"profile": args.profile, "seed": seed, "splits": {}}
    for name, frame in splits.items():
        if hasattr(frame, "to_csv"):
            path = args.output_dir / f"{name}.csv"
            frame.to_csv(path, index=False)
            metadata["splits"][name] = {"path": str(path), "samples": len(frame)}
    metadata["generator"] = generated_metadata
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
