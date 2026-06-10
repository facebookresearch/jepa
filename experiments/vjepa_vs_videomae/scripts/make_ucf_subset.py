#!/usr/bin/env python3
"""Create a balanced subset from an official UCF101 train/test split."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vjepa_vs_videomae.src.dataset_index import (
    select_official_subset,
)

DEFAULT_CLASSES = [
    "ApplyEyeMakeup",
    "Basketball",
    "Biking",
    "Diving",
    "WalkingWithDog",
]

LOGGER = logging.getLogger("make_ucf_subset")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ucf_root", type=Path, required=True)
    parser.add_argument("--official_splits_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--classes", nargs="+", default=DEFAULT_CLASSES)
    parser.add_argument("--split_id", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--train_per_class", type=int, default=60)
    parser.add_argument("--test_per_class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    """Select balanced samples without changing official split membership."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    ucf_root = args.ucf_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    if not ucf_root.is_dir():
        raise FileNotFoundError(f"UCF101 root does not exist: {ucf_root}")
    metadata = select_official_subset(
        ucf_root=ucf_root,
        splits_dir=args.official_splits_dir,
        classes=args.classes,
        split_id=args.split_id,
        train_per_class=args.train_per_class,
        test_per_class=args.test_per_class,
        seed=args.seed,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / "metadata.csv"
    metadata.to_csv(output_path, index=False)
    LOGGER.info(
        "Saved official split %02d: %d videos across %d classes to %s",
        args.split_id,
        len(metadata),
        metadata["class_name"].nunique(),
        output_path,
    )


if __name__ == "__main__":
    main()
