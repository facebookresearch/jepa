#!/usr/bin/env python3
"""Write the effective per-split configuration used by the benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split_id", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--vjepa_config", type=Path, required=True)
    parser.add_argument("--vjepa_checkpoint", type=Path, required=True)
    parser.add_argument("--videomae_model", required=True)
    return parser.parse_args()


def main() -> None:
    """Persist actual classes, sample IDs, and model settings."""
    args = parse_args()
    metadata = pd.read_csv(args.metadata)
    config = {
        "experiment": {
            "name": "vjepa_vs_videomae_ucf101",
            "official_split_id": args.split_id,
            "seed": args.seed,
        },
        "dataset": {
            "classes": metadata.sort_values("label")["class_name"]
            .drop_duplicates()
            .tolist(),
            "num_train_samples": int(metadata["split"].eq("train").sum()),
            "num_test_samples": int(metadata["split"].eq("test").sum()),
            "train_per_class": metadata.loc[metadata["split"].eq("train")]
            .groupby("class_name")
            .size()
            .to_dict(),
            "test_per_class": metadata.loc[metadata["split"].eq("test")]
            .groupby("class_name")
            .size()
            .to_dict(),
            "video_ids": metadata["video_id"].astype(str).tolist(),
            "num_frames": args.num_frames,
            "image_size": args.image_size,
        },
        "models": {
            "vjepa": {
                "config": str(args.vjepa_config.expanduser().resolve()),
                "checkpoint": str(args.vjepa_checkpoint.expanduser().resolve()),
                "pooling": "mean_patch_tokens",
            },
            "videomae": {
                "model_name": args.videomae_model,
                "pooling": "mean_patch_tokens",
            },
        },
        "runtime": {"device_requested": args.device, "batch_size": 1},
        "linear_probe": {
            "pipeline": "StandardScaler + LogisticRegression",
            "max_iter": 2000,
            "random_state": 42,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


if __name__ == "__main__":
    main()
