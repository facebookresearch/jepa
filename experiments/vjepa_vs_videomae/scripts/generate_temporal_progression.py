#!/usr/bin/env python3
"""Measure prediction confidence as progressively more video frames are revealed."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import load
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_DIR = Path(__file__).resolve().parent
FRACTIONS = [0.25, 0.5, 0.75, 1.0]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_metadata", type=Path, required=True)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--work_dir", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--vjepa_config", type=Path, required=True)
    parser.add_argument("--vjepa_checkpoint", type=Path, required=True)
    parser.add_argument("--videomae_model", default="MCG-NJU/videomae-base")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--examples", type=int, default=3)
    return parser.parse_args()


def _create_progressive_metadata(
    source: pd.DataFrame, work_dir: Path, examples: int
) -> pd.DataFrame:
    test = source.loc[source["split"].eq("test")].copy()
    selected = (
        test.sort_values(["class_name", "video_id"])
        .groupby("class_name", as_index=False)
        .first()
        .head(examples)
    )
    rows: list[dict[str, object]] = []
    for source_row in selected.itertuples(index=False):
        frame_paths = sorted(Path(source_row.frames_dir).glob("frame_*.jpg"))
        if len(frame_paths) != 16:
            raise ValueError(f"Expected 16 frames in {source_row.frames_dir}.")
        for fraction in FRACTIONS:
            available = max(1, int(round(len(frame_paths) * fraction)))
            prefix = frame_paths[:available]
            indices = np.linspace(0, len(prefix) - 1, 16).round().astype(int)
            video_id = f"{source_row.video_id}__p{int(fraction * 100):03d}"
            frames_dir = work_dir / "frames" / video_id
            frames_dir.mkdir(parents=True, exist_ok=True)
            for output_index, source_index in enumerate(indices):
                with Image.open(prefix[source_index]) as image:
                    image.convert("RGB").save(
                        frames_dir / f"frame_{output_index:03d}.jpg", quality=95
                    )
            rows.append(
                {
                    "video_id": video_id,
                    "source_video_id": source_row.video_id,
                    "video_path": source_row.video_path,
                    "frames_dir": str(frames_dir),
                    "class_name": source_row.class_name,
                    "label": int(source_row.label),
                    "split": "test",
                    "num_frames": 16,
                    "image_size": 224,
                    "fraction": fraction,
                }
            )

    reference = dict(rows[-1])
    reference["video_id"] = f"{reference['video_id']}__reference"
    reference["split"] = "train"
    rows.insert(0, reference)
    metadata = pd.DataFrame(rows)
    metadata.to_csv(work_dir / "progressive_frames_metadata.csv", index=False)
    return metadata


def _run_extractors(args: argparse.Namespace, metadata_path: Path) -> None:
    commands = {
        "videomae": [
            sys.executable,
            str(SCRIPT_DIR / "extract_videomae_features.py"),
            "--frames_metadata",
            str(metadata_path),
            "--out_dir",
            str(args.work_dir / "features" / "videomae"),
            "--cache_dir",
            str(args.work_dir / "cache" / "videomae"),
            "--model_name",
            args.videomae_model,
            "--device",
            args.device,
            "--batch_size",
            "1",
        ],
        "vjepa": [
            sys.executable,
            str(SCRIPT_DIR / "extract_vjepa_features.py"),
            "--frames_metadata",
            str(metadata_path),
            "--out_dir",
            str(args.work_dir / "features" / "vjepa"),
            "--cache_dir",
            str(args.work_dir / "cache" / "vjepa"),
            "--config",
            str(args.vjepa_config),
            "--checkpoint",
            str(args.vjepa_checkpoint),
            "--device",
            args.device,
            "--batch_size",
            "1",
            "--pooling",
            "mean",
        ],
    }
    for command in commands.values():
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _predict(args: argparse.Namespace) -> pd.DataFrame:
    all_rows: list[pd.DataFrame] = []
    for model in ["vjepa", "videomae"]:
        features_dir = args.work_dir / "features" / model
        features = np.load(features_dir / "X_test.npy")
        index = pd.read_csv(features_dir / "index_test.csv")
        bundle = load(args.results_root / "split_01" / model / "linear_probe.joblib")
        scaled = bundle["scaler"].transform(features)
        classifier = bundle["classifier"]
        probabilities = classifier.predict_proba(scaled)
        predictions = classifier.predict(scaled)
        classes = np.asarray(classifier.classes_)
        true_columns = {
            int(label): int(np.where(classes == label)[0][0]) for label in classes
        }
        index["model_name"] = model
        index["fraction"] = (
            index["video_id"].str.extract(r"__p(\d{3})")[0].astype(int) / 100
        )
        index["source_video_id"] = index["video_id"].str.replace(
            r"__p\d{3}$", "", regex=True
        )
        index["pred_label"] = predictions.astype(int)
        index["pred_class"] = [
            bundle["label_to_class"][int(label)] for label in predictions
        ]
        index["true_class_confidence"] = [
            float(probabilities[position, true_columns[int(label)]])
            for position, label in enumerate(index["label"])
        ]
        index["confidence_vector"] = [
            json.dumps([float(value) for value in row]) for row in probabilities
        ]
        all_rows.append(index)
    return pd.concat(all_rows, ignore_index=True)


def _plot(frame: pd.DataFrame, out_dir: Path) -> None:
    videos = frame["source_video_id"].drop_duplicates().tolist()
    fig, axes = plt.subplots(
        len(videos), 1, figsize=(9, 3.2 * len(videos)), sharex=True
    )
    if len(videos) == 1:
        axes = [axes]
    for ax, video_id in zip(axes, videos):
        subset = frame.loc[frame["source_video_id"].eq(video_id)]
        true_class = str(subset["class_name"].iloc[0])
        for model, color in [("vjepa", "#2563eb"), ("videomae", "#f97316")]:
            model_rows = subset.loc[subset["model_name"].eq(model)].sort_values(
                "fraction"
            )
            ax.plot(
                model_rows["fraction"] * 100,
                model_rows["true_class_confidence"],
                marker="o",
                linewidth=2,
                color=color,
                label=model.upper(),
            )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("True-class confidence")
        ax.set_title(f"{true_class}: {video_id}")
        ax.grid(alpha=0.25)
        ax.legend()
    axes[-1].set_xlabel("Video revealed (%)")
    fig.tight_layout()
    fig.savefig(out_dir / "temporal_progression.png", dpi=220, bbox_inches="tight")
    fig.savefig(out_dir / "temporal_progression.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Create progressive clips, infer both models, and plot confidence."""
    args = parse_args()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.results_root = args.results_root.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = pd.read_csv(args.frames_metadata)
    _create_progressive_metadata(source, args.work_dir, args.examples)
    metadata_path = args.work_dir / "progressive_frames_metadata.csv"
    _run_extractors(args, metadata_path)
    predictions = _predict(args)
    predictions.to_csv(args.out_dir / "temporal_progression.csv", index=False)
    _plot(predictions, args.out_dir)


if __name__ == "__main__":
    main()
