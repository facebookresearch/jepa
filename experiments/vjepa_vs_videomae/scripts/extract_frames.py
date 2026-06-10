#!/usr/bin/env python3
"""Extract a fixed number of uniformly sampled frames for each video."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vjepa_vs_videomae.src.dataset_index import (
    FRAMES_COLUMNS,
    SUBSET_COLUMNS,
    read_metadata,
)
from experiments.vjepa_vs_videomae.src.video_io import sample_video_frames

LOGGER = logging.getLogger("extract_frames")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--metadata_out", type=Path)
    parser.add_argument("--num_frames", type=int, choices=[8, 16], default=16)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def main() -> None:
    """Extract and persist one frame directory per metadata row."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.image_size <= 0:
        raise ValueError("--image_size must be positive.")
    metadata = read_metadata(args.metadata, SUBSET_COLUMNS)
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for row in tqdm(
        metadata.itertuples(index=False), total=len(metadata), desc="Videos"
    ):
        frames_dir = out_dir / str(row.video_id)
        frames_dir.mkdir(parents=True, exist_ok=True)
        existing_frames = sorted(frames_dir.glob("frame_*.jpg"))
        if len(existing_frames) != args.num_frames:
            for stale_frame in existing_frames:
                stale_frame.unlink()
            frames = sample_video_frames(
                Path(row.video_path), args.num_frames, args.image_size
            )
            if len(frames) != args.num_frames:
                raise RuntimeError(
                    f"Expected {args.num_frames} frames for {row.video_id}, "
                    f"got {len(frames)}."
                )
            for index, frame in enumerate(frames):
                Image.fromarray(frame).save(
                    frames_dir / f"frame_{index:03d}.jpg", quality=95
                )
        records.append(
            {
                "video_id": row.video_id,
                "video_path": row.video_path,
                "class_name": row.class_name,
                "label": int(row.label),
                "split": row.split,
                "frames_dir": str(frames_dir),
                "num_frames": args.num_frames,
                "image_size": args.image_size,
            }
        )

    frames_metadata = pd.DataFrame(records)[FRAMES_COLUMNS]
    output_path = (
        args.metadata_out.expanduser().resolve()
        if args.metadata_out
        else out_dir.parent / "frames_metadata.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frames_metadata.to_csv(output_path, index=False)
    LOGGER.info("Saved frame metadata for %d videos to %s", len(records), output_path)


if __name__ == "__main__":
    main()
