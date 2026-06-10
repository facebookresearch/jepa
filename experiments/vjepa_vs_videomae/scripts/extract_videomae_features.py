#!/usr/bin/env python3
"""Extract frozen mean-pooled VideoMAE embeddings from frame directories."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import VideoMAEImageProcessor, VideoMAEModel

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vjepa_vs_videomae.src.dataset_index import (
    FRAMES_COLUMNS,
    read_metadata,
)
from experiments.vjepa_vs_videomae.src.device import (
    get_device,
    mps_error_message,
)
from experiments.vjepa_vs_videomae.src.feature_io import (
    load_cached_feature,
    save_cached_feature,
    save_feature_splits,
)
from experiments.vjepa_vs_videomae.src.reproducibility import (
    base_manifest,
    save_manifest,
)
from experiments.vjepa_vs_videomae.src.timing import Timer, save_timing
from experiments.vjepa_vs_videomae.src.video_io import load_frames_from_dir

LOGGER = logging.getLogger("extract_videomae_features")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_metadata", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--model_name", default="MCG-NJU/videomae-base")
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--cache_dir", type=Path)
    return parser.parse_args()


def _save_outputs(
    metadata: pd.DataFrame,
    embeddings: np.ndarray,
    timing_rows: list[dict[str, object]],
    out_dir: Path,
) -> None:
    train_mask = metadata["split"].eq("train").to_numpy()
    test_mask = metadata["split"].eq("test").to_numpy()
    save_feature_splits(
        out_dir=out_dir,
        x_train=embeddings[train_mask],
        y_train=metadata.loc[train_mask, "label"].to_numpy(dtype=np.int64),
        x_test=embeddings[test_mask],
        y_test=metadata.loc[test_mask, "label"].to_numpy(dtype=np.int64),
        index_train=metadata.loc[train_mask].reset_index(drop=True),
        index_test=metadata.loc[test_mask].reset_index(drop=True),
    )
    save_timing(timing_rows, out_dir / "timing.csv")


def _adapt_temporal_positions(model: VideoMAEModel, num_frames: int) -> None:
    """Adapt fixed VideoMAE temporal positions to an 8- or 16-frame clip."""
    tubelet_size = int(model.config.tubelet_size)
    configured_frames = int(model.config.num_frames)
    if num_frames % tubelet_size != 0:
        raise ValueError(
            f"num_frames={num_frames} must be divisible by tubelet_size={tubelet_size}."
        )
    if num_frames == configured_frames:
        return

    positions = model.embeddings.position_embeddings
    configured_temporal_tokens = configured_frames // tubelet_size
    requested_temporal_tokens = num_frames // tubelet_size
    if positions.shape[1] % configured_temporal_tokens != 0:
        raise ValueError("Unexpected VideoMAE positional embedding shape.")
    spatial_tokens = positions.shape[1] // configured_temporal_tokens
    temporal_indices = (
        torch.linspace(
            0,
            configured_temporal_tokens - 1,
            steps=requested_temporal_tokens,
        )
        .round()
        .long()
    )
    adapted = positions.reshape(
        1, configured_temporal_tokens, spatial_tokens, positions.shape[-1]
    )[:, temporal_indices]
    model.embeddings.position_embeddings = adapted.reshape(
        1, requested_temporal_tokens * spatial_tokens, positions.shape[-1]
    )
    model.embeddings.patch_embeddings.num_patches = (
        requested_temporal_tokens * spatial_tokens
    )
    model.config.num_frames = num_frames
    LOGGER.info(
        "Adapted VideoMAE temporal positions from %d to %d frames.",
        configured_frames,
        num_frames,
    )


def main() -> None:
    """Run frozen VideoMAE inference and save split feature arrays."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    metadata = read_metadata(args.frames_metadata, FRAMES_COLUMNS)
    if metadata["num_frames"].nunique() != 1:
        raise ValueError("All rows must use the same num_frames value.")
    device = get_device(args.device)
    LOGGER.info("Loading %s on %s", args.model_name, device)
    try:
        processor = VideoMAEImageProcessor.from_pretrained(args.model_name)
        model = VideoMAEModel.from_pretrained(args.model_name).to(device).eval()
        _adapt_temporal_positions(model, int(metadata["num_frames"].iloc[0]))
    except RuntimeError as error:
        if device.type == "mps":
            raise mps_error_message(error) from error
        raise
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir
        else args.out_dir.expanduser().resolve() / "cache"
    )
    embeddings_by_id: dict[str, np.ndarray] = {}
    timing_rows: list[dict[str, object]] = []
    try:
        warmup_row = metadata.iloc[0]
        warmup_video = load_frames_from_dir(Path(warmup_row["frames_dir"]))
        warmup_inputs = processor([warmup_video], return_tensors="pt")
        with torch.no_grad():
            model(pixel_values=warmup_inputs["pixel_values"].to(device))
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)

        for start in tqdm(
            range(0, len(metadata), args.batch_size), desc="VideoMAE batches"
        ):
            batch = metadata.iloc[start : start + args.batch_size]
            missing_rows = []
            for row in batch.itertuples(index=False):
                cached = load_cached_feature(cache_dir, str(row.video_id))
                if cached is None:
                    missing_rows.append(row)
                    continue
                feature, cached_timing = cached
                embeddings_by_id[str(row.video_id)] = feature
                timing_rows.append(
                    {
                        **cached_timing,
                        "split": row.split,
                        "cached": True,
                    }
                )
            if not missing_rows:
                continue

            load_start = time.perf_counter()
            videos = [
                load_frames_from_dir(Path(row.frames_dir)) for row in missing_rows
            ]
            load_elapsed = time.perf_counter() - load_start
            expected_counts = [int(row.num_frames) for row in missing_rows]
            actual_counts = [len(video) for video in videos]
            if actual_counts != expected_counts:
                raise ValueError(
                    f"Frame count mismatch: expected {expected_counts}, got {actual_counts}."
                )
            preprocess_start = time.perf_counter()
            inputs = processor(videos, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            preprocess_elapsed = time.perf_counter() - preprocess_start
            timer = Timer(device).start()
            with torch.no_grad():
                outputs = model(pixel_values=pixel_values)
                features = outputs.last_hidden_state.mean(dim=1)
            inference_elapsed = timer.stop()
            feature_array = features.detach().float().cpu().numpy()
            count = len(missing_rows)
            for position, row in enumerate(missing_rows):
                timing = {
                    "video_id": row.video_id,
                    "model_name": "videomae",
                    "split": row.split,
                    "load_time_seconds": load_elapsed / count,
                    "preprocess_time_seconds": preprocess_elapsed / count,
                    "model_inference_time_seconds": inference_elapsed / count,
                    "extraction_time_seconds": (
                        load_elapsed + preprocess_elapsed + inference_elapsed
                    )
                    / count,
                    "num_frames": int(row.num_frames),
                    "device": device.type,
                    "cached": False,
                }
                feature = feature_array[position]
                embeddings_by_id[str(row.video_id)] = feature
                timing_rows.append(timing)
                save_cached_feature(cache_dir, str(row.video_id), feature, timing)
    except RuntimeError as error:
        if device.type == "mps":
            raise mps_error_message(error) from error
        raise

    embeddings = np.stack(
        [embeddings_by_id[str(video_id)] for video_id in metadata["video_id"]]
    )
    out_dir = args.out_dir.expanduser().resolve()
    _save_outputs(metadata, embeddings, timing_rows, out_dir)
    manifest = base_manifest(REPO_ROOT)
    manifest.update(
        {
            "model_name": args.model_name,
            "model_revision": getattr(model.config, "_commit_hash", None),
            "pooling": "mean_patch_tokens",
            "device": device.type,
            "batch_size": args.batch_size,
            "num_frames": int(metadata["num_frames"].iloc[0]),
            "image_size": int(metadata["image_size"].iloc[0]),
            "warmup_excluded": True,
        }
    )
    save_manifest(manifest, out_dir / "extraction_manifest.json")
    LOGGER.info(
        "Saved VideoMAE features with shape %s to %s", embeddings.shape, out_dir
    )


if __name__ == "__main__":
    main()
