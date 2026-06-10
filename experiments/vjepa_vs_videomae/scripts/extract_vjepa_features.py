#!/usr/bin/env python3
"""Extract frozen V-JEPA embeddings using the repository's official loader."""

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
import yaml
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evals.video_classification_frozen.utils import make_transforms
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
    cached_sha256_file,
    save_manifest,
)
from experiments.vjepa_vs_videomae.src.timing import Timer, save_timing
from experiments.vjepa_vs_videomae.src.video_io import load_frames_from_dir
from src.models import vision_transformer as vit

LOGGER = logging.getLogger("extract_vjepa_features")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames_metadata", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "mps", "cuda"], default="auto"
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--pooling", choices=["mean", "cls"], default="mean")
    parser.add_argument("--cache_dir", type=Path)
    return parser.parse_args()


def _read_model_config(path: Path) -> dict[str, object]:
    """Normalize official pretraining or frozen-eval YAML into model kwargs."""
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"Expected a YAML mapping in {path}.")

    if "pretrain" in config:
        pretrain = config["pretrain"]
        optimization = config.get("optimization", {})
        return {
            "model_name": pretrain["model_name"],
            "patch_size": int(pretrain.get("patch_size", 16)),
            "crop_size": int(optimization.get("resolution", 224)),
            "frames_per_clip": int(pretrain.get("frames_per_clip", 16)),
            "tubelet_size": int(pretrain.get("tubelet_size", 2)),
            "uniform_power": bool(pretrain.get("uniform_power", False)),
            "checkpoint_key": pretrain.get("checkpoint_key", "target_encoder"),
            "use_sdpa": bool(pretrain.get("use_sdpa", False)),
            "use_SiLU": bool(pretrain.get("use_silu", False)),
            "tight_SiLU": bool(pretrain.get("tight_silu", True)),
        }

    if "model" in config and "data" in config:
        model = config["model"]
        data = config["data"]
        meta = config.get("meta", {})
        return {
            "model_name": model["model_name"],
            "patch_size": int(data.get("patch_size", 16)),
            "crop_size": int(data.get("crop_size", 224)),
            "frames_per_clip": int(data.get("num_frames", 16)),
            "tubelet_size": int(data.get("tubelet_size", 2)),
            "uniform_power": bool(model.get("uniform_power", False)),
            "checkpoint_key": "target_encoder",
            "use_sdpa": bool(meta.get("use_sdpa", False)),
            "use_SiLU": False,
            "tight_SiLU": True,
        }
    raise ValueError(
        f"{path} is not a recognized official V-JEPA pretrain or eval config."
    )


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


def _init_model_memory_efficient(
    device: torch.device,
    checkpoint_path: Path,
    model_config: dict[str, object],
) -> torch.nn.Module:
    """Apply the official V-JEPA model/loading logic with a memory-mapped checkpoint."""
    model = vit.__dict__[str(model_config["model_name"])](
        img_size=int(model_config["crop_size"]),
        patch_size=int(model_config["patch_size"]),
        num_frames=int(model_config["frames_per_clip"]),
        tubelet_size=int(model_config["tubelet_size"]),
        uniform_power=bool(model_config["uniform_power"]),
        use_sdpa=bool(model_config["use_sdpa"]),
        use_SiLU=bool(model_config["use_SiLU"]),
        tight_SiLU=bool(model_config["tight_SiLU"]),
    ).to(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        mmap=True,
        weights_only=False,
    )
    checkpoint_key = str(model_config["checkpoint_key"])
    state = checkpoint.get(checkpoint_key, checkpoint.get("encoder"))
    if state is None:
        raise KeyError(f"Checkpoint contains neither {checkpoint_key!r} nor 'encoder'.")
    state = {
        key.replace("module.", "").replace("backbone.", ""): value
        for key, value in state.items()
    }
    mismatched = [
        key
        for key, value in model.state_dict().items()
        if key in state and state[key].shape != value.shape
    ]
    if mismatched:
        raise ValueError(
            f"V-JEPA checkpoint/model shape mismatch for keys: {mismatched[:10]}"
        )
    message = model.load_state_dict(state, strict=False)
    if message.missing_keys or message.unexpected_keys:
        LOGGER.warning("V-JEPA checkpoint load message: %s", message)
    del checkpoint
    return model


def main() -> None:
    """Load the official V-JEPA backbone and extract one feature per video."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive.")
    config_path = args.config.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"V-JEPA config does not exist: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"V-JEPA checkpoint does not exist: {checkpoint_path}")
    metadata = read_metadata(args.frames_metadata, FRAMES_COLUMNS)
    if metadata["num_frames"].nunique() != 1:
        raise ValueError("All rows must use the same num_frames value.")

    model_config = _read_model_config(config_path)
    actual_num_frames = int(metadata["num_frames"].iloc[0])
    tubelet_size = int(model_config["tubelet_size"])
    if actual_num_frames % tubelet_size != 0:
        raise ValueError(
            f"num_frames={actual_num_frames} must be divisible by "
            f"tubelet_size={tubelet_size}."
        )

    device = get_device(args.device)
    LOGGER.info(
        "Loading official V-JEPA %s checkpoint on %s",
        model_config["model_name"],
        device,
    )
    try:
        model = _init_model_memory_efficient(
            device=device,
            checkpoint_path=checkpoint_path,
            model_config=model_config,
        ).eval()
    except RuntimeError as error:
        if device.type == "mps":
            raise mps_error_message(error) from error
        raise
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    transform = make_transforms(
        training=False,
        crop_size=int(model_config["crop_size"]),
        num_views_per_clip=1,
    )

    cache_dir = (
        args.cache_dir.expanduser().resolve()
        if args.cache_dir
        else args.out_dir.expanduser().resolve() / "cache"
    )
    embeddings_by_id: dict[str, np.ndarray] = {}
    timing_rows: list[dict[str, object]] = []
    try:
        warmup_row = metadata.iloc[0]
        warmup_frames = load_frames_from_dir(Path(warmup_row["frames_dir"]))
        warmup_buffer = np.stack([np.asarray(frame) for frame in warmup_frames])
        warmup_clip = transform(warmup_buffer)[0].unsqueeze(0).to(device)
        with torch.no_grad():
            model(warmup_clip)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize(device)

        for start in tqdm(
            range(0, len(metadata), args.batch_size), desc="V-JEPA batches"
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
            loaded_frames = [
                load_frames_from_dir(Path(row.frames_dir)) for row in missing_rows
            ]
            load_elapsed = time.perf_counter() - load_start
            preprocess_start = time.perf_counter()
            clips: list[torch.Tensor] = []
            for row, frames in zip(missing_rows, loaded_frames):
                if len(frames) != int(row.num_frames):
                    raise ValueError(
                        f"Expected {row.num_frames} frames for {row.video_id}, "
                        f"found {len(frames)}."
                    )
                buffer = np.stack([np.asarray(frame) for frame in frames])
                clips.append(transform(buffer)[0])
            pixel_values = torch.stack(clips).to(device)
            preprocess_elapsed = time.perf_counter() - preprocess_start
            timer = Timer(device).start()
            with torch.no_grad():
                tokens = model(pixel_values)
                if args.pooling == "mean":
                    features = tokens.mean(dim=1)
                else:
                    # V-JEPA has patch tokens and no explicit CLS token; `cls`
                    # intentionally selects the first patch token.
                    features = tokens[:, 0]
            inference_elapsed = timer.stop()
            feature_array = features.detach().float().cpu().numpy()
            count = len(missing_rows)
            for position, row in enumerate(missing_rows):
                timing = {
                    "video_id": row.video_id,
                    "model_name": "vjepa",
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
            "model_name": str(model_config["model_name"]),
            "config_path": str(config_path),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": cached_sha256_file(checkpoint_path),
            "pooling": args.pooling,
            "device": device.type,
            "batch_size": args.batch_size,
            "num_frames": actual_num_frames,
            "image_size": int(metadata["image_size"].iloc[0]),
            "warmup_excluded": True,
        }
    )
    save_manifest(manifest, out_dir / "extraction_manifest.json")
    LOGGER.info("Saved V-JEPA features with shape %s to %s", embeddings.shape, out_dir)


if __name__ == "__main__":
    main()
