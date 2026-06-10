"""Dataset metadata loading, official split parsing, and identifiers."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

SUBSET_COLUMNS = ["video_id", "video_path", "class_name", "label", "split"]
FRAMES_COLUMNS = [
    "video_id",
    "video_path",
    "frames_dir",
    "class_name",
    "label",
    "split",
    "num_frames",
    "image_size",
]
VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".webm"}


def read_metadata(path: Path, required_columns: list[str]) -> pd.DataFrame:
    """Read a metadata CSV and validate its required columns and values."""
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Metadata file does not exist: {path}")
    frame = pd.read_csv(path)
    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"Metadata file is empty: {path}")
    if frame["video_id"].duplicated().any():
        duplicates = frame.loc[frame["video_id"].duplicated(), "video_id"].tolist()
        raise ValueError(f"Duplicate video_id values in {path}: {duplicates[:5]}")
    invalid_splits = sorted(set(frame["split"]) - {"train", "test"})
    if invalid_splits:
        raise ValueError(f"Unsupported split values in {path}: {invalid_splits}")
    return frame


def make_video_id(class_name: str, video_path: Path) -> str:
    """Create a readable, collision-resistant identifier for one video."""
    readable = re.sub(r"[^A-Za-z0-9_-]+", "_", f"{class_name}_{video_path.stem}")
    stable_source = f"{class_name}/{video_path.name}"
    digest = hashlib.sha1(stable_source.encode("utf-8")).hexdigest()[:8]
    return f"{readable}_{digest}"


def discover_class_videos(ucf_root: Path, class_name: str) -> list[Path]:
    """Return sorted video files found directly inside one UCF101 class."""
    class_dir = ucf_root / class_name
    if not class_dir.is_dir():
        raise FileNotFoundError(
            f"Requested UCF101 class directory is missing: {class_dir}"
        )
    return sorted(
        path.resolve()
        for path in class_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def read_official_split(
    splits_dir: Path,
    split_id: int,
) -> tuple[set[str], set[str]]:
    """Return normalized train and test paths from one official UCF101 split."""
    if split_id not in {1, 2, 3}:
        raise ValueError("UCF101 split_id must be 1, 2, or 3.")
    splits_dir = splits_dir.expanduser().resolve()
    train_path = splits_dir / f"trainlist{split_id:02d}.txt"
    test_path = splits_dir / f"testlist{split_id:02d}.txt"
    missing = [path for path in (train_path, test_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing official UCF101 split files: {missing}")

    train_entries = {
        line.split()[0].strip()
        for line in train_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    test_entries = {
        line.strip()
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    overlap = train_entries & test_entries
    if overlap:
        raise ValueError(
            f"Official split {split_id} contains train/test overlap: "
            f"{sorted(overlap)[:5]}"
        )
    return train_entries, test_entries


def select_official_subset(
    ucf_root: Path,
    splits_dir: Path,
    classes: list[str],
    split_id: int,
    train_per_class: int,
    test_per_class: int,
    seed: int,
) -> pd.DataFrame:
    """Select a balanced deterministic subset within an official UCF101 split."""
    import numpy as np

    if train_per_class <= 0 or test_per_class <= 0:
        raise ValueError("train_per_class and test_per_class must be positive.")
    if len(classes) != len(set(classes)):
        raise ValueError("classes contains duplicate class names.")

    train_entries, test_entries = read_official_split(splits_dir, split_id)
    rng = np.random.default_rng(seed + split_id)
    records: list[dict[str, object]] = []
    for label, class_name in enumerate(classes):
        class_videos = discover_class_videos(ucf_root, class_name)
        by_relative = {
            f"{class_name}/{video_path.name}": video_path for video_path in class_videos
        }
        train_candidates = sorted(set(by_relative) & train_entries)
        test_candidates = sorted(set(by_relative) & test_entries)
        if len(train_candidates) < train_per_class:
            raise ValueError(
                f"Split {split_id}, class {class_name}: requested "
                f"{train_per_class} train videos, only {len(train_candidates)} available."
            )
        if len(test_candidates) < test_per_class:
            raise ValueError(
                f"Split {split_id}, class {class_name}: requested "
                f"{test_per_class} test videos, only {len(test_candidates)} available."
            )

        for split, candidates, count in (
            ("train", train_candidates, train_per_class),
            ("test", test_candidates, test_per_class),
        ):
            selected = rng.choice(candidates, size=count, replace=False)
            for relative_path in sorted(selected.tolist()):
                video_path = by_relative[relative_path]
                records.append(
                    {
                        "video_id": make_video_id(class_name, video_path),
                        "video_path": str(video_path),
                        "class_name": class_name,
                        "label": label,
                        "split": split,
                    }
                )

    metadata = pd.DataFrame(records, columns=SUBSET_COLUMNS)
    if metadata["video_id"].duplicated().any():
        raise ValueError("Official subset selection produced duplicate video IDs.")
    expected = {("train", class_name): train_per_class for class_name in classes} | {
        ("test", class_name): test_per_class for class_name in classes
    }
    actual = metadata.groupby(["split", "class_name"]).size().to_dict()
    if actual != expected:
        raise ValueError(
            f"Unexpected class distribution: {actual}; expected {expected}"
        )
    return metadata.sort_values(["split", "label", "video_id"], kind="stable")
