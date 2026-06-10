"""Persistence and validation helpers for extracted feature arrays."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_cached_feature(
    cache_dir: Path, video_id: str
) -> tuple[np.ndarray, dict[str, object]] | None:
    """Load one cached embedding and its timing metadata when available."""
    feature_path = cache_dir / f"{video_id}.npy"
    metadata_path = cache_dir / f"{video_id}.json"
    if not feature_path.is_file() or not metadata_path.is_file():
        return None
    feature = np.load(feature_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if feature.ndim != 1 or not np.isfinite(feature).all():
        return None
    return feature.astype(np.float32, copy=False), metadata


def save_cached_feature(
    cache_dir: Path,
    video_id: str,
    feature: np.ndarray,
    timing: dict[str, object],
) -> None:
    """Save one embedding and its timing metadata atomically enough for resume."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"{video_id}.npy", feature.astype(np.float32, copy=False))
    (cache_dir / f"{video_id}.json").write_text(
        json.dumps(timing, indent=2), encoding="utf-8"
    )


def validate_features(features: np.ndarray, name: str) -> None:
    """Reject malformed, NaN, or infinite feature matrices."""
    if features.ndim != 2:
        raise ValueError(f"{name} must be a 2D array, got shape {features.shape}.")
    if features.shape[0] == 0:
        raise ValueError(f"{name} contains no samples.")
    if not np.isfinite(features).all():
        raise ValueError(f"{name} contains NaN or infinite values.")


def save_index_csv(index: pd.DataFrame, path: Path) -> None:
    """Save a sample index CSV, creating its parent directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    index.to_csv(path, index=False)


def compute_feature_summary(
    x_train: np.ndarray, x_test: np.ndarray
) -> dict[str, object]:
    """Compute the required descriptive statistics across both splits."""
    combined = np.concatenate([x_train, x_test], axis=0)
    return {
        "feature_shape_train": list(x_train.shape),
        "feature_shape_test": list(x_test.shape),
        "feature_mean": float(np.mean(combined)),
        "feature_std": float(np.std(combined)),
        "feature_min": float(np.min(combined)),
        "feature_max": float(np.max(combined)),
        "contains_nan": bool(np.isnan(combined).any()),
        "contains_inf": bool(np.isinf(combined).any()),
    }


def save_feature_splits(
    out_dir: Path,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    index_train: pd.DataFrame,
    index_test: pd.DataFrame,
) -> dict[str, object]:
    """Validate and save train/test arrays, indexes, and summary JSON."""
    out_dir.mkdir(parents=True, exist_ok=True)
    validate_features(x_train, "X_train")
    validate_features(x_test, "X_test")
    if len(x_train) != len(y_train) or len(x_train) != len(index_train):
        raise ValueError("Train features, labels, and index lengths do not match.")
    if len(x_test) != len(y_test) or len(x_test) != len(index_test):
        raise ValueError("Test features, labels, and index lengths do not match.")
    if x_train.shape[1] != x_test.shape[1]:
        raise ValueError("Train and test feature dimensions do not match.")

    np.save(out_dir / "X_train.npy", x_train.astype(np.float32, copy=False))
    np.save(out_dir / "y_train.npy", y_train.astype(np.int64, copy=False))
    np.save(out_dir / "X_test.npy", x_test.astype(np.float32, copy=False))
    np.save(out_dir / "y_test.npy", y_test.astype(np.int64, copy=False))
    save_index_csv(index_train, out_dir / "index_train.csv")
    save_index_csv(index_test, out_dir / "index_test.csv")

    summary = compute_feature_summary(x_train, x_test)
    with (out_dir / "feature_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def load_feature_splits(
    features_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame, pd.DataFrame]:
    """Load all arrays and indexes required by a linear probe."""
    features_dir = features_dir.expanduser().resolve()
    required = [
        "X_train.npy",
        "y_train.npy",
        "X_test.npy",
        "y_test.npy",
        "index_train.csv",
        "index_test.csv",
    ]
    missing = [name for name in required if not (features_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing feature files in {features_dir}: {missing}")
    x_train = np.load(features_dir / "X_train.npy")
    y_train = np.load(features_dir / "y_train.npy")
    x_test = np.load(features_dir / "X_test.npy")
    y_test = np.load(features_dir / "y_test.npy")
    index_train = pd.read_csv(features_dir / "index_train.csv")
    index_test = pd.read_csv(features_dir / "index_test.csv")
    validate_features(x_train, "X_train")
    validate_features(x_test, "X_test")
    return x_train, y_train, x_test, y_test, index_train, index_test
