#!/usr/bin/env python3
"""Train and evaluate a standardized linear classifier on frozen features."""

from __future__ import annotations

# ruff: noqa: E402

import argparse
import logging
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.vjepa_vs_videomae.src.feature_io import load_feature_splits
from experiments.vjepa_vs_videomae.src.metrics import (
    compute_metrics,
    compute_top_k_accuracy,
    per_class_metrics,
    save_evaluation_artifacts,
    save_json,
)

LOGGER = logging.getLogger("train_linear_probe")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features_dir", type=Path, required=True)
    parser.add_argument("--model_name", required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument(
        "--classifier",
        choices=["logistic_regression", "linear_svc"],
        default="logistic_regression",
    )
    parser.add_argument("--max_iter", type=int, default=2000)
    parser.add_argument("--run_config", type=Path, required=True)
    parser.add_argument("--device_used", required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    return parser.parse_args()


def _softmax(scores: np.ndarray) -> np.ndarray:
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)


def _linear_svc_confidences(classifier: LinearSVC, features: np.ndarray) -> np.ndarray:
    scores = classifier.decision_function(features)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return _softmax(scores)


def _label_mapping(
    index_train: pd.DataFrame, index_test: pd.DataFrame
) -> dict[int, str]:
    combined = pd.concat(
        [index_train[["label", "class_name"]], index_test[["label", "class_name"]]],
        ignore_index=True,
    ).drop_duplicates()
    if (
        combined["label"].duplicated().any()
        or combined["class_name"].duplicated().any()
    ):
        raise ValueError("Labels and class names are not in a one-to-one relationship.")
    return {
        int(row.label): str(row.class_name)
        for row in combined.sort_values("label").itertuples(index=False)
    }


def main() -> None:
    """Fit the requested classifier and save all evaluation outputs."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if args.max_iter <= 0:
        raise ValueError("--max_iter must be positive.")
    features_dir = args.features_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    run_config = args.run_config.expanduser().resolve()
    if not run_config.is_file():
        raise FileNotFoundError(f"Run config does not exist: {run_config}")
    with run_config.open("r", encoding="utf-8") as handle:
        if not isinstance(yaml.safe_load(handle), dict):
            raise ValueError(f"Run config must contain a YAML mapping: {run_config}")

    x_train, y_train, x_test, y_test, index_train, index_test = load_feature_splits(
        features_dir
    )
    if len(x_train) != len(index_train) or len(x_test) != len(index_test):
        raise ValueError("Feature arrays and index CSV lengths do not match.")
    label_to_class = _label_mapping(index_train, index_test)
    if set(np.unique(y_train)) != set(label_to_class):
        raise ValueError("Every class must be represented in the training split.")

    scaler = StandardScaler()
    if args.classifier == "logistic_regression":
        classifier = LogisticRegression(max_iter=args.max_iter, random_state=42)
    else:
        classifier = LinearSVC(max_iter=args.max_iter, random_state=42)

    train_start = time.perf_counter()
    x_train_scaled = scaler.fit_transform(x_train)
    classifier.fit(x_train_scaled, y_train)
    train_time = time.perf_counter() - train_start
    x_test_scaled = scaler.transform(x_test)
    y_pred = classifier.predict(x_test_scaled)
    if args.classifier == "logistic_regression":
        confidences = classifier.predict_proba(x_test_scaled)
    else:
        confidences = _linear_svc_confidences(classifier, x_test_scaled)

    timing_path = features_dir / "timing.csv"
    summary_path = features_dir / "feature_summary.json"
    if not timing_path.is_file() or not summary_path.is_file():
        raise FileNotFoundError(
            f"{features_dir} must contain timing.csv and feature_summary.json."
        )
    timing = pd.read_csv(timing_path)
    timing_devices = timing["device"].dropna().astype(str).unique().tolist()
    if len(timing_devices) != 1:
        raise ValueError(f"Expected one extraction device, found: {timing_devices}")
    actual_device = timing_devices[0]
    if args.device_used not in {"auto", actual_device}:
        LOGGER.warning(
            "--device_used=%s differs from timing.csv device=%s; using timing.csv.",
            args.device_used,
            actual_device,
        )
    metadata_frames = set(index_train["num_frames"]) | set(index_test["num_frames"])
    metadata_sizes = set(index_train["image_size"]) | set(index_test["image_size"])
    if metadata_frames != {args.num_frames}:
        raise ValueError(
            f"--num_frames={args.num_frames} disagrees with feature indexes: "
            f"{sorted(metadata_frames)}"
        )
    if metadata_sizes != {args.image_size}:
        raise ValueError(
            f"--image_size={args.image_size} disagrees with feature indexes: "
            f"{sorted(metadata_sizes)}"
        )
    pipeline_total = float(timing["extraction_time_seconds"].sum())
    inference_column = (
        "model_inference_time_seconds"
        if "model_inference_time_seconds" in timing
        else "extraction_time_seconds"
    )
    inference_total = float(timing[inference_column].sum())
    inference_per_video = inference_total / len(timing)

    out_dir.mkdir(parents=True, exist_ok=True)
    scalar_metrics = compute_metrics(y_test, y_pred)
    metrics: dict[str, object] = {
        "model_name": args.model_name,
        "feature_dim": int(x_train.shape[1]),
        "num_train_samples": int(len(x_train)),
        "num_test_samples": int(len(x_test)),
        "num_classes": int(len(label_to_class)),
        **scalar_metrics,
        "top3_accuracy": compute_top_k_accuracy(
            y_test, confidences, np.asarray(classifier.classes_), k=3
        ),
        "train_time_seconds": float(train_time),
        "inference_time_total_seconds": inference_total,
        "inference_time_per_video_seconds": inference_per_video,
        "pipeline_time_total_seconds": pipeline_total,
        "pipeline_time_per_video_seconds": pipeline_total / len(timing),
        "classifier_type": args.classifier,
        "num_frames": int(args.num_frames),
        "image_size": int(args.image_size),
        "device_used": actual_device,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "class_names": [label_to_class[label] for label in sorted(label_to_class)],
        "train_video_ids": index_train["video_id"].astype(str).tolist(),
        "test_video_ids": index_test["video_id"].astype(str).tolist(),
    }
    save_json(metrics, out_dir / "metrics.json")
    save_evaluation_artifacts(
        y_true=y_test,
        y_pred=y_pred,
        confidences=confidences,
        score_classes=np.asarray(classifier.classes_),
        index_test=index_test,
        label_to_class=label_to_class,
        out_dir=out_dir,
    )
    per_class_metrics(y_test, y_pred, label_to_class).to_csv(
        out_dir / "per_class_metrics.csv", index=False
    )
    dump(
        {"scaler": scaler, "classifier": classifier, "label_to_class": label_to_class},
        out_dir / "linear_probe.joblib",
    )
    shutil.copy2(summary_path, out_dir / "feature_summary.json")
    shutil.copy2(timing_path, out_dir / "timing.csv")
    shutil.copy2(run_config, out_dir / "run_config.yaml")
    extraction_manifest = features_dir / "extraction_manifest.json"
    if extraction_manifest.is_file():
        shutil.copy2(extraction_manifest, out_dir / "extraction_manifest.json")
    LOGGER.info(
        "Saved %s probe results to %s (accuracy %.4f, macro-F1 %.4f)",
        args.model_name,
        out_dir,
        scalar_metrics["accuracy"],
        scalar_metrics["f1_macro"],
    )


if __name__ == "__main__":
    main()
