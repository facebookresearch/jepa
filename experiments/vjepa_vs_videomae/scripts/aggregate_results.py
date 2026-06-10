#!/usr/bin/env python3
"""Aggregate V-JEPA and VideoMAE metrics across official UCF101 splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "top3_accuracy",
    "inference_time_per_video_seconds",
    "pipeline_time_per_video_seconds",
    "train_time_seconds",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--split_ids", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--models", nargs="+", default=["vjepa", "videomae"])
    return parser.parse_args()


def _markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False, floatfmt=".4f") + "\n"


def main() -> None:
    """Write split-level and mean/std comparison tables."""
    args = parse_args()
    root = args.results_root.expanduser().resolve()
    rows: list[dict[str, object]] = []
    per_class_frames: list[pd.DataFrame] = []
    for split_id in args.split_ids:
        split_metrics: dict[str, dict[str, object]] = {}
        for model in args.models:
            result_dir = root / f"split_{split_id:02d}" / model
            metrics_path = result_dir / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing metrics: {metrics_path}")
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            split_metrics[model] = metrics
            rows.append(
                {
                    "split_id": split_id,
                    "model_name": model,
                    **{metric: metrics[metric] for metric in METRICS},
                    "feature_dim": metrics["feature_dim"],
                    "num_train_samples": metrics["num_train_samples"],
                    "num_test_samples": metrics["num_test_samples"],
                    "device_used": metrics["device_used"],
                }
            )
            per_class = pd.read_csv(result_dir / "per_class_metrics.csv")
            per_class.insert(0, "model_name", model)
            per_class.insert(0, "split_id", split_id)
            per_class_frames.append(per_class)
        reference = split_metrics[args.models[0]]
        for model in args.models[1:]:
            candidate = split_metrics[model]
            for key in ("train_video_ids", "test_video_ids", "class_names"):
                if set(reference[key]) != set(candidate[key]):
                    raise ValueError(
                        f"Split {split_id}: {model} and {args.models[0]} "
                        f"do not share identical {key}."
                    )

    split_frame = pd.DataFrame(rows)
    aggregate_rows: list[dict[str, object]] = []
    for model, group in split_frame.groupby("model_name", sort=False):
        row: dict[str, object] = {"model_name": model}
        for metric in METRICS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["feature_dim"] = int(group["feature_dim"].iloc[0])
        row["device_used"] = str(group["device_used"].iloc[0])
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)

    out_dir = root / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    split_frame.to_csv(out_dir / "split_metrics.csv", index=False)
    aggregate.to_csv(out_dir / "aggregate_metrics.csv", index=False)
    (out_dir / "aggregate_metrics.md").write_text(
        _markdown(aggregate), encoding="utf-8"
    )

    per_class_all = pd.concat(per_class_frames, ignore_index=True)
    per_class_all.to_csv(out_dir / "per_class_split_metrics.csv", index=False)
    per_class_aggregate = (
        per_class_all.groupby(["model_name", "class_name"], as_index=False)
        .agg(
            precision_mean=("precision", "mean"),
            precision_std=("precision", "std"),
            recall_mean=("recall", "mean"),
            recall_std=("recall", "std"),
            f1_mean=("f1_score", "mean"),
            f1_std=("f1_score", "std"),
        )
        .fillna(0.0)
    )
    per_class_aggregate.to_csv(out_dir / "per_class_aggregate_metrics.csv", index=False)

    best_accuracy = aggregate.loc[aggregate["accuracy_mean"].idxmax(), "model_name"]
    best_f1 = aggregate.loc[aggregate["f1_macro_mean"].idxmax(), "model_name"]
    fastest = aggregate.loc[
        aggregate["inference_time_per_video_seconds_mean"].idxmin(), "model_name"
    ]
    vjepa = aggregate.set_index("model_name").loc["vjepa"]
    videomae = aggregate.set_index("model_name").loc["videomae"]
    summary = [
        "V-JEPA vs VideoMAE - UCF101 official splits 1/2/3",
        "",
        f"Best mean accuracy: {best_accuracy}",
        f"Best mean macro-F1: {best_f1}",
        f"Fastest model inference: {fastest}",
        "Mean accuracy difference (V-JEPA - VideoMAE): "
        f"{vjepa['accuracy_mean'] - videomae['accuracy_mean']:+.4f}",
        "Mean macro-F1 difference (V-JEPA - VideoMAE): "
        f"{vjepa['f1_macro_mean'] - videomae['f1_macro_mean']:+.4f}",
    ]
    (out_dir / "summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
