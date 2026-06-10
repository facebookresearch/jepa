#!/usr/bin/env python3
"""Compare V-JEPA and VideoMAE result directories."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger("compare_metrics")

COMPARISON_COLUMNS = [
    "model_name",
    "accuracy",
    "balanced_accuracy",
    "f1_macro",
    "f1_weighted",
    "precision_macro",
    "recall_macro",
    "inference_time_total_seconds",
    "inference_time_per_video_seconds",
    "train_time_seconds",
    "feature_dim",
    "num_train_samples",
    "num_test_samples",
    "device_used",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    default_results = Path(__file__).resolve().parents[1] / "outputs" / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_dir", type=Path, default=default_results)
    parser.add_argument("--vjepa_name", default="vjepa")
    parser.add_argument("--videomae_name", default="videomae")
    return parser.parse_args()


def _load_metrics(results_dir: Path, model_dir: str) -> dict[str, object]:
    path = results_dir / model_dir / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    missing = [column for column in COMPARISON_COLUMNS if column not in metrics]
    if missing:
        raise ValueError(f"{path} is missing comparison fields: {missing}")
    return metrics


def _markdown_table(frame: pd.DataFrame) -> str:
    headers = list(frame.columns)
    rows = [
        [f"{value:.6f}" if isinstance(value, float) else str(value) for value in row]
        for row in frame.itertuples(index=False, name=None)
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines) + "\n"


def _best_model(
    first: dict[str, object],
    second: dict[str, object],
    metric: str,
    lower: bool = False,
) -> str:
    first_value = float(first[metric])
    second_value = float(second[metric])
    if first_value == second_value:
        return f"égalité ({first['model_name']} et {second['model_name']})"
    first_wins = first_value < second_value if lower else first_value > second_value
    return str(first["model_name"] if first_wins else second["model_name"])


def main() -> None:
    """Create comparison CSV, Markdown table, and readable summary."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    results_dir = args.results_dir.expanduser().resolve()
    vjepa = _load_metrics(results_dir, args.vjepa_name)
    videomae = _load_metrics(results_dir, args.videomae_name)
    comparison = pd.DataFrame(
        [
            {column: vjepa[column] for column in COMPARISON_COLUMNS},
            {column: videomae[column] for column in COMPARISON_COLUMNS},
        ],
        columns=COMPARISON_COLUMNS,
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(results_dir / "comparison_metrics.csv", index=False)
    (results_dir / "comparison_metrics.md").write_text(
        _markdown_table(comparison), encoding="utf-8"
    )

    same_counts = (
        vjepa["num_train_samples"] == videomae["num_train_samples"]
        and vjepa["num_test_samples"] == videomae["num_test_samples"]
    )
    same_classes = set(vjepa.get("class_names", [])) == set(
        videomae.get("class_names", [])
    )
    same_videos = set(vjepa.get("train_video_ids", [])) == set(
        videomae.get("train_video_ids", [])
    ) and set(vjepa.get("test_video_ids", [])) == set(
        videomae.get("test_video_ids", [])
    )
    accuracy_difference = float(vjepa["accuracy"]) - float(videomae["accuracy"])
    f1_difference = float(vjepa["f1_macro"]) - float(videomae["f1_macro"])
    lines = [
        "Comparaison V-JEPA vs VideoMAE",
        "",
        f"Meilleur modèle en accuracy : {_best_model(vjepa, videomae, 'accuracy')}",
        f"Meilleur modèle en f1_macro : {_best_model(vjepa, videomae, 'f1_macro')}",
        "Modèle le plus rapide en inférence : "
        f"{_best_model(vjepa, videomae, 'inference_time_per_video_seconds', lower=True)}",
        f"Différence d'accuracy (V-JEPA - VideoMAE) : {accuracy_difference:+.6f}",
        f"Différence de f1_macro (V-JEPA - VideoMAE) : {f1_difference:+.6f}",
    ]
    warnings: list[str] = []
    if not same_counts:
        warnings.append("les nombres d'échantillons train/test diffèrent")
    if not same_classes:
        warnings.append("les ensembles de classes diffèrent")
    if not same_videos:
        warnings.append("les vidéos de test ne sont pas exactement les mêmes")
    if warnings:
        lines.extend(["", "WARNING: " + "; ".join(warnings) + "."])
    else:
        lines.extend(
            ["", "Les deux modèles ont été évalués sur les mêmes vidéos et classes."]
        )
    (results_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Saved comparison artifacts to %s", results_dir)


if __name__ == "__main__":
    main()
