"""Metric computation and evaluation artifact persistence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the scalar metrics required by the experiment protocol."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(
            f1_score(y_true, y_pred, average="weighted", zero_division=0)
        ),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
    }


def save_json(data: dict[str, object], path: Path) -> None:
    """Serialize a dictionary as readable UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def save_classification_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    class_names: list[str],
    out_dir: Path,
) -> None:
    """Save sklearn classification reports as text and JSON."""
    text_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
    )
    (out_dir / "classification_report.txt").write_text(text_report, encoding="utf-8")
    json_report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        zero_division=0,
        output_dict=True,
    )
    save_json(json_report, out_dir / "classification_report.json")


def save_confusion_matrices(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    class_names: list[str],
    out_dir: Path,
) -> None:
    """Save the confusion matrix as NumPy and labeled CSV files."""
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    np.save(out_dir / "confusion_matrix.npy", matrix)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        out_dir / "confusion_matrix.csv", index_label="true_class"
    )


def save_predictions(
    index_test: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    score_classes: np.ndarray,
    label_to_class: dict[int, str],
    out_dir: Path,
) -> None:
    """Save per-video predictions, confidence, and JSON-encoded top-3 values."""
    if confidences.ndim != 2 or confidences.shape[0] != len(y_true):
        raise ValueError("Confidence matrix shape does not match test labels.")
    if confidences.shape[1] != len(score_classes):
        raise ValueError("Confidence columns do not match classifier classes.")

    rows: list[dict[str, object]] = []
    top_k = min(3, confidences.shape[1])
    for position, (true_label, pred_label) in enumerate(zip(y_true, y_pred)):
        order = np.argsort(confidences[position])[::-1][:top_k]
        top_labels = [int(score_classes[column]) for column in order]
        top_names = [label_to_class[label] for label in top_labels]
        top_confidences = [float(confidences[position, column]) for column in order]
        predicted_column = int(np.where(score_classes == pred_label)[0][0])
        source = index_test.iloc[position]
        rows.append(
            {
                "video_id": source["video_id"],
                "video_path": source["video_path"],
                "split": source["split"],
                "true_label": int(true_label),
                "true_class": label_to_class[int(true_label)],
                "pred_label": int(pred_label),
                "pred_class": label_to_class[int(pred_label)],
                "correct": bool(true_label == pred_label),
                "confidence": float(confidences[position, predicted_column]),
                "top3_predictions": json.dumps(top_names),
                "top3_confidences": json.dumps(top_confidences),
            }
        )
    pd.DataFrame(rows).to_csv(out_dir / "predictions.csv", index=False)


def save_evaluation_artifacts(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    score_classes: np.ndarray,
    index_test: pd.DataFrame,
    label_to_class: dict[int, str],
    out_dir: Path,
) -> None:
    """Save reports, confusion matrices, and prediction details."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = sorted(label_to_class)
    class_names = [label_to_class[label] for label in labels]
    save_classification_reports(y_true, y_pred, labels, class_names, out_dir)
    save_confusion_matrices(y_true, y_pred, labels, class_names, out_dir)
    save_predictions(
        index_test,
        y_true,
        y_pred,
        confidences,
        score_classes,
        label_to_class,
        out_dir,
    )


def compute_top_k_accuracy(
    y_true: np.ndarray,
    confidences: np.ndarray,
    score_classes: np.ndarray,
    k: int = 3,
) -> float:
    """Compute top-k accuracy from classifier confidence columns."""
    top_k = min(k, confidences.shape[1])
    top_columns = np.argsort(confidences, axis=1)[:, -top_k:]
    top_labels = score_classes[top_columns]
    return float(
        np.mean([truth in labels for truth, labels in zip(y_true, top_labels)])
    )


def per_class_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_to_class: dict[int, str],
) -> pd.DataFrame:
    """Return one compact metrics row per class."""
    labels = sorted(label_to_class)
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=[label_to_class[label] for label in labels],
        zero_division=0,
        output_dict=True,
    )
    rows = []
    for label in labels:
        class_name = label_to_class[label]
        values = report[class_name]
        rows.append(
            {
                "label": label,
                "class_name": class_name,
                "precision": float(values["precision"]),
                "recall": float(values["recall"]),
                "f1_score": float(values["f1-score"]),
                "support": int(values["support"]),
            }
        )
    return pd.DataFrame(rows)
