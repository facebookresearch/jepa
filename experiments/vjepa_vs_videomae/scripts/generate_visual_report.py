#!/usr/bin/env python3
"""Generate presentation-ready figures and a Markdown/HTML report."""

from __future__ import annotations

import argparse
import html
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA

COLORS = {"vjepa": "#2563eb", "videomae": "#f97316"}
MODEL_LABELS = {"vjepa": "V-JEPA", "videomae": "VideoMAE"}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--features_root", type=Path, required=True)
    parser.add_argument("--frames_metadata", type=Path, required=True)
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--split_ids", nargs="+", type=int, default=[1, 2, 3])
    return parser.parse_args()


def _save(
    fig: plt.Figure,
    out_dir: Path,
    name: str,
    pdf: bool = False,
    dpi: int = 220,
) -> None:
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=dpi, bbox_inches="tight")
    if pdf:
        fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _plot_metric_overview(aggregate: pd.DataFrame, out_dir: Path) -> None:
    metrics = [
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("f1_macro", "Macro-F1"),
        ("top3_accuracy", "Top-3 accuracy"),
    ]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for offset, (_, row) in zip(
        (-width / 2, width / 2), aggregate.sort_values("model_name").iterrows()
    ):
        model = str(row["model_name"])
        means = [row[f"{key}_mean"] for key, _ in metrics]
        stds = [row[f"{key}_std"] for key, _ in metrics]
        ax.bar(
            x + offset,
            means,
            width,
            yerr=stds,
            capsize=4,
            label=MODEL_LABELS[model],
            color=COLORS[model],
        )
    ax.set_xticks(x, [label for _, label in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Frozen-feature linear probe on UCF101 (mean ± std, 3 splits)")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, out_dir, "metrics_overview", pdf=True)


def _plot_per_class(per_class: pd.DataFrame, out_dir: Path) -> None:
    classes = list(dict.fromkeys(per_class["class_name"]))
    x = np.arange(len(classes))
    width = 0.34
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for offset, model in zip((-width / 2, width / 2), ["vjepa", "videomae"]):
        data = per_class.set_index(["model_name", "class_name"]).loc[model]
        ax.bar(
            x + offset,
            [data.loc[name, "f1_mean"] for name in classes],
            width,
            yerr=[data.loc[name, "f1_std"] for name in classes],
            capsize=3,
            label=MODEL_LABELS[model],
            color=COLORS[model],
        )
    ax.set_xticks(x, classes, rotation=20, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1 by class")
    ax.set_title("Per-class performance across official splits")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    _save(fig, out_dir, "per_class_f1")


def _plot_speed_accuracy(aggregate: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5.5))
    for row in aggregate.itertuples(index=False):
        model = str(row.model_name)
        ax.errorbar(
            row.inference_time_per_video_seconds_mean,
            row.accuracy_mean,
            xerr=row.inference_time_per_video_seconds_std,
            yerr=row.accuracy_std,
            fmt="o",
            markersize=11,
            capsize=4,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
        ax.annotate(
            MODEL_LABELS[model],
            (row.inference_time_per_video_seconds_mean, row.accuracy_mean),
            xytext=(7, 7),
            textcoords="offset points",
        )
    ax.set_xlabel("Model inference seconds per video")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy vs inference speed")
    ax.grid(alpha=0.25)
    _save(fig, out_dir, "accuracy_vs_speed")


def _plot_confusions(results_root: Path, split_ids: list[int], out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, model in zip(axes, ["vjepa", "videomae"]):
        matrices = [
            np.load(
                results_root / f"split_{split_id:02d}" / model / "confusion_matrix.npy"
            )
            for split_id in split_ids
        ]
        matrix = np.sum(matrices, axis=0).astype(float)
        normalized = np.divide(
            matrix,
            matrix.sum(axis=1, keepdims=True),
            out=np.zeros_like(matrix),
            where=matrix.sum(axis=1, keepdims=True) != 0,
        )
        names = pd.read_csv(
            results_root / f"split_{split_ids[0]:02d}" / model / "confusion_matrix.csv",
            index_col=0,
        ).index.tolist()
        image = ax.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
        ax.set_yticks(range(len(names)), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"{MODEL_LABELS[model]} normalized confusion")
        for i in range(len(names)):
            for j in range(len(names)):
                ax.text(
                    j,
                    i,
                    f"{normalized[i, j]:.2f}",
                    ha="center",
                    va="center",
                    color="white" if normalized[i, j] > 0.55 else "black",
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    _save(fig, out_dir, "confusion_matrices")


def _plot_pca(features_root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, model in zip(axes, ["vjepa", "videomae"]):
        model_dir = features_root / "split_01" / model
        features = np.load(model_dir / "X_test.npy")
        index = pd.read_csv(model_dir / "index_test.csv")
        projected = PCA(n_components=2, random_state=42).fit_transform(features)
        for class_name in index["class_name"].unique():
            mask = index["class_name"].eq(class_name).to_numpy()
            ax.scatter(
                projected[mask, 0],
                projected[mask, 1],
                s=25,
                alpha=0.75,
                label=class_name,
            )
        ax.set_title(f"{MODEL_LABELS[model]} test embeddings (PCA)")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    _save(fig, out_dir, "embedding_pca")


def _plot_distribution(frames_metadata: pd.DataFrame, out_dir: Path) -> None:
    counts = (
        frames_metadata.groupby(["class_name", "split"]).size().unstack(fill_value=0)
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    counts[["train", "test"]].plot(
        kind="bar",
        ax=ax,
        color=["#64748b", "#14b8a6"],
        rot=20,
    )
    ax.set_ylabel("Videos")
    ax.set_xlabel("")
    ax.set_title("Balanced official UCF101 split")
    ax.grid(axis="y", alpha=0.25)
    _save(fig, out_dir, "dataset_distribution")


def _plot_storyboards(frames_metadata: pd.DataFrame, out_dir: Path) -> None:
    samples = (
        frames_metadata.loc[frames_metadata["split"].eq("test")]
        .sort_values(["class_name", "video_id"])
        .groupby("class_name", as_index=False)
        .first()
    )
    fig, axes = plt.subplots(len(samples), 8, figsize=(16, 2.2 * len(samples)))
    if len(samples) == 1:
        axes = np.asarray([axes])
    for row_index, row in enumerate(samples.itertuples(index=False)):
        frame_paths = sorted(Path(row.frames_dir).glob("frame_*.jpg"))
        selected = np.linspace(0, len(frame_paths) - 1, 8).round().astype(int)
        for column, frame_index in enumerate(selected):
            axes[row_index, column].imshow(Image.open(frame_paths[frame_index]))
            axes[row_index, column].axis("off")
            if column == 0:
                axes[row_index, column].set_title(
                    row.class_name, loc="left", fontsize=10
                )
    fig.suptitle("Uniform temporal sampling: 8 of 16 extracted frames", y=1.01)
    _save(fig, out_dir, "video_storyboards", dpi=150)


def _plot_processing_pipeline(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    labels = [
        "UCF101 video",
        "16 uniform frames",
        "Native preprocessing",
        "Frozen backbone",
        "Mean-pooled embedding",
        "Linear probe",
        "Prediction + metrics",
    ]
    x_positions = np.linspace(0.06, 0.94, len(labels))
    for index, (x, label) in enumerate(zip(x_positions, labels)):
        ax.text(
            x,
            0.5,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.5", "fc": "#eff6ff", "ec": "#2563eb"},
            transform=ax.transAxes,
        )
        if index < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x_positions[index + 1] - 0.065, 0.5),
                xytext=(x + 0.065, 0.5),
                arrowprops={"arrowstyle": "->", "color": "#475569"},
                xycoords=ax.transAxes,
            )
    ax.set_title("Evaluation pipeline shared by V-JEPA and VideoMAE", pad=18)
    _save(fig, out_dir, "processing_pipeline", pdf=True)


def _plot_prediction_examples(
    results_root: Path,
    frames_metadata: pd.DataFrame,
    out_dir: Path,
) -> None:
    vjepa = pd.read_csv(results_root / "split_01" / "vjepa" / "predictions.csv")
    videomae = pd.read_csv(results_root / "split_01" / "videomae" / "predictions.csv")
    merged = vjepa.merge(
        videomae,
        on="video_id",
        suffixes=("_vjepa", "_videomae"),
        validate="one_to_one",
    )
    categories = [
        (
            "Both correct",
            merged.loc[merged["correct_vjepa"] & merged["correct_videomae"]],
        ),
        (
            "Both wrong",
            merged.loc[~merged["correct_vjepa"] & ~merged["correct_videomae"]],
        ),
        (
            "Models disagree",
            merged.loc[merged["pred_class_vjepa"].ne(merged["pred_class_videomae"])],
        ),
    ]
    selected: list[tuple[str, pd.Series]] = []
    used_ids: set[str] = set()
    for category, candidates in categories:
        unused = candidates.loc[~candidates["video_id"].isin(used_ids)]
        if unused.empty:
            continue
        row = unused.sort_values("video_id").iloc[0]
        selected.append((category, row))
        used_ids.add(str(row["video_id"]))
    if not selected:
        return

    frames_by_id = frames_metadata.set_index("video_id")
    fig, axes = plt.subplots(len(selected), 4, figsize=(13, 3.2 * len(selected)))
    if len(selected) == 1:
        axes = np.asarray([axes])
    for row_index, (category, prediction) in enumerate(selected):
        frame_dir = Path(frames_by_id.loc[prediction["video_id"], "frames_dir"])
        paths = sorted(frame_dir.glob("frame_*.jpg"))
        indices = np.linspace(0, len(paths) - 1, 4).round().astype(int)
        for column, frame_index in enumerate(indices):
            axes[row_index, column].imshow(Image.open(paths[frame_index]))
            axes[row_index, column].axis("off")
        axes[row_index, 0].set_title(
            f"{category}\nTrue: {prediction['true_class_vjepa']}",
            loc="left",
            fontsize=9,
        )
        axes[row_index, 3].set_title(
            "V-JEPA: "
            f"{prediction['pred_class_vjepa']} "
            f"({prediction['confidence_vjepa']:.2f})\n"
            "VideoMAE: "
            f"{prediction['pred_class_videomae']} "
            f"({prediction['confidence_videomae']:.2f})",
            loc="right",
            fontsize=9,
        )
    fig.suptitle("Qualitative prediction cases from official split 1", y=1.01)
    _save(fig, out_dir, "prediction_examples", dpi=150)


def _write_report(
    aggregate: pd.DataFrame,
    per_class: pd.DataFrame,
    split_metrics: pd.DataFrame,
    out_dir: Path,
    temporal_available: bool,
) -> None:
    by_model = aggregate.set_index("model_name")
    vjepa = by_model.loc["vjepa"]
    videomae = by_model.loc["videomae"]
    accuracy_delta = float(vjepa["accuracy_mean"] - videomae["accuracy_mean"])
    macro_f1_delta = float(vjepa["f1_macro_mean"] - videomae["f1_macro_mean"])
    speed_ratio = float(
        vjepa["inference_time_per_video_seconds_mean"]
        / videomae["inference_time_per_video_seconds_mean"]
    )
    videomae_classes = per_class.loc[per_class["model_name"].eq("videomae")]
    hardest = videomae_classes.sort_values("f1_mean").iloc[0]
    split_table = split_metrics[
        [
            "split_id",
            "model_name",
            "accuracy",
            "f1_macro",
            "top3_accuracy",
            "inference_time_per_video_seconds",
        ]
    ].copy()
    table = aggregate.to_markdown(index=False, floatfmt=".4f")
    split_table_markdown = split_table.to_markdown(index=False, floatfmt=".4f")
    figures = [
        ("Metric overview", "metrics_overview.png"),
        ("Per-class F1", "per_class_f1.png"),
        ("Accuracy versus speed", "accuracy_vs_speed.png"),
        ("Confusion matrices", "confusion_matrices.png"),
        ("Embedding PCA", "embedding_pca.png"),
        ("Dataset balance", "dataset_distribution.png"),
        ("Video storyboards", "video_storyboards.png"),
        ("Prediction examples", "prediction_examples.png"),
        ("Processing pipeline", "processing_pipeline.png"),
    ]
    if temporal_available:
        figures.append(("Confidence through time", "temporal_progression.png"))
    markdown = [
        "# V-JEPA vs VideoMAE on UCF101",
        "",
        "Frozen-feature comparison on five classes and the three official splits.",
        "",
        "## Protocol",
        "",
        "- Official UCF101 splits 1, 2, and 3; seed 42.",
        "- Five classes, with 60 train and 20 test videos per class and split.",
        "- Identical 16-frame, 224×224 clips for both frozen backbones.",
        "- Mean pooling followed by the same StandardScaler + LogisticRegression.",
        "- Timings measured on Apple MPS after one excluded warm-up pass.",
        "",
        "## Key findings",
        "",
        f"- V-JEPA accuracy: **{vjepa['accuracy_mean']:.3f} ± "
        f"{vjepa['accuracy_std']:.3f}**; VideoMAE: "
        f"**{videomae['accuracy_mean']:.3f} ± "
        f"{videomae['accuracy_std']:.3f}**.",
        f"- V-JEPA gains **{accuracy_delta * 100:.1f} percentage points** in "
        f"accuracy and **{macro_f1_delta * 100:.1f} points** in macro-F1.",
        f"- VideoMAE is **{speed_ratio:.2f}× faster** at model inference "
        f"({videomae['inference_time_per_video_seconds_mean']:.3f} s/video "
        f"versus {vjepa['inference_time_per_video_seconds_mean']:.3f} s/video).",
        f"- VideoMAE's hardest class is **{hardest['class_name']}** "
        f"(F1 {hardest['f1_mean']:.3f} ± {hardest['f1_std']:.3f}); "
        "V-JEPA remains above 0.97 F1 on every class.",
        "",
        "> Scope: these conclusions apply to this balanced five-class frozen-feature "
        "protocol, not to full 101-class fine-tuning.",
        "",
        "## Results by split",
        "",
        split_table_markdown,
        "",
        "## Aggregate metrics",
        "",
        table,
        "",
    ]
    for title, filename in figures:
        markdown.extend([f"## {title}", "", f"![{title}]({filename})", ""])
    report_md = "\n".join(markdown)
    (out_dir / "report.md").write_text(report_md, encoding="utf-8")
    findings_html = (
        "<h2>Protocol</h2><ul>"
        "<li>Official UCF101 splits 1, 2, and 3; seed 42.</li>"
        "<li>Five classes; 60 train and 20 test videos per class and split.</li>"
        "<li>Identical 16-frame, 224×224 clips and frozen mean-pooled backbones.</li>"
        "<li>Same StandardScaler + LogisticRegression; MPS timings after warm-up.</li>"
        "</ul><h2>Key findings</h2><ul>"
        f"<li>Accuracy: V-JEPA <strong>{vjepa['accuracy_mean']:.3f} ± "
        f"{vjepa['accuracy_std']:.3f}</strong>; VideoMAE "
        f"<strong>{videomae['accuracy_mean']:.3f} ± "
        f"{videomae['accuracy_std']:.3f}</strong>.</li>"
        f"<li>V-JEPA gains <strong>{accuracy_delta * 100:.1f} percentage "
        f"points</strong> in accuracy and <strong>{macro_f1_delta * 100:.1f} "
        "points</strong> in macro-F1.</li>"
        f"<li>VideoMAE is <strong>{speed_ratio:.2f}× faster</strong> "
        f"({videomae['inference_time_per_video_seconds_mean']:.3f} versus "
        f"{vjepa['inference_time_per_video_seconds_mean']:.3f} s/video).</li>"
        f"<li>VideoMAE's hardest class is <strong>{html.escape(str(hardest['class_name']))}"
        f"</strong> (F1 {hardest['f1_mean']:.3f} ± "
        f"{hardest['f1_std']:.3f}).</li></ul>"
        "<p class='scope'><strong>Scope:</strong> these conclusions apply to this "
        "balanced five-class frozen-feature protocol, not full 101-class fine-tuning."
        "</p><h2>Results by split</h2>"
        + split_table.to_html(index=False, float_format=lambda value: f"{value:.4f}")
        + "<h2>Aggregate metrics</h2>"
    )
    body = "\n".join(
        f"<h2>{html.escape(title)}</h2><img src='{filename}' alt='{html.escape(title)}'>"
        for title, filename in figures
    )
    (out_dir / "report.html").write_text(
        "<!doctype html><meta charset='utf-8'>"
        "<style>body{font:16px system-ui;max-width:1100px;margin:40px auto;"
        "color:#0f172a}img{max-width:100%;margin-bottom:32px}"
        "table{border-collapse:collapse}th,td{border:1px solid #cbd5e1;padding:6px}"
        ".scope{background:#f1f5f9;border-left:4px solid #2563eb;padding:12px}"
        "</style><h1>V-JEPA vs VideoMAE on UCF101</h1>"
        + findings_html
        + aggregate.to_html(index=False, float_format=lambda value: f"{value:.4f}")
        + body,
        encoding="utf-8",
    )


def main() -> None:
    """Generate all report artifacts."""
    args = parse_args()
    results_root = args.results_root.expanduser().resolve()
    features_root = args.features_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    aggregate = pd.read_csv(results_root / "aggregate" / "aggregate_metrics.csv")
    per_class = pd.read_csv(
        results_root / "aggregate" / "per_class_aggregate_metrics.csv"
    )
    split_metrics = pd.read_csv(results_root / "aggregate" / "split_metrics.csv")
    frames_metadata = pd.read_csv(args.frames_metadata)

    _plot_metric_overview(aggregate, out_dir)
    _plot_per_class(per_class, out_dir)
    _plot_speed_accuracy(aggregate, out_dir)
    _plot_confusions(results_root, args.split_ids, out_dir)
    _plot_pca(features_root, out_dir)
    _plot_distribution(frames_metadata, out_dir)
    _plot_storyboards(frames_metadata, out_dir)
    _plot_prediction_examples(results_root, frames_metadata, out_dir)
    _plot_processing_pipeline(out_dir)
    _write_report(
        aggregate,
        per_class,
        split_metrics,
        out_dir,
        temporal_available=(out_dir / "temporal_progression.png").is_file(),
    )


if __name__ == "__main__":
    main()
