"""Protocol and reporting tests using small synthetic fixtures."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiments.vjepa_vs_videomae.src.dataset_index import select_official_subset
from experiments.vjepa_vs_videomae.src.metrics import compute_top_k_accuracy


class OfficialSplitTests(unittest.TestCase):
    """Validate deterministic, balanced official split selection."""

    def test_balanced_selection_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ucf_root = root / "UCF101"
            splits = root / "splits"
            splits.mkdir()
            classes = ["ClassA", "ClassB"]
            train_lines = []
            test_lines = []
            for class_name in classes:
                class_dir = ucf_root / class_name
                class_dir.mkdir(parents=True)
                for index in range(5):
                    name = f"{class_name}_{index}.avi"
                    (class_dir / name).touch()
                    relative = f"{class_name}/{name}"
                    if index < 3:
                        train_lines.append(f"{relative} 1")
                    else:
                        test_lines.append(relative)
            (splits / "trainlist01.txt").write_text(
                "\n".join(train_lines), encoding="utf-8"
            )
            (splits / "testlist01.txt").write_text(
                "\n".join(test_lines), encoding="utf-8"
            )

            first = select_official_subset(ucf_root, splits, classes, 1, 2, 1, seed=42)
            second = select_official_subset(ucf_root, splits, classes, 1, 2, 1, seed=42)
            self.assertTrue(first.equals(second))
            self.assertEqual(len(first), 6)
            self.assertEqual(
                first.groupby(["split", "class_name"]).size().to_dict(),
                {
                    ("test", "ClassA"): 1,
                    ("test", "ClassB"): 1,
                    ("train", "ClassA"): 2,
                    ("train", "ClassB"): 2,
                },
            )
            train_ids = set(first.loc[first["split"].eq("train"), "video_id"])
            test_ids = set(first.loc[first["split"].eq("test"), "video_id"])
            self.assertFalse(train_ids & test_ids)

    def test_insufficient_official_samples_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "UCF101" / "ClassA").mkdir(parents=True)
            splits = root / "splits"
            splits.mkdir()
            video = root / "UCF101" / "ClassA" / "one.avi"
            video.touch()
            (splits / "trainlist01.txt").write_text(
                "ClassA/one.avi 1\n", encoding="utf-8"
            )
            (splits / "testlist01.txt").write_text("ClassA/one.avi\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                select_official_subset(
                    root / "UCF101", splits, ["ClassA"], 1, 2, 1, seed=42
                )


class MetricTests(unittest.TestCase):
    """Validate confidence-derived metrics."""

    def test_top_three_accuracy(self) -> None:
        y_true = np.asarray([0, 1, 3])
        probabilities = np.asarray(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.4, 0.3, 0.2, 0.1],
                [0.4, 0.3, 0.2, 0.1],
            ]
        )
        classes = np.asarray([0, 1, 2, 3])
        self.assertAlmostEqual(
            compute_top_k_accuracy(y_true, probabilities, classes, k=3),
            2 / 3,
        )


if __name__ == "__main__":
    unittest.main()
