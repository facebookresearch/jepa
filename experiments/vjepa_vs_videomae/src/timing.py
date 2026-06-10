"""Small timing utilities and timing CSV persistence."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch

from experiments.vjepa_vs_videomae.src.device import synchronize_device

TIMING_COLUMNS = [
    "video_id",
    "model_name",
    "split",
    "load_time_seconds",
    "preprocess_time_seconds",
    "model_inference_time_seconds",
    "extraction_time_seconds",
    "num_frames",
    "device",
    "cached",
]


@dataclass
class Timer:
    """Measure wall-clock duration, synchronizing asynchronous accelerators."""

    device: torch.device | None = None
    _start: float | None = field(default=None, init=False)

    def start(self) -> "Timer":
        """Start the timer and return it for convenient inline use."""
        if self.device is not None:
            synchronize_device(self.device)
        self._start = time.perf_counter()
        return self

    def stop(self) -> float:
        """Stop the timer and return elapsed seconds."""
        if self._start is None:
            raise RuntimeError("Timer.stop() called before Timer.start().")
        if self.device is not None:
            synchronize_device(self.device)
        elapsed = time.perf_counter() - self._start
        self._start = None
        return elapsed

    def __enter__(self) -> "Timer":
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.elapsed = self.stop()


def save_timing(rows: list[dict[str, object]], path: Path) -> None:
    """Save timing records using a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=TIMING_COLUMNS).to_csv(path, index=False)
