"""Device selection helpers with Apple Silicon support."""

from __future__ import annotations

import logging

import torch

LOGGER = logging.getLogger(__name__)


def get_device(requested: str) -> torch.device:
    """Return the requested PyTorch device, preferring MPS for ``auto``."""
    requested = requested.lower()
    if requested == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if requested == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        LOGGER.warning("MPS was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        LOGGER.warning("CUDA was requested but is unavailable; falling back to CPU.")
        return torch.device("cpu")
    if requested == "cpu":
        return torch.device("cpu")
    raise ValueError(
        f"Unsupported device {requested!r}. Choose from auto, cpu, mps, or cuda."
    )


def synchronize_device(device: torch.device) -> None:
    """Synchronize asynchronous work before or after a timing measurement."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def mps_error_message(error: Exception) -> RuntimeError:
    """Wrap an MPS failure with an actionable CPU fallback recommendation."""
    return RuntimeError(
        "Model execution failed on MPS. Retry the command with `--device cpu`. "
        f"Original error: {error}"
    )
