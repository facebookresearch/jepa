"""Video decoding and frame loading helpers."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

LOGGER = logging.getLogger(__name__)


def _uniform_indices(length: int, num_frames: int) -> np.ndarray:
    if length <= 0:
        raise ValueError("Cannot sample frames from an empty video.")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive.")
    return np.linspace(0, length - 1, num=num_frames).round().astype(np.int64)


def _resize_rgb(frame: np.ndarray, image_size: int) -> np.ndarray:
    return cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)


def _sample_with_decord(
    video_path: Path, num_frames: int, image_size: int
) -> list[np.ndarray]:
    import decord

    reader = decord.VideoReader(str(video_path), ctx=decord.cpu(0))
    indices = _uniform_indices(len(reader), num_frames)
    frames = reader.get_batch(indices).asnumpy()
    return [_resize_rgb(frame, image_size) for frame in frames]


def _sample_with_opencv(
    video_path: Path, num_frames: int, image_size: int
) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            raise RuntimeError(f"OpenCV reported no frames for video: {video_path}")
        indices = _uniform_indices(frame_count, num_frames)
        frames: list[np.ndarray] = []
        last_frame: np.ndarray | None = None
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, bgr = capture.read()
            if ok:
                last_frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            elif last_frame is None:
                raise RuntimeError(
                    f"OpenCV failed to decode frame {index} from {video_path}"
                )
            frames.append(_resize_rgb(last_frame.copy(), image_size))
        return frames
    finally:
        capture.release()


def sample_video_frames(
    video_path: Path, num_frames: int, image_size: int
) -> list[np.ndarray]:
    """Uniformly sample exactly ``num_frames`` RGB arrays from one video.

    Decord is attempted first. If it is unavailable or fails for a particular
    file, OpenCV is used as a fallback. Repeated uniform indices naturally
    repeat frames for videos shorter than the requested clip length.
    """
    video_path = video_path.expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"Video does not exist: {video_path}")
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    try:
        return _sample_with_decord(video_path, num_frames, image_size)
    except (ImportError, ModuleNotFoundError):
        LOGGER.debug("Decord is unavailable; using OpenCV for %s", video_path)
    except Exception as error:
        LOGGER.warning(
            "Decord failed for %s (%s); retrying with OpenCV.", video_path, error
        )
    return _sample_with_opencv(video_path, num_frames, image_size)


def load_frames_from_dir(frames_dir: Path) -> list[Image.Image]:
    """Load one extracted clip as sorted RGB PIL images."""
    frames_dir = frames_dir.expanduser().resolve()
    if not frames_dir.is_dir():
        raise FileNotFoundError(f"Frames directory does not exist: {frames_dir}")
    frame_paths = sorted(frames_dir.glob("frame_*.jpg"))
    if not frame_paths:
        raise ValueError(f"No frame_*.jpg files found in {frames_dir}")
    frames: list[Image.Image] = []
    for frame_path in frame_paths:
        with Image.open(frame_path) as image:
            frames.append(image.convert("RGB").copy())
    return frames
