"""Frame source: webcam or recorded file behind one interface.

Both must be interchangeable so the same sequence can be replayed while the
detection thresholds change, instead of comparing two different throws.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_FILE_FPS = 30.0


class VideoSource:
    """A webcam or a video file, read frame by frame with a timestamp."""

    def __init__(
        self,
        source: str | int,
        width: int | None = None,
        height: int | None = None,
        fps: float | None = None,
        warmup_frames: int = 0,
    ) -> None:
        self.is_file = isinstance(source, str)

        if self.is_file:
            self._capture = cv2.VideoCapture(source)
        else:
            # AVFoundation is the native macOS backend; the default one opens
            # the camera far more slowly.
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            self._capture = cv2.VideoCapture(source, backend)

        if not self._capture.isOpened():
            # User-facing message, hence French.
            raise RuntimeError(
                f"Impossible d'ouvrir la source vidéo {source!r}. "
                "Sur macOS, vérifier Réglages Système > Confidentialité et "
                "sécurité > Caméra."
            )

        if not self.is_file:
            if width:
                self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            if height:
                self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            if fps:
                self._capture.set(cv2.CAP_PROP_FPS, fps)

        self.width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        reported_fps = self._capture.get(cv2.CAP_PROP_FPS)
        self.fps = (
            float(reported_fps)
            if reported_fps and reported_fps > 0
            else float(fps or DEFAULT_FILE_FPS)
        )

        self._frame_index = 0
        self._start_time = time.perf_counter()

        if not self.is_file:
            self._warmup(warmup_frames)

    @classmethod
    def from_config(
        cls, config: dict[str, Any], source: str | None = None
    ) -> "VideoSource":
        """Build from the `capture` section; `source` None means the webcam."""
        capture_config = config["capture"]
        return cls(
            source=source if source is not None else int(capture_config["device_index"]),
            width=capture_config.get("width"),
            height=capture_config.get("height"),
            fps=capture_config.get("fps"),
            warmup_frames=int(capture_config.get("warmup_frames", 0)),
        )

    def _warmup(self, frames: int) -> None:
        """Drop the first frames: the Mac webcam returns black ones on open."""
        for _ in range(max(0, frames)):
            self._capture.read()
        self._start_time = time.perf_counter()

    def read(self) -> tuple[bool, np.ndarray | None, float]:
        """Return (ok, frame, timestamp_seconds).

        The timestamp is wall clock for the webcam and frame index over file
        rate for a file, so velocities in pixels per second stay correct even
        when a replay does not run at capture speed.
        """
        ok, frame = self._capture.read()
        if not ok or frame is None:
            return False, None, self.timestamp()

        timestamp = self.timestamp()
        self._frame_index += 1
        return True, frame, timestamp

    def timestamp(self) -> float:
        if self.is_file:
            return self._frame_index / self.fps
        return time.perf_counter() - self._start_time

    def release(self) -> None:
        self._capture.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()


class FrameRecorder:
    """Writes frames to disk so a run can be replayed later, unchanged."""

    def __init__(
        self, path: str | Path, fps: float, size: tuple[int, int], fourcc: str = "mp4v"
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = cv2.VideoWriter(
            str(self.path), cv2.VideoWriter.fourcc(*fourcc), fps, size
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Impossible d'écrire dans {self.path}")

    def write(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def release(self) -> None:
        self._writer.release()

    def __enter__(self) -> "FrameRecorder":
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
