"""HSV masking and downscaling, shared by every POC.

The ball is isolated by its hue; the court lines are isolated as white, which is
an ordinary HSV range with a low saturation ceiling. Both go through the same
mask builder, which is also why one calibration tool serves both configs.

Pure functions only, so the same logic ports to Swift without the capture loop.
"""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

HsvRange = tuple[np.ndarray, np.ndarray]


def odd_ksize(size: int) -> int:
    """Kernel sizes must be odd and at least 1."""
    size = max(1, int(size))
    return size if size % 2 == 1 else size + 1


def resize_for_processing(
    frame: np.ndarray, processing_width: int
) -> tuple[np.ndarray, float]:
    """Downscale the frame for detection and return the scale factor applied.

    Detection runs on a smaller image to hold the frame rate; coordinates are
    divided by this factor to come back to the display resolution.
    """
    height, width = frame.shape[:2]
    if processing_width <= 0 or processing_width >= width:
        return frame, 1.0
    scale = processing_width / width
    new_size = (processing_width, max(1, int(round(height * scale))))
    return cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA), scale


def build_mask(
    frame_bgr: np.ndarray,
    ranges: Sequence[HsvRange],
    blur_ksize: int,
    open_ksize: int,
    close_ksize: int,
) -> np.ndarray:
    """Blur, convert to HSV, threshold each range, then clean up the mask.

    Thresholding happens in HSV and never in BGR: hue stays usable when the
    room light changes, a raw RGB threshold does not.
    """
    blur = odd_ksize(blur_ksize)
    if blur > 1:
        frame_bgr = cv2.GaussianBlur(frame_bgr, (blur, blur), 0)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper), dst=mask)

    # Opening removes isolated speckles, closing fills the holes left by a
    # printed pattern, a specular highlight or a gap in a taped line.
    open_k = odd_ksize(open_ksize)
    if open_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    close_k = odd_ksize(close_ksize)
    if close_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask
