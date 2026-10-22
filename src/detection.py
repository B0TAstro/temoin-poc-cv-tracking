"""Color-based ball detection: HSV mask, contour filtering, candidate scoring.

Every function here is pure and holds no state, so the same logic can be
ported to Swift later without dragging the capture loop along.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import cv2
import numpy as np

HsvRange = tuple[np.ndarray, np.ndarray]


class Candidate(NamedTuple):
    """One blob that could be the ball, in processing-resolution pixels."""

    center: tuple[float, float]
    radius: float
    area: float
    circularity: float


class ScoreWeights(NamedTuple):
    area: float
    circularity: float
    distance: float


def _odd(size: int) -> int:
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
    blur = _odd(blur_ksize)
    if blur > 1:
        frame_bgr = cv2.GaussianBlur(frame_bgr, (blur, blur), 0)

    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

    mask = np.zeros(frame_bgr.shape[:2], dtype=np.uint8)
    for lower, upper in ranges:
        cv2.bitwise_or(mask, cv2.inRange(hsv, lower, upper), dst=mask)

    # Opening removes isolated speckles, closing fills the holes left by the
    # ball's printed pattern and by specular highlights.
    open_k = _odd(open_ksize)
    if open_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    close_k = _odd(close_ksize)
    if close_k > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def circularity(area: float, perimeter: float) -> float:
    """How close a contour is to a disc: 1.0 for a perfect circle, 0 for a line.

    Clamped to 1.0 because a rasterised circle can score slightly above it.
    """
    if perimeter <= 0.0:
        return 0.0
    return min(1.0, 4.0 * math.pi * area / (perimeter * perimeter))


def find_candidates(
    mask: np.ndarray,
    min_area: float,
    max_area: float,
    min_circularity: float,
) -> list[Candidate]:
    """Extract every blob of the mask that could plausibly be the ball."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates: list[Candidate] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < min_area or area > max_area:
            continue

        shape = circularity(area, float(cv2.arcLength(contour, True)))
        if shape < min_circularity:
            continue

        (x, y), radius = cv2.minEnclosingCircle(contour)
        candidates.append(Candidate((float(x), float(y)), float(radius), area, shape))

    return candidates


def select_best(
    candidates: Sequence[Candidate],
    predicted_center: tuple[float, float] | None,
    weights: ScoreWeights,
    frame_diagonal: float,
) -> Candidate | None:
    """Pick the most likely ball among the candidates.

    The score mixes relative area, circularity and closeness to where the
    tracker expects the ball to be. Without a prediction (first frame, or the
    ball was lost) the distance term is neutral for everyone.
    """
    if not candidates:
        return None

    largest_area = max(candidate.area for candidate in candidates)

    best: Candidate | None = None
    best_score = -math.inf
    for candidate in candidates:
        area_score = candidate.area / largest_area if largest_area > 0 else 0.0

        if predicted_center is None or frame_diagonal <= 0:
            distance_score = 1.0
        else:
            distance = math.dist(candidate.center, predicted_center)
            distance_score = max(0.0, 1.0 - distance / frame_diagonal)

        score = (
            weights.area * area_score
            + weights.circularity * candidate.circularity
            + weights.distance * distance_score
        )
        if score > best_score:
            best, best_score = candidate, score

    return best


def scale_candidate(candidate: Candidate, scale: float) -> Candidate:
    """Bring a candidate found on the downscaled frame back to display pixels."""
    if scale <= 0 or scale == 1.0:
        return candidate
    x, y = candidate.center
    return Candidate(
        center=(x / scale, y / scale),
        radius=candidate.radius / scale,
        area=candidate.area / (scale * scale),
        circularity=candidate.circularity,
    )
