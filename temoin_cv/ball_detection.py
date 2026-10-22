"""Ball detection: contours of the colour mask, centre and radius.

Builds on temoin_cv.masking for the mask itself. Every function here is pure and
holds no state, so the same logic can be ported to Swift without dragging the
capture loop along.
"""
from __future__ import annotations

import math
from typing import NamedTuple, Sequence

import cv2
import numpy as np


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
