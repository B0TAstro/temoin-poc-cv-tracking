"""Court geometry: white line detection, corner ordering and homography.

The homography produced here maps image pixels to **centimetres** on the court
plane. Keeping the real unit in the matrix itself is what makes image_to_court
reusable as is in POC 3; turning it into pixels is a display concern, handled by
scale_to_pixels.

Pure functions and one small accumulator, so the same logic ports to Swift
without the capture loop.
"""
from __future__ import annotations

import math
from collections import deque
from typing import Any, Deque, NamedTuple, Sequence

import cv2
import numpy as np

from temoin_cv.masking import odd_ksize

# Four corners, shape (4, 2), float32, ordered top-left, top-right,
# bottom-right, bottom-left.
Corners = np.ndarray

MIN_DENOMINATOR = 1e-9


def enhance_lines(mask: np.ndarray, tophat_ksize: int) -> np.ndarray:
    """Make thin bright lines stand out, for a light floor.

    A top-hat keeps what is smaller and brighter than its surroundings, which is
    exactly a strip of tape on a pale floor. Disabled when the kernel is 0 or 1,
    because on a dark floor the plain threshold is already cleaner.
    """
    kernel_size = odd_ksize(tophat_ksize)
    if tophat_ksize <= 1:
        return mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    return cv2.morphologyEx(mask, cv2.MORPH_TOPHAT, kernel)


def order_corners(points: Sequence[Sequence[float]] | np.ndarray) -> Corners:
    """Order four corners as top-left, top-right, bottom-right, bottom-left.

    Sorted by angle around the centroid, then rotated so the corner nearest the
    image origin comes first. Ordering by the sum and difference of the
    coordinates, the usual trick, picks the wrong corner as soon as the
    rectangle is seen at an angle — which is the normal case here.
    """
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if array.shape[0] != 4:
        raise ValueError(f"expected 4 corners, got {array.shape[0]}")

    centroid = array.mean(axis=0)
    # With y growing downwards, increasing atan2 walks the corners clockwise as
    # seen on screen.
    angles = np.arctan2(array[:, 1] - centroid[1], array[:, 0] - centroid[0])
    clockwise = array[np.argsort(angles)]

    distances_to_origin = np.hypot(clockwise[:, 0], clockwise[:, 1])
    start = int(np.argmin(distances_to_origin))
    return np.roll(clockwise, -start, axis=0).astype(np.float32)


def find_quad_by_contour(
    mask: np.ndarray,
    min_area_ratio: float,
    epsilon_ratio: float,
    min_support: float = 0.0,
) -> Corners | None:
    """Largest convex quadrilateral in the mask, as ordered corners.

    Contours are traced on the outside of the taped lines, so the quadrilateral
    follows the **outer** edge of the tape. Measure the real rectangle to that
    same edge, otherwise half the tape width shows up as a systematic error.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    minimum_area = min_area_ratio * mask.shape[0] * mask.shape[1]
    best: np.ndarray | None = None
    best_area = minimum_area

    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < best_area:
            continue

        perimeter = float(cv2.arcLength(contour, True))
        approximation = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
        if len(approximation) != 4 or not cv2.isContourConvex(approximation):
            continue

        best, best_area = approximation, area

    if best is None:
        return None

    corners = order_corners(best.reshape(4, 2))
    if quad_support(mask, corners) < min_support:
        return None
    return corners


def _fit_line(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares line through points, as (origin, unit direction)."""
    vx, vy, x0, y0 = cv2.fitLine(
        points.astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01
    ).ravel()
    return np.array([x0, y0], dtype=np.float64), np.array([vx, vy], dtype=np.float64)


def _intersect(
    first: tuple[np.ndarray, np.ndarray], second: tuple[np.ndarray, np.ndarray]
) -> tuple[float, float] | None:
    """Intersection of two lines given as (origin, direction)."""
    (origin_a, direction_a), (origin_b, direction_b) = first, second
    denominator = (
        direction_a[0] * direction_b[1] - direction_a[1] * direction_b[0]
    )
    if abs(denominator) < MIN_DENOMINATOR:
        return None  # parallel
    delta = origin_b - origin_a
    t = (delta[0] * direction_b[1] - delta[1] * direction_b[0]) / denominator
    return float(origin_a[0] + t * direction_a[0]), float(origin_a[1] + t * direction_a[1])


def _drop_border_segments(
    segments: np.ndarray, shape: tuple[int, int], margin: float
) -> np.ndarray:
    """Remove the segments that merely trace the edge of the frame.

    A band of tape running out of the frame is cut off by it, and that cut is a
    perfectly straight bright edge. Hough votes for it like any other line, and
    it then drags the fitted line of a real family away from the tape.
    """
    if margin <= 0:
        return segments
    height, width = shape[:2]
    xs = segments[:, [0, 2]]
    ys = segments[:, [1, 3]]
    on_edge = (
        np.all(xs < margin, axis=1)
        | np.all(xs > width - margin, axis=1)
        | np.all(ys < margin, axis=1)
        | np.all(ys > height - margin, axis=1)
    )
    return segments[~on_edge]


def _cluster_by_offset(offsets: np.ndarray, tolerance: float) -> list[np.ndarray]:
    """Group segment indices that share the same line, by their offset.

    One band of tape gives two parallel Canny edges plus any number of broken
    pieces, all at nearly the same offset; two different lines sit further
    apart than the tape is wide. Splitting wherever the sorted offsets jump by
    more than the tolerance therefore recovers one group per line, however many
    lines there are — unlike a fixed split into two, which forces a single band
    to pose as two lines.
    """
    order = np.argsort(offsets)
    cuts = np.flatnonzero(np.diff(offsets[order]) > tolerance) + 1
    return [group for group in np.split(order, cuts) if len(group) > 0]


class LineCandidate(NamedTuple):
    """One fitted line, with the total length of tape backing it."""

    origin: np.ndarray
    direction: np.ndarray
    weight: float
    horizontal: bool


def find_line_candidates(
    mask: np.ndarray, params: dict[str, Any]
) -> list[LineCandidate]:
    """Every straight line the mask supports, as fitted candidates.

    This is the honest intermediate result: a court is a network of lines, and
    which four of them bound the reference rectangle is a separate question,
    answered by searching the correspondences. Exposing the candidates also
    makes a failure legible — a frame with three lines on the floor can be told
    apart from a frame with none.
    """
    merge_tolerance = float(params.get("merge_tolerance", 60.0))
    border_margin = float(params.get("border_margin", 8.0))

    edges = cv2.Canny(mask, int(params["canny_low"]), int(params["canny_high"]))
    found = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=int(params["threshold"]),
        minLineLength=int(params["min_line_length"]),
        maxLineGap=int(params["max_line_gap"]),
    )
    if found is None:
        return []

    segments = _drop_border_segments(
        found.reshape(-1, 4).astype(np.float64), mask.shape, border_margin
    )
    if len(segments) == 0:
        return []

    midpoints = np.stack(
        [(segments[:, 0] + segments[:, 2]) / 2, (segments[:, 1] + segments[:, 3]) / 2],
        axis=1,
    )
    angles = np.degrees(
        np.arctan2(segments[:, 3] - segments[:, 1], segments[:, 2] - segments[:, 0])
    )
    horizontal = np.abs((angles + 90) % 180 - 90) < 45
    lengths = np.hypot(segments[:, 2] - segments[:, 0], segments[:, 3] - segments[:, 1])

    candidates: list[LineCandidate] = []
    for is_horizontal in (True, False):
        selected = np.flatnonzero(horizontal == is_horizontal)
        if len(selected) == 0:
            continue
        # Two parallel lines differ along the axis they do not run along.
        offsets = midpoints[selected][:, 1 if is_horizontal else 0]
        for group in _cluster_by_offset(offsets, merge_tolerance):
            rows = segments[selected[group]]
            origin, direction = _fit_line(
                np.concatenate([rows[:, :2], rows[:, 2:]])
            )
            candidates.append(
                LineCandidate(
                    origin=origin,
                    direction=direction,
                    weight=float(lengths[selected[group]].sum()),
                    horizontal=bool(is_horizontal),
                )
            )

    candidates.sort(key=lambda candidate: candidate.weight, reverse=True)
    return candidates


def _line_separation(first: LineCandidate, second: LineCandidate) -> float:
    """Perpendicular distance between two nearly parallel fitted lines."""
    normal = np.array([-first.direction[1], first.direction[0]])
    return abs(float(np.dot(second.origin - first.origin, normal)))


def _within_frame(point: tuple[float, float], shape: tuple[int, int]) -> bool:
    """Is the intersection anywhere near the picture, or off at infinity?"""
    height, width = shape[:2]
    return (
        -width <= point[0] <= 2 * width and -height <= point[1] <= 2 * height
    )


def find_quad_from_candidates(
    mask: np.ndarray,
    candidates: Sequence[LineCandidate],
    params: dict[str, Any],
    min_area_ratio: float = 0.0,
    min_support: float = 0.0,
) -> Corners | None:
    """The best-supported rectangle among already-detected candidate lines.

    Two lines are taken from each angle family and the mask decides which two,
    because four lines can always be intersected into a quadrilateral but only
    the real ones have tape underneath. Taking the candidates as an argument
    lets the caller draw them, which is the difference between "nothing is
    detected" and "only one crossing line exists, so no rectangle can close".
    """
    max_candidates = int(params.get("max_candidates", 4))
    merge_tolerance = float(params.get("merge_tolerance", 60.0))
    families = [
        [item for item in candidates if item.horizontal == is_horizontal][
            :max_candidates
        ]
        for is_horizontal in (True, False)
    ]
    if any(len(family) < 2 for family in families):
        return None

    # Dilate once rather than inside every scoring call: the search below asks
    # the same question of the same mask a few hundred times.
    tolerance = 3
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
    )
    dilated = cv2.dilate(mask, kernel)
    minimum_area = min_area_ratio * mask.shape[0] * mask.shape[1]

    best: Corners | None = None
    best_support = min_support

    for first_h in range(len(families[0])):
        for second_h in range(first_h + 1, len(families[0])):
            # Two lines closer together than the tape is wide are the two edges
            # of one band, and the sliver between them scores a perfect support
            # while meaning nothing. Same threshold as the merge: it is the
            # width of the widest band.
            if (
                _line_separation(families[0][first_h], families[0][second_h])
                <= merge_tolerance
            ):
                continue
            for first_v in range(len(families[1])):
                for second_v in range(first_v + 1, len(families[1])):
                    if (
                        _line_separation(
                            families[1][first_v], families[1][second_v]
                        )
                        <= merge_tolerance
                    ):
                        continue
                    corners = [
                        _intersect(
                            (families[0][h].origin, families[0][h].direction),
                            (families[1][v].origin, families[1][v].direction),
                        )
                        for h in (first_h, second_h)
                        for v in (first_v, second_v)
                    ]
                    if any(corner is None for corner in corners):
                        continue
                    if not all(_within_frame(corner, mask.shape) for corner in corners):
                        continue
                    ordered = order_corners(corners)
                    area = float(cv2.contourArea(ordered.reshape(-1, 1, 2)))
                    if area < minimum_area:
                        continue
                    support = quad_support(dilated, ordered, tolerance=0)
                    if support > best_support:
                        best, best_support = ordered, support

    return best


def find_quad_by_hough(
    mask: np.ndarray,
    params: dict[str, Any],
    min_area_ratio: float = 0.0,
    min_support: float = 0.0,
) -> Corners | None:
    """Fallback detection: candidate lines, then the best-supported rectangle.

    Used when the lines are broken or open, which is the normal case: a court
    is a ladder of separate bands, never a closed outline, so no contour ever
    traces it.
    """
    return find_quad_from_candidates(
        mask,
        find_line_candidates(mask, params),
        params,
        min_area_ratio,
        min_support,
    )


def clip_line_to_frame(
    candidate: LineCandidate, shape: tuple[int, int]
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    """The chord a fitted line cuts across the frame, for drawing it."""
    height, width = shape[:2]
    origin, direction = candidate.origin, candidate.direction
    distances: list[float] = []
    for axis, limit in ((0, width - 1), (1, height - 1)):
        if abs(direction[axis]) < MIN_DENOMINATOR:
            continue
        for bound in (0.0, float(limit)):
            distances.append((bound - origin[axis]) / direction[axis])
    inside = []
    for distance in distances:
        point = origin + distance * direction
        if -1 <= point[0] <= width and -1 <= point[1] <= height:
            inside.append((float(point[0]), float(point[1])))
    if len(inside) < 2:
        return None
    first, last = inside[0], max(
        inside, key=lambda point: (point[0] - inside[0][0]) ** 2 + (point[1] - inside[0][1]) ** 2
    )
    return first, last


def segment_support(
    mask: np.ndarray,
    start: Sequence[float],
    end: Sequence[float],
    samples: int = 64,
) -> float:
    """Share of a segment that actually lies on mask pixels, from 0 to 1."""
    xs = np.linspace(float(start[0]), float(end[0]), samples)
    ys = np.linspace(float(start[1]), float(end[1]), samples)
    height, width = mask.shape[:2]
    columns = np.clip(np.round(xs).astype(int), 0, width - 1)
    rows = np.clip(np.round(ys).astype(int), 0, height - 1)
    return float(np.count_nonzero(mask[rows, columns]) / samples)


def quad_support(mask: np.ndarray, corners: Corners, tolerance: int = 3) -> float:
    """Support of the WEAKEST of the four edges.

    Four lines can always be intersected into a quadrilateral, so a detector
    that only checks shape will happily report one made of unrelated edges —
    a door frame and a skirting board will do. This asks the only question
    that matters: is each of those four edges actually drawn on the floor?
    The weakest edge decides, because three good edges and one invented one
    is still an invented rectangle.
    """
    if tolerance > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * tolerance + 1, 2 * tolerance + 1)
        )
        mask = cv2.dilate(mask, kernel)

    points = np.asarray(corners, dtype=float)
    return min(
        segment_support(mask, points[index], points[(index + 1) % 4])
        for index in range(4)
    )


def court_corners_cm(width_cm: float, height_cm: float) -> Corners:
    """The real rectangle's corners in centimetres, in the same order."""
    return np.array(
        [[0.0, 0.0], [width_cm, 0.0], [width_cm, height_cm], [0.0, height_cm]],
        dtype=np.float32,
    )


def compute_homography(image_corners: Corners, court_corners: Corners) -> np.ndarray:
    """Matrix mapping image pixels to court centimetres.

    getPerspectiveTransform rather than findHomography: with exactly four
    correspondences the fit is exact, and there is nothing for a least-squares
    solver to improve. It also means reprojection error is always about zero
    here and says nothing about quality — watch corner stability instead.
    """
    return cv2.getPerspectiveTransform(
        image_corners.astype(np.float32), court_corners.astype(np.float32)
    )


def scale_to_pixels(matrix: np.ndarray, px_per_cm: float) -> np.ndarray:
    """Turn a centimetre homography into one for warpPerspective, in pixels."""
    scale = np.array(
        [[px_per_cm, 0.0, 0.0], [0.0, px_per_cm, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return scale @ matrix


def _project(point: Sequence[float], matrix: np.ndarray) -> tuple[float, float]:
    source = np.array([[[float(point[0]), float(point[1])]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(source, matrix.astype(np.float32))
    return float(projected[0, 0, 0]), float(projected[0, 0, 1])


def image_to_court(point: Sequence[float], matrix: np.ndarray) -> tuple[float, float]:
    """Project an image point onto the court plane, in centimetres.

    The projection assumes the point lies **on** the plane. Anything with height
    lands off target by its own parallax, which is why a validation marker must
    be flat, and why POC 3 has to deal with the ball's centre sitting one radius
    above the ground at the bounce.
    """
    return _project(point, matrix)


def court_to_image(point: Sequence[float], matrix: np.ndarray) -> tuple[float, float]:
    """Project a court point in centimetres back into the image."""
    return _project(point, np.linalg.inv(matrix))


def warp_top_down(
    frame: np.ndarray,
    matrix: np.ndarray,
    width_cm: float,
    height_cm: float,
    px_per_cm: float,
) -> np.ndarray:
    """Rectified top-down view of the court plane."""
    size = (
        max(1, int(round(width_cm * px_per_cm))),
        max(1, int(round(height_cm * px_per_cm))),
    )
    return cv2.warpPerspective(frame, scale_to_pixels(matrix, px_per_cm), size)


class CornerSmoother:
    """Rolling mean of the corners, with the jitter it smooths out.

    The jitter is the quality indicator for this POC: reprojection error cannot
    be one, and a corner that dances by several pixels between frames is what
    will turn into centimetres of doubt in POC 3.
    """

    def __init__(self, window: int) -> None:
        self._history: Deque[np.ndarray] = deque(maxlen=max(1, int(window)))

    def update(self, corners: Corners) -> Corners:
        self._history.append(np.asarray(corners, dtype=np.float32))
        return np.mean(np.stack(self._history), axis=0).astype(np.float32)

    def reset(self) -> None:
        self._history.clear()

    @property
    def jitter(self) -> float:
        """Mean RMS displacement of a corner over the window, in pixels."""
        if len(self._history) < 2:
            return 0.0
        stack = np.stack(self._history)
        variance = stack.var(axis=0)
        return float(np.mean(np.sqrt(variance[:, 0] + variance[:, 1])))
