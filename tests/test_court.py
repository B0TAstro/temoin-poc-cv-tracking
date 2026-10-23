"""Tests for the court geometry: corner ordering, homography, detection.

A wrong corner order or an inverted axis produces a top-down view that looks
plausible and is mirrored, which is exactly the kind of mistake the eye waves
through. These functions are also the ones POC 3 will reuse.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from temoin_cv.court import (
    CornerSmoother,
    compute_homography,
    court_corners_cm,
    court_to_image,
    enhance_lines,
    clip_line_to_frame,
    find_line_candidates,
    find_quad_by_contour,
    find_quad_by_hough,
    image_to_court,
    order_corners,
    quad_support,
    scale_to_pixels,
    segment_support,
    warp_top_down,
)

# A rectangle on the floor seen from the side: it reaches the image as a
# trapezoid, with the far edge shorter than the near one.
TRAPEZOID = [(220.0, 150.0), (420.0, 150.0), (520.0, 330.0), (120.0, 330.0)]

HOUGH_PARAMS = {
    "canny_low": 50,
    "canny_high": 150,
    "threshold": 40,
    "min_line_length": 40,
    "max_line_gap": 30,
}


def white_quad_mask(
    corners=TRAPEZOID, size=(480, 640), thickness: int = 6, gaps: bool = False
) -> np.ndarray:
    """A taped rectangle as the white mask sees it: a hollow quadrilateral."""
    mask = np.zeros(size, dtype=np.uint8)
    points = np.array(corners, dtype=np.int32)
    if not gaps:
        cv2.polylines(mask, [points], True, 255, thickness)
        return mask
    # Worn lines: each edge drawn as two segments with a hole in the middle.
    for index in range(4):
        start = points[index].astype(float)
        end = points[(index + 1) % 4].astype(float)
        for lo, hi in ((0.0, 0.38), (0.62, 1.0)):
            cv2.line(
                mask,
                tuple((start + (end - start) * lo).astype(int)),
                tuple((start + (end - start) * hi).astype(int)),
                255,
                thickness,
            )
    return mask


# -- corner ordering -------------------------------------------------------


def test_order_corners_is_stable_under_any_permutation():
    reference = order_corners(TRAPEZOID)
    rng = np.random.default_rng(0)
    for _ in range(10):
        shuffled = [TRAPEZOID[i] for i in rng.permutation(4)]
        assert order_corners(shuffled) == pytest.approx(reference)


def test_order_corners_starts_top_left_and_runs_clockwise():
    ordered = order_corners(TRAPEZOID)
    top_left, top_right, bottom_right, bottom_left = ordered

    assert top_left == pytest.approx((220.0, 150.0))
    assert top_right == pytest.approx((420.0, 150.0))
    assert bottom_right == pytest.approx((520.0, 330.0))
    assert bottom_left == pytest.approx((120.0, 330.0))


def test_order_corners_rejects_the_wrong_count():
    with pytest.raises(ValueError):
        order_corners([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])


# -- homography -------------------------------------------------------------


def test_homography_maps_the_corners_onto_the_real_rectangle():
    court = court_corners_cm(200.0, 100.0)
    matrix = compute_homography(order_corners(TRAPEZOID), court)

    for image_corner, expected in zip(order_corners(TRAPEZOID), court):
        assert image_to_court(image_corner, matrix) == pytest.approx(
            tuple(expected), abs=0.01
        )


def test_image_to_court_and_back_is_a_round_trip():
    matrix = compute_homography(order_corners(TRAPEZOID), court_corners_cm(200.0, 100.0))

    for point in [(300.0, 200.0), (250.0, 300.0), (480.0, 320.0)]:
        in_centimetres = image_to_court(point, matrix)
        assert court_to_image(in_centimetres, matrix) == pytest.approx(point, abs=0.05)


def test_a_point_inside_the_quad_lands_inside_the_court():
    matrix = compute_homography(order_corners(TRAPEZOID), court_corners_cm(200.0, 100.0))
    centroid = np.mean(np.array(TRAPEZOID), axis=0)

    x_cm, y_cm = image_to_court(centroid, matrix)
    assert 0.0 < x_cm < 200.0
    assert 0.0 < y_cm < 100.0

    # Across the trapezoid the view is symmetric, so the centroid is halfway.
    assert x_cm == pytest.approx(100.0, abs=1.0)
    # Along it, it is not: perspective squeezes the far half of the rectangle
    # into fewer pixels, so the middle of the image is well past the middle of
    # the court. Getting this backwards yields a plausible, wrong top-down view.
    assert y_cm == pytest.approx(66.7, abs=1.0)


def test_scale_to_pixels_scales_the_centimetre_mapping():
    matrix = compute_homography(order_corners(TRAPEZOID), court_corners_cm(200.0, 100.0))
    in_pixels = scale_to_pixels(matrix, 2.0)

    corner = order_corners(TRAPEZOID)[2]  # bottom-right, at (200, 100) cm
    projected = cv2.perspectiveTransform(
        np.array([[[corner[0], corner[1]]]], dtype=np.float32), in_pixels.astype(np.float32)
    )
    assert projected[0, 0] == pytest.approx((400.0, 200.0), abs=0.05)


def test_warp_top_down_has_the_configured_size():
    matrix = compute_homography(order_corners(TRAPEZOID), court_corners_cm(200.0, 100.0))
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    warped = warp_top_down(frame, matrix, 200.0, 100.0, 2.0)
    assert warped.shape[:2] == (200, 400)


# -- detection ---------------------------------------------------------------


def test_find_quad_by_contour_finds_the_taped_rectangle():
    found = find_quad_by_contour(white_quad_mask(), min_area_ratio=0.02, epsilon_ratio=0.02)

    assert found is not None
    # The contour runs along the outer edge of the tape, so each corner sits
    # about half a line width outside the centreline.
    assert found == pytest.approx(np.array(order_corners(TRAPEZOID)), abs=6.0)


def test_find_quad_by_contour_ignores_a_shape_that_is_too_small():
    small = [(10.0, 10.0), (40.0, 10.0), (40.0, 40.0), (10.0, 40.0)]
    assert (
        find_quad_by_contour(
            white_quad_mask(small), min_area_ratio=0.02, epsilon_ratio=0.02
        )
        is None
    )


def test_find_quad_by_contour_returns_nothing_on_an_empty_mask():
    empty = np.zeros((480, 640), dtype=np.uint8)
    assert find_quad_by_contour(empty, 0.02, 0.02) is None


def test_find_quad_by_hough_recovers_broken_lines():
    """The fallback exists for worn lines the contour can no longer close."""
    broken = white_quad_mask(gaps=True)
    assert find_quad_by_contour(broken, 0.02, 0.02) is None

    found = find_quad_by_hough(broken, HOUGH_PARAMS, min_area_ratio=0.02)
    assert found is not None
    assert found == pytest.approx(np.array(order_corners(TRAPEZOID)), abs=8.0)


def test_find_quad_by_hough_returns_nothing_without_enough_lines():
    nearly_empty = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(nearly_empty, (100, 100), (400, 100), 255, 5)
    assert find_quad_by_hough(nearly_empty, HOUGH_PARAMS) is None


def test_enhance_lines_is_a_no_op_when_disabled():
    mask = white_quad_mask()
    assert np.array_equal(enhance_lines(mask, 0), mask)
    assert np.array_equal(enhance_lines(mask, 1), mask)


def test_enhance_lines_keeps_thin_lines_and_drops_a_broad_area():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (10, 100), (190, 100), 255, 3)  # a taped line
    cv2.rectangle(mask, (20, 140), (180, 190), 255, -1)  # a broad pale patch

    enhanced = enhance_lines(mask, 15)
    assert enhanced[100, 100] == 255  # the thin line survives
    assert enhanced[165, 100] == 0  # the broad patch is removed


# -- smoothing -----------------------------------------------------------------


def test_corner_smoother_averages_and_reports_no_jitter_when_still():
    smoother = CornerSmoother(window=5)
    corners = order_corners(TRAPEZOID)

    for _ in range(5):
        smoothed = smoother.update(corners)

    assert smoothed == pytest.approx(corners)
    assert smoother.jitter == 0.0


def test_corner_smoother_reports_jitter_when_corners_dance():
    smoother = CornerSmoother(window=4)
    base = np.array(TRAPEZOID, dtype=np.float32)

    for offset in (-3.0, 3.0, -3.0, 3.0):
        smoother.update(base + offset)

    assert smoother.jitter == pytest.approx(math_hypot_of(3.0), abs=0.2)


def math_hypot_of(amplitude: float) -> float:
    """RMS displacement of a point oscillating by +/- amplitude on both axes."""
    return float(np.sqrt(amplitude**2 + amplitude**2))


def test_corner_smoother_reset_clears_the_window():
    smoother = CornerSmoother(window=3)
    smoother.update(order_corners(TRAPEZOID))
    smoother.reset()
    assert smoother.jitter == 0.0


# -- mask support --------------------------------------------------------------


def test_segment_support_is_one_on_the_line_and_zero_off_it():
    mask = np.zeros((200, 200), dtype=np.uint8)
    cv2.line(mask, (10, 100), (190, 100), 255, 5)

    assert segment_support(mask, (10, 100), (190, 100)) == pytest.approx(1.0)
    assert segment_support(mask, (10, 30), (190, 30)) == pytest.approx(0.0)


def test_quad_support_is_high_on_the_taped_rectangle():
    assert quad_support(white_quad_mask(), order_corners(TRAPEZOID)) > 0.9


def test_quad_support_is_zero_on_an_unrelated_rectangle():
    """Four unrelated lines always intersect into a plausible quadrilateral."""
    elsewhere = [(20.0, 20.0), (120.0, 20.0), (120.0, 90.0), (20.0, 90.0)]
    assert quad_support(white_quad_mask(), order_corners(elsewhere)) == pytest.approx(0.0)


def test_quad_support_is_decided_by_the_weakest_edge():
    """Three real edges and one invented one is still an invented rectangle."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    ordered = order_corners(TRAPEZOID)
    corners = np.array(ordered, dtype=np.int32)
    for index in range(3):  # three of the four edges only
        cv2.line(mask, tuple(corners[index]), tuple(corners[index + 1]), 255, 6)

    # The three drawn edges are fully backed...
    for index in range(3):
        assert segment_support(mask, ordered[index], ordered[index + 1]) == pytest.approx(1.0)
    # ...and the one that was never drawn drags the whole quad down. It is not
    # exactly zero because the tolerance dilation catches its two endpoints,
    # which sit on the edges that were drawn.
    assert quad_support(mask, ordered) < 0.2


def test_the_support_gate_rejects_a_quad_the_mask_does_not_back():
    """The regression that mattered: a confident wrong answer in a white room.

    A colour threshold cannot tell a white wall from white tape, so the mask
    arrives with the walls in it and Hough intersects wall edges with tape edges
    into a rectangle that looks entirely plausible. Shape alone cannot reject
    it; mask support can.
    """
    mask = white_quad_mask()
    mask[:150, :] = 255  # a lit wall across the top
    mask[:, 500:] = 255  # a bright door down the right side

    unguarded = find_quad_by_hough(mask, HOUGH_PARAMS, 0.02, min_support=0.0)
    assert unguarded is not None
    assert quad_support(mask, unguarded) < 0.5  # part tape, part wall

    assert find_quad_by_hough(mask, HOUGH_PARAMS, 0.02, min_support=0.6) is None


# -- a ladder of tape, which is what a court actually is -----------------------

# Two long bands running past two crossing bands: nothing closes, so no contour
# traces it. The crossings sit at a fifth and four fifths along the long bands.
LADDER_LEFT = ((240.0, 120.0), (80.0, 400.0))
LADDER_RIGHT = ((400.0, 120.0), (560.0, 400.0))


def _along(band, t):
    (x0, y0), (x1, y1) = band
    return (x0 + (x1 - x0) * t, y0 + (y1 - y0) * t)


LADDER_CORNERS = [
    _along(LADDER_LEFT, 0.2),
    _along(LADDER_RIGHT, 0.2),
    _along(LADDER_RIGHT, 0.8),
    _along(LADDER_LEFT, 0.8),
]


def ladder_mask(crossings: int = 2, thickness: int = 8) -> np.ndarray:
    """Two sidelines plus `crossings` crossing bands, as the mask sees them."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    for band in (LADDER_LEFT, LADDER_RIGHT):
        cv2.line(
            mask,
            tuple(int(v) for v in band[0]),
            tuple(int(v) for v in band[1]),
            255,
            thickness,
        )
    for index in range(crossings):
        t = 0.8 if index == 0 else 0.2
        cv2.line(
            mask,
            tuple(int(v) for v in _along(LADDER_LEFT, t)),
            tuple(int(v) for v in _along(LADDER_RIGHT, t)),
            255,
            thickness,
        )
    return mask


def test_a_ladder_of_tape_is_found_although_no_contour_closes():
    """The real shape of a court: open bands, never a closed outline."""
    mask = ladder_mask()
    assert find_quad_by_contour(mask, 0.02, 0.02) is None

    found = find_quad_by_hough(mask, HOUGH_PARAMS, min_area_ratio=0.02)
    assert found is not None
    assert found == pytest.approx(np.array(order_corners(LADDER_CORNERS)), abs=8.0)


def test_one_crossing_band_cannot_close_a_rectangle():
    """Four points need two lines each way. One crossing is not enough, and
    saying so beats inventing the sliver between that band's two edges."""
    mask = ladder_mask(crossings=1)
    across = [item for item in find_line_candidates(mask, HOUGH_PARAMS) if item.horizontal]
    assert len(across) == 1
    assert find_quad_by_hough(mask, HOUGH_PARAMS, min_area_ratio=0.02) is None


def test_the_two_edges_of_one_band_are_one_line():
    """A band of tape is one line, not the pair of its Canny edges.

    Treating the edges as two lines makes the sliver between them a perfect
    rectangle: support 1.00, and corners a third of the frame away from the
    truth.
    """
    mask = np.zeros((480, 640), dtype=np.uint8)
    cv2.line(mask, (60, 240), (580, 240), 255, 30)

    candidates = find_line_candidates(mask, HOUGH_PARAMS)
    assert len(candidates) == 1
    assert candidates[0].horizontal
    assert candidates[0].origin[1] == pytest.approx(240.0, abs=3.0)


def test_segments_tracing_the_frame_edge_are_dropped():
    """A band running out of frame is cut by it, and the cut is a bright
    straight edge that Hough votes for like any other line."""
    mask = ladder_mask()
    mask[:, :6] = 255  # the frame's own left edge, lit

    candidates = find_line_candidates(mask, HOUGH_PARAMS)
    assert all(
        min(item.origin[0], abs(item.origin[0] - mask.shape[1])) > 10
        or item.horizontal
        for item in candidates
    )


def test_clip_line_to_frame_spans_the_picture():
    candidate = find_line_candidates(ladder_mask(), HOUGH_PARAMS)[0]
    chord = clip_line_to_frame(candidate, (480, 640))
    assert chord is not None
    start, end = chord
    assert np.hypot(end[0] - start[0], end[1] - start[1]) > 100
