"""Tests for the pure geometry and derivative functions.

These are the ones that will be ported to Swift, and the ones where a sign
error or an off-by-one goes unnoticed on screen.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytest

from temoin_cv.ball_detection import (
    Candidate,
    ScoreWeights,
    circularity,
    find_candidates,
    scale_candidate,
    select_best,
)
from temoin_cv.masking import build_mask, resize_for_processing
from temoin_cv.tracking import (
    BallTracker,
    TrackerState,
    finite_difference,
    linear_slope,
)

# Matches the blue ball of configs/ball.yaml; BGR (190, 150, 70) lands on H=100.
BLUE_RANGE = [
    (np.array([85, 60, 40], dtype=np.uint8), np.array([115, 255, 255], dtype=np.uint8))
]
BALL_BGR = (190, 150, 70)


def blue_disc_on_white(
    size: tuple[int, int] = (480, 640), center: tuple[int, int] = (320, 240), radius: int = 40
) -> np.ndarray:
    """The test-room setup: the blue ball against a white classroom wall."""
    frame = np.full((size[0], size[1], 3), 255, dtype=np.uint8)
    cv2.circle(frame, center, radius, BALL_BGR, -1)
    return frame


# -- circularity ---------------------------------------------------------------


def test_circularity_of_a_perfect_circle_is_one():
    radius = 10.0
    area = math.pi * radius**2
    perimeter = 2 * math.pi * radius
    assert circularity(area, perimeter) == pytest.approx(1.0)


def test_circularity_of_a_square_is_below_a_circle():
    side = 10.0
    assert circularity(side**2, 4 * side) < 0.8


def test_circularity_is_zero_without_a_perimeter():
    assert circularity(100.0, 0.0) == 0.0


# -- derivatives ----------------------------------------------------------------


def test_linear_slope_recovers_a_constant_rate():
    times = [0.0, 0.1, 0.2, 0.3]
    values = [0.0, 5.0, 10.0, 15.0]
    assert linear_slope(times, values) == pytest.approx(50.0)


def test_linear_slope_needs_two_points():
    assert linear_slope([1.0], [3.0]) == 0.0
    assert linear_slope([], []) == 0.0


def test_linear_slope_ignores_jitter_around_a_trend():
    times = [i / 30 for i in range(10)]
    clean = [100.0 * t for t in times]
    noisy = [value + (1.0 if index % 2 else -1.0) for index, value in enumerate(clean)]
    assert linear_slope(times, noisy) == pytest.approx(100.0, abs=5.0)


def test_finite_difference_returns_both_axes_signed():
    # Moving right and upwards: y decreases in image coordinates.
    samples = [(i / 30, 10.0 * i, -20.0 * i) for i in range(6)]
    vx, vy = finite_difference(samples, window=5)
    assert vx == pytest.approx(300.0)
    assert vy == pytest.approx(-600.0)


def test_finite_difference_is_flat_without_history():
    assert finite_difference([], window=5) == (0.0, 0.0)
    assert finite_difference([(0.0, 1.0, 2.0)], window=5) == (0.0, 0.0)


# -- candidate selection ---------------------------------------------------------


def test_select_best_returns_nothing_without_candidates():
    assert select_best([], None, ScoreWeights(0.3, 0.3, 0.4), 800.0) is None


def test_select_best_prefers_the_candidate_near_the_prediction():
    near = Candidate(center=(105.0, 100.0), radius=10.0, area=300.0, circularity=0.9)
    far = Candidate(center=(600.0, 400.0), radius=10.0, area=300.0, circularity=0.9)
    best = select_best([far, near], (100.0, 100.0), ScoreWeights(0.3, 0.3, 0.4), 800.0)
    assert best is near


def test_select_best_falls_back_to_area_without_a_prediction():
    small = Candidate(center=(10.0, 10.0), radius=5.0, area=100.0, circularity=0.9)
    big = Candidate(center=(300.0, 300.0), radius=20.0, area=1200.0, circularity=0.9)
    best = select_best([small, big], None, ScoreWeights(0.3, 0.3, 0.4), 800.0)
    assert best is big


# -- mask and scaling -------------------------------------------------------------


def test_build_mask_isolates_the_ball_and_ignores_the_white_background():
    frame = blue_disc_on_white()
    mask = build_mask(frame, BLUE_RANGE, 5, 5, 9)

    assert mask[240, 320] == 255  # the ball
    assert mask[10, 10] == 0  # the white wall
    # The mask covers roughly the disc and nothing else.
    assert mask.sum() / 255 == pytest.approx(math.pi * 40**2, rel=0.1)


def test_find_candidates_locates_the_disc():
    mask = build_mask(blue_disc_on_white(), BLUE_RANGE, 5, 5, 9)
    candidates = find_candidates(mask, min_area=150, max_area=120_000, min_circularity=0.6)

    assert len(candidates) == 1
    assert candidates[0].center == pytest.approx((320.0, 240.0), abs=2.0)
    assert candidates[0].radius == pytest.approx(40.0, abs=2.0)


def test_find_candidates_rejects_a_blob_that_is_too_small():
    mask = build_mask(
        blue_disc_on_white(radius=5), BLUE_RANGE, 5, 5, 9
    )
    assert find_candidates(mask, min_area=150, max_area=120_000, min_circularity=0.6) == []


def test_scale_candidate_undoes_the_processing_downscale():
    frame = blue_disc_on_white()
    small, scale = resize_for_processing(frame, 320)
    assert scale == pytest.approx(0.5)

    found = find_candidates(
        build_mask(small, BLUE_RANGE, 5, 5, 9),
        min_area=50,
        max_area=120_000,
        min_circularity=0.6,
    )[0]
    restored = scale_candidate(found, scale)

    assert restored.center == pytest.approx((320.0, 240.0), abs=4.0)
    assert restored.radius == pytest.approx(40.0, abs=4.0)


def test_resize_for_processing_leaves_small_frames_alone():
    frame = blue_disc_on_white(size=(240, 320))
    resized, scale = resize_for_processing(frame, 640)
    assert scale == 1.0
    assert resized is frame


# -- tracker lifecycle --------------------------------------------------------------


def test_tracker_coasts_on_prediction_then_gives_up():
    tracker = BallTracker(
        process_noise=0.03,
        measurement_noise=0.1,
        max_missed_frames=3,
        history_length=64,
        derivative_window=5,
    )

    for index in range(10):
        timestamp = index / 30
        assert (
            tracker.update((100 * timestamp, 50 * timestamp), timestamp, 10.0)
            is TrackerState.DETECTED
        )

    states = [tracker.update(None, (10 + index) / 30) for index in range(4)]
    assert states == [
        TrackerState.PREDICTED,
        TrackerState.PREDICTED,
        TrackerState.PREDICTED,
        TrackerState.LOST,
    ]
    assert tracker.position is None
    assert len(tracker.history) == 0


def test_tracker_velocity_converges_on_a_constant_motion():
    tracker = BallTracker(0.03, 0.1, 8, 64, 5)
    for index in range(40):
        timestamp = index / 30
        tracker.update((100 * timestamp, 50 * timestamp), timestamp, 10.0)

    vx, vy = tracker.velocity()
    assert vx == pytest.approx(100.0, rel=0.1)
    assert vy == pytest.approx(50.0, rel=0.1)


def test_tracker_vertical_velocity_flips_sign_on_a_bounce():
    """The signal the next POC relies on: vy changes sign at the impact."""
    tracker = BallTracker(0.03, 0.1, 8, 64, 3)

    # Falling, then bouncing back up from y = 400.
    for index in range(20):
        timestamp = index / 60
        tracker.update((200.0, 100.0 + 15.0 * index), timestamp, 10.0)
    falling_vy = tracker.velocity()[1]

    for index in range(20, 40):
        timestamp = index / 60
        tracker.update((200.0, 400.0 - 15.0 * (index - 20)), timestamp, 10.0)
    rising_vy = tracker.velocity()[1]

    assert falling_vy > 0  # y grows downwards in image coordinates
    assert rising_vy < 0
