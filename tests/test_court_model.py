"""Tests for the court model: geometry, projection and per-line scoring.

The model is what turns four known points into every line of the court, named.
A dimension derived wrongly here is invisible on screen — the overlay still
looks like a tennis court — and would quietly bias every in/out call in POC 3.
"""
from __future__ import annotations

import cv2
import numpy as np
import pytest

from temoin_cv.court import compute_homography, court_to_image, image_to_court, order_corners
from temoin_cv.court_model import (
    build_model,
    model_from_config,
    overall_score,
    project_model,
    reference_corners,
    score_lines,
    symmetric_orders,
)

# Full-size tennis court, in centimetres.
FULL = dict(
    length_cm=2377.0,
    doubles_width_cm=1097.0,
    singles_width_cm=823.0,
    service_line_from_net_cm=640.0,
)

DOUBLES_CORNERS = (
    "doubles_near_left",
    "doubles_near_right",
    "doubles_far_right",
    "doubles_far_left",
)


# -- geometry -------------------------------------------------------------------


def test_the_model_has_every_court_line():
    model = build_model(**FULL)
    assert set(model.lines) == {
        "baseline_near",
        "baseline_far",
        "sideline_doubles_left",
        "sideline_doubles_right",
        "sideline_singles_left",
        "sideline_singles_right",
        "service_line_near",
        "service_line_far",
        "centre_service_line",
        "net_line",
    }


def test_the_doubles_alleys_are_symmetric():
    model = build_model(**FULL)
    left = model.points["singles_near_left"][0]
    right = model.doubles_width_cm - model.points["singles_near_right"][0]

    assert left == pytest.approx(right)
    assert left == pytest.approx((1097.0 - 823.0) / 2)


def test_the_service_lines_straddle_the_net():
    model = build_model(**FULL)
    net_y = model.points["net_left"][1]

    assert net_y == pytest.approx(2377.0 / 2)
    assert net_y - model.points["service_near_left"][1] == pytest.approx(640.0)
    assert model.points["service_far_left"][1] - net_y == pytest.approx(640.0)


def test_the_centre_service_line_runs_between_the_service_lines():
    model = build_model(**FULL)
    start, end = model.lines["centre_service_line"]

    assert start[0] == pytest.approx(model.doubles_width_cm / 2)
    assert end[0] == pytest.approx(model.doubles_width_cm / 2)
    assert start[1] == pytest.approx(model.points["service_near_left"][1])
    assert end[1] == pytest.approx(model.points["service_far_left"][1])


def test_scale_shrinks_every_dimension_together():
    full = build_model(**FULL)
    scaled = build_model(**FULL, scale=0.15)

    assert scaled.length_cm == pytest.approx(full.length_cm * 0.15)
    assert scaled.doubles_width_cm == pytest.approx(full.doubles_width_cm * 0.15)
    for name, point in scaled.points.items():
        assert point == pytest.approx(tuple(v * 0.15 for v in full.points[name]))


def test_model_from_config_reads_the_court_section():
    model = model_from_config({**FULL, "scale": 0.15})
    assert model.doubles_width_cm == pytest.approx(1097.0 * 0.15)


# -- reference points --------------------------------------------------------------


def test_reference_corners_returns_the_named_points_in_order():
    model = build_model(**FULL)
    corners = reference_corners(model, DOUBLES_CORNERS)

    assert corners.shape == (4, 2)
    assert corners[0] == pytest.approx((0.0, 0.0))
    assert corners[2] == pytest.approx((1097.0, 2377.0))


def test_reference_corners_can_use_inner_points():
    """The outer corners are often out of frame on a camera placed close."""
    model = build_model(**FULL)
    corners = reference_corners(
        model,
        ("service_near_left", "service_near_right", "service_far_right", "service_far_left"),
    )
    assert corners[0] == pytest.approx((137.0, 548.5))


def test_reference_corners_rejects_a_wrong_count_or_unknown_name():
    model = build_model(**FULL)
    with pytest.raises(ValueError):
        reference_corners(model, DOUBLES_CORNERS[:3])
    with pytest.raises(KeyError):
        reference_corners(model, ("doubles_near_left",) * 3 + ("no_such_point",))


# -- projection and scoring ----------------------------------------------------------


def court_drawn_in_perspective(
    size=(480, 640), thickness: int = 4
) -> tuple[np.ndarray, np.ndarray, object]:
    """Draw the whole model through a known homography, as a camera would see it."""
    model = build_model(**FULL, scale=0.15)
    corners_cm = reference_corners(model, DOUBLES_CORNERS)
    in_image = np.array(
        [[120, 440], [520, 440], [400, 200], [240, 200]], dtype=np.float32
    )
    court_to_image = cv2.getPerspectiveTransform(corners_cm, in_image)

    mask = np.zeros(size, dtype=np.uint8)
    for start, end in model.lines.values():
        points = cv2.perspectiveTransform(
            np.array([[start, end]], dtype=np.float32), court_to_image
        )[0]
        cv2.line(mask, tuple(points[0].astype(int)), tuple(points[1].astype(int)), 255, thickness)

    return mask, in_image, model


def test_every_line_scores_full_marks_on_an_exact_homography():
    mask, in_image, model = court_drawn_in_perspective()
    matrix = compute_homography(in_image, reference_corners(model, DOUBLES_CORNERS))

    scored = score_lines(project_model(model, matrix), mask)

    assert len(scored) == 10
    assert min(scored.values()) > 0.95
    assert overall_score(scored) == pytest.approx(1.0, abs=0.05)


def test_the_score_collapses_when_the_corners_are_wrong():
    """The score has to discriminate, or it is decoration.

    It is also the objective function the automatic search will optimise, and
    the confidence POC 3 reports with an in or out call.
    """
    mask, in_image, model = court_drawn_in_perspective()
    corners_cm = reference_corners(model, DOUBLES_CORNERS)

    exact = overall_score(
        score_lines(project_model(model, compute_homography(in_image, corners_cm)), mask)
    )
    shifted = overall_score(
        score_lines(
            project_model(
                model,
                compute_homography(in_image + np.float32([20, -20]), corners_cm),
            ),
            mask,
        )
    )

    assert exact > 0.95
    assert shifted < 0.3


def test_a_line_projected_outside_the_frame_scores_zero():
    mask, in_image, model = court_drawn_in_perspective()
    # Push the whole court far off to the right of the frame.
    matrix = compute_homography(
        in_image + np.float32([5000, 0]), reference_corners(model, DOUBLES_CORNERS)
    )

    assert overall_score(score_lines(project_model(model, matrix), mask)) == 0.0


def test_score_lines_rescales_to_the_processing_frame():
    """The mask is built on a downscaled frame, the projection is not."""
    mask, in_image, model = court_drawn_in_perspective()
    matrix = compute_homography(in_image * 2.0, reference_corners(model, DOUBLES_CORNERS))

    scored = score_lines(project_model(model, matrix), mask, scale=0.5)
    assert overall_score(scored) == pytest.approx(1.0, abs=0.05)


def test_overall_score_of_nothing_is_zero():
    assert overall_score({}) == 0.0


# -- the symmetry trap ---------------------------------------------------------


def test_symmetric_orders_lists_four_distinct_assignments():
    orders = symmetric_orders(DOUBLES_CORNERS)

    assert len(orders) == 4
    assert len(set(orders)) == 4
    assert orders[0] == tuple(DOUBLES_CORNERS)


def test_all_four_symmetric_orders_score_perfectly_but_only_one_is_right():
    """The trap this POC must not fall into twice.

    A tennis court is symmetric about the net and along its length, so four
    corner assignments map it exactly onto itself. Every one of them projects
    all ten lines onto the tape and scores full marks, so neither the score nor
    the overlay can tell them apart. Only the coordinates can: three of them are
    out by most of the court's length.

    Which means the score must never be used to pick the orientation, and the
    only honest check is clicking a point whose real position is known.
    """
    mask, in_image, model = court_drawn_in_perspective()
    ordered_in_image = order_corners(in_image)

    errors, scores = [], []
    for names in symmetric_orders(DOUBLES_CORNERS):
        matrix = compute_homography(ordered_in_image, reference_corners(model, names))
        scores.append(overall_score(score_lines(project_model(model, matrix), mask)))

        # Where does the court's own centre come out?
        centre = (model.doubles_width_cm / 2, model.length_cm / 2)
        in_frame = court_to_image(centre, matrix)
        errors.append(
            float(np.hypot(*np.subtract(image_to_court(in_frame, matrix), centre)))
        )

    assert min(scores) > 0.95, "all four assignments overlay the court perfectly"

    # And yet they are genuinely different mappings: the net-line midpoint of
    # one lands at the far end of another.
    probes = [
        image_to_court(
            tuple(ordered_in_image[0]),
            compute_homography(ordered_in_image, reference_corners(model, names)),
        )
        for names in symmetric_orders(DOUBLES_CORNERS)
    ]
    assert len(set(tuple(round(v, 1) for v in p) for p in probes)) == 4
