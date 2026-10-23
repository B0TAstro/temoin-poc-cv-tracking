"""The court as a known geometry: named points and lines, in centimetres.

A court is not one rectangle, it is a network of lines whose dimensions are
known in advance. That changes what detection has to do: find enough of it to
pin down the homography, and every remaining line follows by projection —
already named, which raw line detection could never give.

Everything here is pure arithmetic on a handful of dimensions, so it ports to
Swift as is and the same model will drive the in/out call of POC 3.
"""
from __future__ import annotations

from typing import Any, NamedTuple, Sequence

import numpy as np

from temoin_cv.court import Corners, court_to_image, segment_support

Point = tuple[float, float]
Segment = tuple[Point, Point]


class CourtModel(NamedTuple):
    """A court in its own coordinates: x across, y from near to far baseline."""

    points: dict[str, Point]
    lines: dict[str, Segment]
    length_cm: float
    doubles_width_cm: float


def build_model(
    length_cm: float,
    doubles_width_cm: float,
    singles_width_cm: float,
    service_line_from_net_cm: float,
    scale: float = 1.0,
) -> CourtModel:
    """Derive every court point and line from four dimensions.

    `scale` lets the real tennis measurements stay in the config while a scaled
    copy is what is actually taped on the floor: write the full-size numbers
    once, then say how big the copy is.
    """
    length = length_cm * scale
    width = doubles_width_cm * scale
    singles = singles_width_cm * scale
    service_offset = service_line_from_net_cm * scale

    alley = (width - singles) / 2.0  # the doubles alley, each side
    net_y = length / 2.0
    service_near_y = net_y - service_offset
    service_far_y = net_y + service_offset

    points: dict[str, Point] = {
        "doubles_near_left": (0.0, 0.0),
        "doubles_near_right": (width, 0.0),
        "doubles_far_right": (width, length),
        "doubles_far_left": (0.0, length),
        "singles_near_left": (alley, 0.0),
        "singles_near_right": (width - alley, 0.0),
        "singles_far_left": (alley, length),
        "singles_far_right": (width - alley, length),
        "service_near_left": (alley, service_near_y),
        "service_near_right": (width - alley, service_near_y),
        "service_far_left": (alley, service_far_y),
        "service_far_right": (width - alley, service_far_y),
        "centre_near": (width / 2.0, service_near_y),
        "centre_far": (width / 2.0, service_far_y),
        "net_left": (0.0, net_y),
        "net_right": (width, net_y),
    }

    lines: dict[str, Segment] = {
        "baseline_near": (points["doubles_near_left"], points["doubles_near_right"]),
        "baseline_far": (points["doubles_far_left"], points["doubles_far_right"]),
        "sideline_doubles_left": (
            points["doubles_near_left"],
            points["doubles_far_left"],
        ),
        "sideline_doubles_right": (
            points["doubles_near_right"],
            points["doubles_far_right"],
        ),
        "sideline_singles_left": (
            points["singles_near_left"],
            points["singles_far_left"],
        ),
        "sideline_singles_right": (
            points["singles_near_right"],
            points["singles_far_right"],
        ),
        "service_line_near": (points["service_near_left"], points["service_near_right"]),
        "service_line_far": (points["service_far_left"], points["service_far_right"]),
        "centre_service_line": (points["centre_near"], points["centre_far"]),
        "net_line": (points["net_left"], points["net_right"]),
    }

    return CourtModel(points, lines, length, width)


def model_from_config(court_config: dict[str, Any]) -> CourtModel:
    """Build the model from the `court` section of a config file."""
    return build_model(
        length_cm=float(court_config["length_cm"]),
        doubles_width_cm=float(court_config["doubles_width_cm"]),
        singles_width_cm=float(court_config["singles_width_cm"]),
        service_line_from_net_cm=float(court_config["service_line_from_net_cm"]),
        scale=float(court_config.get("scale", 1.0)),
    )


def reference_corners(model: CourtModel, names: Sequence[str]) -> Corners:
    """The four model points the user calibrates on, in centimetres.

    Configurable by name because the outer corners are often out of frame: a
    camera close to a taped court sees the service box corners long before it
    sees the baselines.
    """
    if len(names) != 4:
        raise ValueError(f"expected 4 reference points, got {len(names)}")

    missing = [name for name in names if name not in model.points]
    if missing:
        raise KeyError(f"unknown court points: {', '.join(missing)}")

    return np.array([model.points[name] for name in names], dtype=np.float32)


def symmetric_orders(names: Sequence[str]) -> list[tuple[str, ...]]:
    """The four point orders a symmetric court cannot tell apart.

    A tennis court is symmetric about the net AND along its length, so four
    different corner assignments map it exactly onto itself. All four project
    every line onto the tape and score full marks; only one gives the right
    coordinates, and the other three are out by up to the length of the court.

    No image can resolve this — it takes knowing which end the camera stands at.
    Hence a list to cycle through rather than a rule to apply.
    """
    base = tuple(names)
    flipped = tuple(reversed(base))
    return [base, base[2:] + base[:2], flipped, flipped[2:] + flipped[:2]]


def project_model(model: CourtModel, matrix: np.ndarray) -> dict[str, Segment]:
    """Project every model line into the image, given a centimetre homography."""
    return {
        name: (court_to_image(start, matrix), court_to_image(end, matrix))
        for name, (start, end) in model.lines.items()
    }


def score_lines(
    projected: dict[str, Segment], line_mask: np.ndarray, scale: float = 1.0
) -> dict[str, float]:
    """How much of each projected line actually sits on detected line pixels.

    This is the honest per-line confidence: a line the camera cannot see, or a
    homography that is wrong, both show up as a low score. `scale` converts
    display coordinates to the processing frame the mask was built on.
    """
    height, width = line_mask.shape[:2]

    scored: dict[str, float] = {}
    for name, (start, end) in projected.items():
        scaled_start = (start[0] * scale, start[1] * scale)
        scaled_end = (end[0] * scale, end[1] * scale)
        # A line projected far outside the frame cannot be judged, only missed.
        if not (
            _inside(scaled_start, width, height) or _inside(scaled_end, width, height)
        ):
            scored[name] = 0.0
            continue
        scored[name] = segment_support(line_mask, scaled_start, scaled_end)

    return scored


def _inside(point: Point, width: int, height: int, margin: float = 0.25) -> bool:
    """Is the point within the frame, allowing a margin for a line running out?"""
    return (
        -margin * width <= point[0] <= width * (1 + margin)
        and -margin * height <= point[1] <= height * (1 + margin)
    )


def overall_score(scored: dict[str, float]) -> float:
    """Mean support over the lines, the single number to put on screen."""
    return float(np.mean(list(scored.values()))) if scored else 0.0
