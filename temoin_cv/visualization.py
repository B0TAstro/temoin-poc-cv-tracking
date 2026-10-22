"""Overlay drawing: ball, trail, velocity vector, HUD.

Drawing only. No decision about what the ball is or where it goes is taken
here, so the overlay can be dropped entirely when porting to iPhone.
"""
from __future__ import annotations

import math
from typing import Sequence

import cv2
import numpy as np

Color = tuple[int, int, int]
Sample = tuple[float, float, float]

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.55
LINE_HEIGHT = 24
MARGIN = 12


def draw_ball(
    frame: np.ndarray,
    center: tuple[float, float],
    radius: float,
    color: Color,
    thickness: int = 2,
) -> np.ndarray:
    """Circle around the ball plus a dot on its centre."""
    point = (int(round(center[0])), int(round(center[1])))
    cv2.circle(frame, point, max(1, int(round(radius))), color, thickness)
    cv2.circle(frame, point, 3, color, -1)
    return frame


def draw_trail(
    frame: np.ndarray,
    history: Sequence[Sample],
    color: Color,
    max_points: int,
) -> np.ndarray:
    """Recent positions, drawn thinner as they get older."""
    points = list(history)[-max_points:]
    if len(points) < 2:
        return frame

    total = len(points)
    for index in range(1, total):
        start = (int(round(points[index - 1][1])), int(round(points[index - 1][2])))
        end = (int(round(points[index][1])), int(round(points[index][2])))
        age_ratio = index / total
        thickness = max(1, int(round(1 + 3 * age_ratio)))
        faded = tuple(int(channel * (0.35 + 0.65 * age_ratio)) for channel in color)
        cv2.line(frame, start, end, faded, thickness)

    return frame


def draw_velocity_vector(
    frame: np.ndarray,
    center: tuple[float, float],
    velocity: tuple[float, float],
    scale: float,
    color: Color,
) -> np.ndarray:
    """Arrow from the ball, length proportional to speed."""
    vx, vy = velocity
    if math.hypot(vx, vy) < 1.0:
        return frame

    start = (int(round(center[0])), int(round(center[1])))
    end = (int(round(center[0] + vx * scale)), int(round(center[1] + vy * scale)))
    cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.25)
    return frame


def draw_hud(frame: np.ndarray, lines: Sequence[str], color: Color) -> np.ndarray:
    """Text lines in the top-left corner, over a dark band for readability."""
    if not lines:
        return frame

    band_height = MARGIN + LINE_HEIGHT * len(lines)
    overlay = frame[:band_height, :].copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], band_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame[:band_height, :], 0.55, 0, dst=frame[:band_height, :])

    for index, line in enumerate(lines):
        position = (MARGIN, MARGIN + LINE_HEIGHT * index + 6)
        cv2.putText(frame, line, position, FONT, FONT_SCALE, color, 1, cv2.LINE_AA)

    return frame


def mask_to_bgr(mask: np.ndarray) -> np.ndarray:
    """Turn the single-channel mask into something imshow can display."""
    return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
