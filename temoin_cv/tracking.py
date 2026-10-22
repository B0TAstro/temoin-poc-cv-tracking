"""Ball tracking: constant-velocity Kalman filter plus trajectory history.

The filter smooths the measured position and predicts the next one, which lets
the ball survive a few frames without a detection. The history feeds the trail
and the finite differences; the vertical velocity it exposes is what the next
POC will watch for a sign change to detect a bounce.
"""
from __future__ import annotations

import enum
from collections import deque
from typing import Deque, Sequence

import cv2
import numpy as np

# A timestamped position: (t_seconds, x_pixels, y_pixels).
Sample = tuple[float, float, float]

MIN_DT = 1e-4


class TrackerState(enum.Enum):
    LOST = "LOST"
    DETECTED = "DETECTED"
    PREDICTED = "PREDICTED"


def linear_slope(times: Sequence[float], values: Sequence[float]) -> float:
    """Least-squares slope of values against times.

    A least-squares fit over the window rather than a plain difference between
    two points: the two-point version reacts to a single pixel of jitter, which
    is exactly the noise the bounce detection must not trip on. Pure arithmetic,
    so it ports to Swift as is.
    """
    count = len(times)
    if count < 2:
        return 0.0

    mean_t = sum(times) / count
    mean_v = sum(values) / count

    numerator = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
    denominator = sum((t - mean_t) ** 2 for t in times)
    if denominator <= 0.0:
        return 0.0
    return numerator / denominator


def finite_difference(samples: Sequence[Sample], window: int) -> tuple[float, float]:
    """Derivative of the last `window` samples, per second, on both axes."""
    recent = list(samples)[-max(2, window) :]
    if len(recent) < 2:
        return 0.0, 0.0

    times = [sample[0] for sample in recent]
    return (
        linear_slope(times, [sample[1] for sample in recent]),
        linear_slope(times, [sample[2] for sample in recent]),
    )


class BallTracker:
    """Follows one ball across frames, tolerating short detection dropouts."""

    def __init__(
        self,
        process_noise: float,
        measurement_noise: float,
        max_missed_frames: int,
        history_length: int,
        derivative_window: int,
    ) -> None:
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.max_missed_frames = max_missed_frames
        self.derivative_window = derivative_window

        self.state = TrackerState.LOST
        self.missed_frames = 0
        self.radius = 0.0

        self.history: Deque[Sample] = deque(maxlen=history_length)
        self.velocity_history: Deque[Sample] = deque(maxlen=history_length)

        self._kf: cv2.KalmanFilter | None = None
        self._last_time: float | None = None
        self._predicted_center: tuple[float, float] | None = None

    # -- public read-only view -------------------------------------------------

    @property
    def position(self) -> tuple[float, float] | None:
        """Filtered position, or None while the ball is lost."""
        if self._kf is None or self.state is TrackerState.LOST:
            return None
        return float(self._kf.statePost[0, 0]), float(self._kf.statePost[1, 0])

    @property
    def predicted_center(self) -> tuple[float, float] | None:
        """Where the next detection is expected; gates candidate selection."""
        return self._predicted_center

    def velocity(self) -> tuple[float, float]:
        """Velocity in pixels per second, from the position history."""
        return finite_difference(self.history, self.derivative_window)

    def acceleration(self) -> tuple[float, float]:
        """Acceleration in pixels per second squared, from the velocity history."""
        return finite_difference(self.velocity_history, self.derivative_window)

    # -- update ----------------------------------------------------------------

    def update(
        self,
        measurement: tuple[float, float] | None,
        timestamp: float,
        radius: float = 0.0,
    ) -> TrackerState:
        """Advance the tracker by one frame and return the resulting state."""
        if self.state is TrackerState.LOST:
            if measurement is None:
                self._predicted_center = None
                return self.state
            self._start(measurement, timestamp, radius)
            return self.state

        assert self._kf is not None and self._last_time is not None

        dt = max(MIN_DT, timestamp - self._last_time)
        self._kf.transitionMatrix = np.array(
            [
                [1.0, 0.0, dt, 0.0],
                [0.0, 1.0, 0.0, dt],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        prediction = self._kf.predict()
        self._predicted_center = (float(prediction[0, 0]), float(prediction[1, 0]))

        if measurement is not None:
            self._kf.correct(
                np.array([[measurement[0]], [measurement[1]]], dtype=np.float32)
            )
            self.state = TrackerState.DETECTED
            self.missed_frames = 0
            self.radius = radius
        else:
            # Coast on the prediction for a few frames before giving up.
            self.missed_frames += 1
            if self.missed_frames > self.max_missed_frames:
                self._reset()
                return self.state
            self.state = TrackerState.PREDICTED

        self._last_time = timestamp
        self._record(timestamp)
        return self.state

    # -- internals -------------------------------------------------------------

    def _start(
        self, measurement: tuple[float, float], timestamp: float, radius: float
    ) -> None:
        """Create the filter on the first detection after being lost."""
        kf = cv2.KalmanFilter(4, 2)
        kf.measurementMatrix = np.array(
            [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32
        )
        kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.process_noise
        kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.measurement_noise
        kf.errorCovPost = np.eye(4, dtype=np.float32)
        kf.statePost = np.array(
            [[measurement[0]], [measurement[1]], [0.0], [0.0]], dtype=np.float32
        )

        self._kf = kf
        self._last_time = timestamp
        self._predicted_center = measurement
        self.state = TrackerState.DETECTED
        self.missed_frames = 0
        self.radius = radius
        self._record(timestamp)

    def _record(self, timestamp: float) -> None:
        """Append the filtered position, and the velocity it implies."""
        position = self.position
        if position is None:
            return

        self.history.append((timestamp, position[0], position[1]))
        vx, vy = finite_difference(self.history, self.derivative_window)
        self.velocity_history.append((timestamp, vx, vy))

    def _reset(self) -> None:
        """Drop the ball: clear the filter and the trajectory."""
        self.state = TrackerState.LOST
        self.missed_frames = 0
        self.radius = 0.0
        self._kf = None
        self._last_time = None
        self._predicted_center = None
        self.history.clear()
        self.velocity_history.clear()
