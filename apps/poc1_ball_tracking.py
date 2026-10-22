"""Main loop: capture, detect, track, display.

Code and comments are in English; everything shown on screen or printed to the
terminal is in French, since that is the user-facing surface.

    python src/main.py                                  # webcam
    python src/main.py --source data/samples/test.mov   # replay a recording
    python src/main.py --record data/samples/test.mov   # webcam + save raw frames
"""
from __future__ import annotations

import argparse
import math
import time
from collections import deque
from pathlib import Path

import cv2

from capture import FrameRecorder, VideoSource
from config import load_config, parse_hsv_ranges
from detection import (
    ScoreWeights,
    build_mask,
    find_candidates,
    resize_for_processing,
    scale_candidate,
    select_best,
)
from tracking import BallTracker, TrackerState
from visualization import draw_ball, draw_hud, draw_trail, draw_velocity_vector, mask_to_bgr

SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "data" / "samples"

# -- user-facing strings, in French -------------------------------------------

WINDOW_MAIN = "Témoin — suivi"
WINDOW_MASK = "Témoin — masque"
KEY_HINTS = "q quitter   m masque   p pause   s capture"

STATE_LABELS = {
    TrackerState.DETECTED: "DÉTECTÉE",
    TrackerState.PREDICTED: "PRÉDITE",
    TrackerState.LOST: "PERDUE",
}

# Which configured colour to draw the ball with, per tracker state.
STATE_COLORS = {
    TrackerState.DETECTED: "ball_color",
    TrackerState.PREDICTED: "predicted_color",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="POC Témoin — suivi de balle en temps réel"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Vidéo à rejouer. Par défaut, la webcam définie dans config.yaml.",
    )
    parser.add_argument(
        "--config", default=None, help="Chemin du fichier de configuration YAML."
    )
    parser.add_argument(
        "--record",
        default=None,
        help="Enregistre les images brutes ici, pour rejouer la même séquence.",
    )
    return parser.parse_args()


def average_fps(durations: deque[float]) -> float:
    """Rolling average, not the instantaneous rate, which is unreadable."""
    if not durations:
        return 0.0
    mean = sum(durations) / len(durations)
    return 1.0 / mean if mean > 0 else 0.0


def build_hud_lines(
    tracker: BallTracker,
    fps: float,
    source: VideoSource,
    processing_width: int,
    detection_rate: float,
) -> list[str]:
    position = tracker.position
    vx, vy = tracker.velocity()
    ax, ay = tracker.acceleration()

    lines = [
        f"FPS {fps:5.1f}   {source.width}x{source.height} -> {processing_width}px"
        f"   balle vue {detection_rate * 100:.0f}%",
        f"ÉTAT {STATE_LABELS[tracker.state]}"
        f"   images sans détection {tracker.missed_frames}",
    ]

    if position is None:
        lines.append("position -            vitesse -")
    else:
        lines.append(
            f"position ({position[0]:6.1f}, {position[1]:6.1f})"
            f"   rayon {tracker.radius:5.1f}"
        )
        lines.append(
            f"vitesse  ({vx:7.1f}, {vy:7.1f}) px/s   norme {math.hypot(vx, vy):7.1f}"
        )
        lines.append(f"accél.   ({ax:7.0f}, {ay:7.0f}) px/s²")

    lines.append(KEY_HINTS)
    return lines


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    detection_config = config["detection"]
    tracking_config = config["tracking"]
    view_config = config["visualization"]

    ranges = parse_hsv_ranges(detection_config["hsv_ranges"])
    weights = ScoreWeights(
        area=float(detection_config["score_weights"]["area"]),
        circularity=float(detection_config["score_weights"]["circularity"]),
        distance=float(detection_config["score_weights"]["distance"]),
    )
    processing_width = int(detection_config["processing_width"])

    cv2.setUseOptimized(True)

    tracker = BallTracker(
        process_noise=float(tracking_config["process_noise"]),
        measurement_noise=float(tracking_config["measurement_noise"]),
        max_missed_frames=int(tracking_config["max_missed_frames"]),
        history_length=int(tracking_config["history_length"]),
        derivative_window=int(tracking_config["derivative_window"]),
    )

    durations: deque[float] = deque(maxlen=int(view_config["fps_window"]))
    show_mask = bool(view_config["show_mask_on_start"])
    paused = False
    frames_seen = 0
    frames_detected = 0

    with VideoSource.from_config(config, args.source) as source:
        recorder = (
            FrameRecorder(
                args.record,
                source.fps,
                (source.width, source.height),
                str(config["capture"].get("record_fourcc", "mp4v")),
            )
            if args.record
            else None
        )
        # A file is played back at its own rate; the webcam is read as fast as
        # frames arrive.
        key_delay = max(1, int(1000 / source.fps)) if source.is_file else 1

        try:
            while True:
                if paused:
                    key = cv2.waitKey(50) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("p"):
                        paused = False
                    continue

                loop_start = time.perf_counter()

                ok, frame, timestamp = source.read()
                if not ok or frame is None:
                    break

                if recorder is not None:
                    # Raw frames, without the overlay, so a replay can be run
                    # again through the detection with other thresholds.
                    recorder.write(frame)

                small, scale = resize_for_processing(frame, processing_width)
                mask = build_mask(
                    small,
                    ranges,
                    int(detection_config["blur_ksize"]),
                    int(detection_config["open_ksize"]),
                    int(detection_config["close_ksize"]),
                )
                candidates = find_candidates(
                    mask,
                    float(detection_config["min_area"]),
                    float(detection_config["max_area"]),
                    float(detection_config["min_circularity"]),
                )

                predicted = tracker.predicted_center
                predicted_small = (
                    (predicted[0] * scale, predicted[1] * scale) if predicted else None
                )
                best = select_best(
                    candidates,
                    predicted_small,
                    weights,
                    math.hypot(small.shape[1], small.shape[0]),
                )

                if best is not None:
                    found = scale_candidate(best, scale)
                    tracker.update(found.center, timestamp, found.radius)
                    frames_detected += 1
                else:
                    tracker.update(None, timestamp)

                frames_seen += 1

                position = tracker.position
                if position is not None:
                    draw_trail(
                        frame,
                        tracker.history,
                        tuple(view_config["trail_color"]),
                        int(view_config["trail_length"]),
                    )
                    color_key = STATE_COLORS.get(tracker.state, "ball_color")
                    draw_ball(
                        frame, position, tracker.radius, tuple(view_config[color_key])
                    )
                    draw_velocity_vector(
                        frame,
                        position,
                        tracker.velocity(),
                        float(view_config["velocity_scale"]),
                        tuple(view_config["velocity_color"]),
                    )

                draw_hud(
                    frame,
                    build_hud_lines(
                        tracker,
                        average_fps(durations),
                        source,
                        processing_width,
                        frames_detected / frames_seen if frames_seen else 0.0,
                    ),
                    tuple(view_config["hud_color"]),
                )

                cv2.imshow(WINDOW_MAIN, frame)
                if show_mask:
                    cv2.imshow(WINDOW_MASK, mask_to_bgr(mask))

                durations.append(time.perf_counter() - loop_start)

                key = cv2.waitKey(key_delay) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("p"):
                    paused = True
                if key == ord("m"):
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow(WINDOW_MASK)
                if key == ord("s"):
                    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                    path = SNAPSHOT_DIR / f"capture_{int(time.time())}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"Capture enregistrée dans {path}")
        finally:
            if recorder is not None:
                recorder.release()
            cv2.destroyAllWindows()

    detected_share = (frames_detected / frames_seen * 100) if frames_seen else 0.0
    print(
        f"{frames_seen} images, balle détectée sur {frames_detected} "
        f"({detected_share:.0f} %), moyenne {average_fps(durations):.1f} FPS"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
