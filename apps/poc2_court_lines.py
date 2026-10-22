"""POC 2: detect a taped rectangle on the floor and map it to real centimetres.

Code and comments are in English; everything shown on screen or printed to the
terminal is in French, since that is the user-facing surface.

    python -m apps.poc2_court_lines
    python -m apps.poc2_court_lines --source data/samples/court.mov
    python -m apps.poc2_court_lines --record data/samples/court.mov
"""
from __future__ import annotations

import argparse
import enum
import time
from collections import deque

import cv2
import numpy as np

from temoin_cv.capture import FrameRecorder, VideoSource
from temoin_cv.config import DATA_DIR, config_path, load_config, parse_hsv_ranges
from temoin_cv.court import (
    CornerSmoother,
    compute_homography,
    court_corners_cm,
    enhance_lines,
    find_quad_by_contour,
    find_quad_by_hough,
    image_to_court,
    order_corners,
    warp_top_down,
)
from temoin_cv.masking import build_mask, resize_for_processing
from temoin_cv.visualization import (
    draw_corners,
    draw_court_marker,
    draw_grid,
    draw_hud,
    draw_quad,
    mask_to_bgr,
)

SNAPSHOT_DIR = DATA_DIR / "samples"
DEFAULT_CONFIG = "court.yaml"
CORNERS_NEEDED = 4

# -- user-facing strings, in French -------------------------------------------

WINDOW_MAIN = "Témoin — court"
WINDOW_TOP = "Témoin — vue de dessus"
WINDOW_MASK = "Témoin — masque du blanc"
KEY_HINTS = "m masque   c calibrer   r réinitialiser   l verrouiller   q quitter"

CORNER_NAMES = ("haut-gauche", "haut-droite", "bas-droite", "bas-gauche")


class Mode(enum.Enum):
    AUTO = "AUTO"
    MANUAL = "MANUAL"
    LOCKED = "LOCKED"


MODE_LABELS = {
    Mode.AUTO: "AUTOMATIQUE",
    Mode.MANUAL: "CALIBRAGE MANUEL",
    Mode.LOCKED: "VERROUILLÉ",
}

METHOD_LABELS = {
    "contour": "contours",
    "hough": "Hough (secours)",
    "manual": "coins cliqués",
}


class AppState:
    """What the mouse callback and the loop share."""

    def __init__(self) -> None:
        self.mode = Mode.AUTO
        self.manual_points: list[tuple[float, float]] = []
        self.corners: np.ndarray | None = None
        self.method = "contour"
        self.probe_cm: tuple[float, float] | None = None
        self.probe_px: tuple[float, float] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="POC Témoin — détection des lignes du court et homographie"
    )
    parser.add_argument(
        "--source",
        default=None,
        help="Vidéo à rejouer. Par défaut, la webcam définie dans la config.",
    )
    parser.add_argument(
        "--config",
        default=str(config_path(DEFAULT_CONFIG)),
        help="Chemin du fichier de configuration YAML.",
    )
    parser.add_argument(
        "--record",
        default=None,
        help="Enregistre les images brutes ici, pour rejouer la même séquence.",
    )
    return parser.parse_args()


def average_fps(durations: deque[float]) -> float:
    if not durations:
        return 0.0
    mean = sum(durations) / len(durations)
    return 1.0 / mean if mean > 0 else 0.0


def detect_quad(
    mask: np.ndarray, quad_config: dict
) -> tuple[np.ndarray | None, str]:
    """Find the rectangle, falling back to Hough when the contour will not close.

    The fallback is what makes worn or taped-over lines usable: a broken edge
    never closes into a quadrilateral, but it still votes for the right line.
    """
    min_area_ratio = float(quad_config["min_area_ratio"])

    if str(quad_config["method"]) == "hough":
        return find_quad_by_hough(mask, quad_config["hough"], min_area_ratio), "hough"

    found = find_quad_by_contour(
        mask, min_area_ratio, float(quad_config["epsilon_ratio"])
    )
    if found is not None:
        return found, "contour"
    return find_quad_by_hough(mask, quad_config["hough"], min_area_ratio), "hough"


def on_mouse(event: int, x: int, y: int, _flags: int, state: AppState) -> None:
    """Collect the manual corners, or probe a point's real position."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state.mode is Mode.MANUAL:
        state.manual_points.append((float(x), float(y)))
        return

    state.probe_px = (float(x), float(y))


def build_hud_lines(
    state: AppState,
    fps: float,
    source: VideoSource,
    processing_width: int,
    jitter: float,
    court_size_cm: tuple[float, float],
) -> list[str]:
    lines = [
        f"FPS {fps:5.1f}   {source.width}x{source.height} -> {processing_width}px"
        f"   rectangle {court_size_cm[0]:.0f} x {court_size_cm[1]:.0f} cm",
    ]

    if state.mode is Mode.MANUAL:
        index = len(state.manual_points)
        if index < CORNERS_NEEDED:
            lines.append(
                f"ÉTAT {MODE_LABELS[state.mode]}"
                f"   cliquer le coin {index + 1}/4 : {CORNER_NAMES[index]}"
            )
        else:
            lines.append(f"ÉTAT {MODE_LABELS[state.mode]}   coins complets")
    else:
        detail = (
            METHOD_LABELS.get(state.method, state.method)
            if state.corners is not None
            else "rectangle non trouvé"
        )
        lines.append(f"ÉTAT {MODE_LABELS[state.mode]}   {detail}")

    if state.corners is not None and state.mode is Mode.AUTO:
        lines.append(f"stabilité des coins {jitter:5.2f} px")

    if state.probe_cm is not None:
        lines.append(
            f"point cliqué : {state.probe_cm[0]:7.1f} cm, {state.probe_cm[1]:7.1f} cm"
        )

    lines.append(KEY_HINTS)
    return lines


def main() -> int:
    args = parse_args()
    config = load_config(args.config)

    detection_config = config["detection"]
    quad_config = config["quad"]
    court_config = config["court"]
    view_config = config["visualization"]

    ranges = parse_hsv_ranges(detection_config["hsv_ranges"])
    processing_width = int(detection_config["processing_width"])
    tophat_ksize = int(detection_config.get("tophat_ksize", 0))

    width_cm = float(court_config["width_cm"])
    height_cm = float(court_config["height_cm"])
    px_per_cm = float(court_config["px_per_cm"])
    grid_step_cm = float(court_config["grid_step_cm"])
    court_corners = court_corners_cm(width_cm, height_cm)

    cv2.setUseOptimized(True)

    state = AppState()
    smoother = CornerSmoother(int(config["smoothing"]["corner_window"]))
    durations: deque[float] = deque(maxlen=int(view_config["fps_window"]))
    show_mask = bool(view_config["show_mask_on_start"])

    cv2.namedWindow(WINDOW_MAIN)
    cv2.setMouseCallback(WINDOW_MAIN, on_mouse, state)

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
        key_delay = max(1, int(1000 / source.fps)) if source.is_file else 1
        last_frame = None

        try:
            while True:
                loop_start = time.perf_counter()

                ok, frame, _ = source.read()
                if not ok or frame is None:
                    if source.is_file and last_frame is not None:
                        # Hold on the last frame so the corners can be clicked
                        # calmly once the recording has played out.
                        frame = last_frame.copy()
                    else:
                        break
                else:
                    last_frame = frame.copy()
                    if recorder is not None:
                        recorder.write(frame)

                small, scale = resize_for_processing(frame, processing_width)
                mask = build_mask(
                    small,
                    ranges,
                    int(detection_config["blur_ksize"]),
                    int(detection_config["open_ksize"]),
                    int(detection_config["close_ksize"]),
                )
                mask = enhance_lines(mask, tophat_ksize)

                if state.mode is Mode.MANUAL:
                    if len(state.manual_points) >= CORNERS_NEEDED:
                        state.corners = order_corners(
                            state.manual_points[:CORNERS_NEEDED]
                        )
                        state.method = "manual"
                        state.mode = Mode.LOCKED
                        smoother.reset()
                        print("Calibrage manuel enregistré, coins verrouillés.")
                elif state.mode is Mode.AUTO:
                    found, method = detect_quad(mask, quad_config)
                    if found is not None:
                        # Detection runs downscaled; bring the corners back to
                        # display pixels, which is also what a click gives.
                        state.corners = smoother.update(found / scale)
                        state.method = method
                    else:
                        state.corners = None
                        smoother.reset()

                matrix = (
                    compute_homography(state.corners, court_corners)
                    if state.corners is not None
                    else None
                )

                if state.probe_px is not None and matrix is not None:
                    state.probe_cm = image_to_court(state.probe_px, matrix)
                    print(
                        f"Point cliqué : {state.probe_cm[0]:.1f} cm, "
                        f"{state.probe_cm[1]:.1f} cm"
                    )
                    state.probe_px = None

                # The top-down view is built BEFORE the overlay is drawn:
                # warping an annotated frame rectifies the overlay too, which
                # puts the HUD band and the quad outline inside the view.
                top_down = None
                if matrix is not None:
                    top_down = warp_top_down(
                        frame, matrix, width_cm, height_cm, px_per_cm
                    )
                    draw_grid(
                        top_down, px_per_cm, grid_step_cm, tuple(view_config["grid_color"])
                    )
                    if state.probe_cm is not None:
                        draw_court_marker(
                            top_down,
                            state.probe_cm,
                            px_per_cm,
                            tuple(view_config["marker_color"]),
                        )

                color_key = {
                    Mode.AUTO: "quad_color",
                    Mode.MANUAL: "manual_color",
                    Mode.LOCKED: "locked_color",
                }[state.mode]

                if state.corners is not None:
                    draw_quad(frame, state.corners, tuple(view_config[color_key]))
                    draw_corners(
                        frame,
                        state.corners,
                        tuple(view_config["corner_color"]),
                        [str(index + 1) for index in range(CORNERS_NEEDED)],
                    )
                if state.mode is Mode.MANUAL and state.manual_points:
                    draw_corners(
                        frame,
                        np.array(state.manual_points),
                        tuple(view_config["manual_color"]),
                        [str(index + 1) for index in range(len(state.manual_points))],
                    )

                draw_hud(
                    frame,
                    build_hud_lines(
                        state,
                        average_fps(durations),
                        source,
                        processing_width,
                        smoother.jitter,
                        (width_cm, height_cm),
                    ),
                    tuple(view_config["hud_color"]),
                )
                cv2.imshow(WINDOW_MAIN, frame)
                if top_down is not None:
                    cv2.imshow(WINDOW_TOP, top_down)

                if show_mask:
                    cv2.imshow(WINDOW_MASK, mask_to_bgr(mask))

                durations.append(time.perf_counter() - loop_start)

                key = cv2.waitKey(key_delay) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("m"):
                    show_mask = not show_mask
                    if not show_mask:
                        cv2.destroyWindow(WINDOW_MASK)
                if key == ord("c"):
                    state.mode = Mode.MANUAL
                    state.manual_points.clear()
                    state.corners = None
                    smoother.reset()
                    print("Calibrage manuel : cliquer les quatre coins dans l'ordre.")
                if key == ord("r"):
                    state.mode = Mode.AUTO
                    state.manual_points.clear()
                    state.corners = None
                    state.probe_cm = None
                    smoother.reset()
                    print("Retour à la détection automatique.")
                if key == ord("l") and state.corners is not None:
                    state.mode = (
                        Mode.AUTO if state.mode is Mode.LOCKED else Mode.LOCKED
                    )
                    print(
                        "Coins verrouillés."
                        if state.mode is Mode.LOCKED
                        else "Coins déverrouillés."
                    )
                if key == ord("s"):
                    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                    path = SNAPSHOT_DIR / f"court_{int(time.time())}.png"
                    cv2.imwrite(str(path), frame)
                    print(f"Capture enregistrée dans {path}")
        finally:
            if recorder is not None:
                recorder.release()
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
