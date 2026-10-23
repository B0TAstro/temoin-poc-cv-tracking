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
    clip_line_to_frame,
    compute_homography,
    enhance_lines,
    find_line_candidates,
    find_quad_by_contour,
    find_quad_from_candidates,
    image_to_court,
    order_corners,
    warp_top_down,
)
from temoin_cv.court_model import (
    model_from_config,
    overall_score,
    project_model,
    reference_corners,
    score_lines,
    symmetric_orders,
)
from temoin_cv.masking import build_mask, resize_for_processing
from temoin_cv.visualization import (
    draw_corners,
    draw_line_candidates,
    draw_court_lines,
    draw_court_marker,
    draw_grid,
    draw_hud,
    draw_quad,
    mask_to_bgr,
)

SNAPSHOT_DIR = DATA_DIR / "samples"
DEFAULT_CONFIG = "court.yaml"
CORNERS_NEEDED = 4
# Keep the top-down window clear of the main window's banner. It is positioned
# on its first imshow: moveWindow does nothing to a window with no content yet,
# which is how it kept landing back on the banner.
TOP_WINDOW_Y = 760

# -- user-facing strings, in French -------------------------------------------

WINDOW_MAIN = "Temoin - court"
WINDOW_TOP = "Temoin - vue de dessus"
WINDOW_MASK = "Temoin - masque du blanc"
KEY_HINTS = "m masque   c calibrer   o orientation   r réinit.   l verrouiller   q quitter"

# French labels for the model's English point and line names.
POINT_LABELS = {
    "doubles_near_left": "fond proche, couloir gauche",
    "doubles_near_right": "fond proche, couloir droit",
    "doubles_far_right": "fond opposé, couloir droit",
    "doubles_far_left": "fond opposé, couloir gauche",
    "singles_near_left": "fond proche, simple gauche",
    "singles_near_right": "fond proche, simple droit",
    "singles_far_left": "fond opposé, simple gauche",
    "singles_far_right": "fond opposé, simple droit",
    "service_near_left": "service proche, gauche",
    "service_near_right": "service proche, droit",
    "service_far_left": "service opposé, gauche",
    "service_far_right": "service opposé, droit",
    "centre_near": "médiane, côté proche",
    "centre_far": "médiane, côté opposé",
    "net_left": "filet, gauche",
    "net_right": "filet, droit",
}

LINE_LABELS = {
    "baseline_near": "fond proche",
    "baseline_far": "fond opposé",
    "sideline_doubles_left": "couloir gauche",
    "sideline_doubles_right": "couloir droit",
    "sideline_singles_left": "simple gauche",
    "sideline_singles_right": "simple droit",
    "service_line_near": "service proche",
    "service_line_far": "service opposé",
    "centre_service_line": "ligne médiane",
    "net_line": "filet",
}


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
        self.scored: dict[str, float] = {}
        self.reference_names: tuple[str, ...] = ()
        self.orientation = 0
        self.candidates: list = []


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
) -> tuple[np.ndarray | None, str, list]:
    """Find the rectangle, and return the candidate lines either way.

    The contour route only ever fires on a closed outline. A court is a ladder
    of separate bands, so the Hough route over the candidate lines is the
    normal path, not the exception.
    """
    min_area_ratio = float(quad_config["min_area_ratio"])
    min_support = float(quad_config.get("min_support", 0.0))
    candidates = find_line_candidates(mask, quad_config["hough"])

    if str(quad_config["method"]) == "contour":
        found = find_quad_by_contour(
            mask, min_area_ratio, float(quad_config["epsilon_ratio"]), min_support
        )
        if found is not None:
            return found, "contour", candidates

    found = find_quad_from_candidates(
        mask, candidates, quad_config["hough"], min_area_ratio, min_support
    )
    return found, "hough", candidates


def on_mouse(event: int, x: int, y: int, _flags: int, state: AppState) -> None:
    """Collect the manual corners, or probe a point's real position."""
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    if state.mode is Mode.MANUAL:
        state.manual_points.append((float(x), float(y)))
        index = len(state.manual_points)
        if index <= CORNERS_NEEDED:
            name = (
                state.reference_names[index - 1]
                if index <= len(state.reference_names)
                else "?"
            )
            print(
                f"Point {index}/{CORNERS_NEEDED} pris en ({x}, {y}) :"
                f" {POINT_LABELS.get(name, name)}"
            )
        return

    state.probe_px = (float(x), float(y))


def build_hud_lines(
    state: AppState,
    fps: float,
    source: VideoSource,
    processing_width: int,
    jitter: float,
    court_size_cm: tuple[float, float],
    good: float,
) -> list[str]:
    lines = [
        f"FPS {fps:5.1f}   {source.width}x{source.height} -> {processing_width}px"
        f"   court {court_size_cm[0]:.0f} x {court_size_cm[1]:.0f} cm",
    ]

    if state.mode is Mode.MANUAL:
        index = len(state.manual_points)
        if index < CORNERS_NEEDED:
            point_name = state.reference_names[index]
            label = POINT_LABELS.get(point_name, point_name)
            lines.append(
                f"ÉTAT {MODE_LABELS[state.mode]}"
                f"   cliquer le point {index + 1}/4 : {label}"
            )
        else:
            lines.append(f"ÉTAT {MODE_LABELS[state.mode]}   points complets")
    else:
        detail = (
            METHOD_LABELS.get(state.method, state.method)
            if state.corners is not None
            else "aucun rectangle"
        )
        lines.append(f"ÉTAT {MODE_LABELS[state.mode]}   {detail}")

    if state.mode is Mode.AUTO:
        across = sum(1 for item in state.candidates if item.horizontal)
        along = len(state.candidates) - across
        hint = "" if across >= 2 and along >= 2 else "   il en faut 2 de chaque"
        lines.append(
            f"lignes candidates : {across} en travers, {along} en long{hint}"
        )

    if state.scored:
        seen = sum(1 for value in state.scored.values() if value >= good)
        worst_name = min(state.scored, key=state.scored.get)
        worst_label = LINE_LABELS.get(worst_name, worst_name)
        lines.append(
            f"lignes confirmées {seen}/{len(state.scored)}"
            f"   score {overall_score(state.scored):.2f}"
            f"   plus faible : {worst_label} {state.scored[worst_name]:.2f}"
        )

    if state.corners is not None:
        first = state.reference_names[0] if state.reference_names else "?"
        lines.append(
            f"orientation {state.orientation + 1}/4"
            f"   point 1 = {POINT_LABELS.get(first, first)}"
        )

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

    px_per_cm = float(court_config["px_per_cm"])
    grid_step_cm = float(court_config["grid_step_cm"])

    model = model_from_config(court_config)
    reference_names = tuple(court_config["reference_points"])
    orientations = symmetric_orders(reference_names)
    width_cm = model.doubles_width_cm
    height_cm = model.length_cm

    scoring = config["scoring"]
    good_score = float(scoring["good"])
    partial_score = float(scoring["partial"])
    line_colors = {
        "good": tuple(view_config["line_good_color"]),
        "partial": tuple(view_config["line_partial_color"]),
        "missing": tuple(view_config["line_missing_color"]),
    }

    cv2.setUseOptimized(True)

    state = AppState()
    state.reference_names = reference_names
    smoother = CornerSmoother(int(config["smoothing"]["corner_window"]))
    durations: deque[float] = deque(maxlen=int(view_config["fps_window"]))
    show_mask = bool(view_config["show_mask_on_start"])

    cv2.namedWindow(WINDOW_MAIN)
    cv2.setMouseCallback(WINDOW_MAIN, on_mouse, state)

    # Both windows otherwise open at the top left, and the top-down view lands
    # squarely on the banner that says which method fired and how stable the
    # corners are — the one thing needed to tell a good fit from a bad one.
    cv2.moveWindow(WINDOW_MAIN, 0, 0)
    top_window_placed = False

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
                    found, method, state.candidates = detect_quad(mask, quad_config)
                    if found is not None:
                        # Detection runs downscaled; bring the corners back to
                        # display pixels, which is also what a click gives.
                        state.corners = smoother.update(found / scale)
                        state.method = method
                    else:
                        state.corners = None
                        smoother.reset()

                court_corners = reference_corners(
                    model, orientations[state.orientation]
                )
                matrix = (
                    compute_homography(state.corners, court_corners)
                    if state.corners is not None
                    else None
                )

                if matrix is not None:
                    projected = project_model(model, matrix)
                    # The mask lives in the processing frame; the projection is
                    # in display pixels, hence the scale.
                    state.scored = score_lines(projected, mask, scale)
                else:
                    projected = {}
                    state.scored = {}

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

                if state.corners is None and state.candidates:
                    draw_line_candidates(
                        frame,
                        [
                            (start, end, f"{item.weight:.0f}")
                            for item, chord in (
                                (item, clip_line_to_frame(item, mask.shape))
                                for item in state.candidates
                            )
                            if chord is not None
                            for start, end in [
                                (
                                    (chord[0][0] / scale, chord[0][1] / scale),
                                    (chord[1][0] / scale, chord[1][1] / scale),
                                )
                            ]
                        ],
                        tuple(view_config["candidate_color"]),
                    )

                if projected:
                    draw_court_lines(
                        frame,
                        projected,
                        state.scored,
                        line_colors,
                        good_score,
                        partial_score,
                    )
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
                        good_score,
                    ),
                    tuple(view_config["hud_color"]),
                )
                cv2.imshow(WINDOW_MAIN, frame)
                if top_down is not None:
                    cv2.imshow(WINDOW_TOP, top_down)
                    if not top_window_placed:
                        cv2.moveWindow(WINDOW_TOP, 0, TOP_WINDOW_Y)
                        top_window_placed = True

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
                if key == ord("o"):
                    state.orientation = (state.orientation + 1) % len(orientations)
                    state.reference_names = orientations[state.orientation]
                    first = state.reference_names[0]
                    print(
                        f"Orientation {state.orientation + 1}/{len(orientations)} :"
                        f" le point 1 est « {POINT_LABELS.get(first, first)} »"
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
