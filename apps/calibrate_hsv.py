"""HSV threshold tuning tool, shared by every POC.

Shows the frame next to its mask with live sliders. Press `s` to write the
values back into the config file, `q` to quit without saving.

White court lines are an ordinary HSV range (any hue, low saturation, high
value), so the same tool tunes configs/court.yaml as well as configs/ball.yaml.

Code and comments are in English; the sliders and the on-screen text are in
French, since that is the user-facing surface.

    python -m apps.calibrate_hsv
    python -m apps.calibrate_hsv --config configs/court.yaml
    python -m apps.calibrate_hsv --source data/samples/test.mov
"""
from __future__ import annotations

import argparse
from typing import NamedTuple

import cv2
import numpy as np

from temoin_cv.ball_detection import find_candidates
from temoin_cv.capture import VideoSource
from temoin_cv.config import config_path, load_config, save_config
from temoin_cv.masking import build_mask, resize_for_processing
from temoin_cv.visualization import draw_hud, mask_to_bgr

WINDOW = "Témoin — calibrage"
DEFAULT_CONFIG = "ball.yaml"

# Area and circularity filter the ball's blob; a court rectangle has no use for
# them, so court.yaml leaves them out and they fall back to these.
BALL_KEYS = ("min_area", "max_area", "min_circularity")
BALL_DEFAULTS = {"min_area": 150, "max_area": 120_000, "min_circularity": 0.6}
AREA_STEP = 1000  # the max-area slider counts in thousands of pixels


class Slider(NamedTuple):
    """One trackbar: an English key for the code, a French label on screen."""

    key: str
    label: str
    maximum: int


SLIDERS = (
    Slider("h_min", "teinte min", 179),
    Slider("h_max", "teinte max", 179),
    Slider("s_min", "saturation min", 255),
    Slider("s_max", "saturation max", 255),
    Slider("v_min", "valeur min", 255),
    Slider("v_max", "valeur max", 255),
    Slider("blur", "flou", 31),
    Slider("open", "ouverture", 31),
    Slider("close", "fermeture", 31),
    Slider("min_area", "surface min", 5000),
    Slider("max_area", "surface max /1k", 500),
    Slider("min_circularity", "circularite x100", 100),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Réglage des seuils de couleur HSV")
    parser.add_argument(
        "--source", default=None, help="Vidéo à utiliser au lieu de la webcam"
    )
    parser.add_argument(
        "--config",
        default=str(config_path(DEFAULT_CONFIG)),
        help="Chemin du fichier de configuration YAML",
    )
    return parser.parse_args()


def _noop(_: int) -> None:
    """createTrackbar wants a callback; the values are polled in the loop."""


def initial_values(config: dict) -> dict[str, int]:
    """Slider positions matching the config currently on disk."""
    detection = config["detection"]
    lower = detection["hsv_ranges"][0]["lower"]
    upper = detection["hsv_ranges"][0]["upper"]

    return {
        "h_min": lower[0],
        "h_max": upper[0],
        "s_min": lower[1],
        "s_max": upper[1],
        "v_min": lower[2],
        "v_max": upper[2],
        "blur": int(detection["blur_ksize"]),
        "open": int(detection["open_ksize"]),
        "close": int(detection["close_ksize"]),
        "min_area": int(detection.get("min_area", BALL_DEFAULTS["min_area"])),
        "max_area": int(detection.get("max_area", BALL_DEFAULTS["max_area"]))
        // AREA_STEP,
        "min_circularity": int(
            float(detection.get("min_circularity", BALL_DEFAULTS["min_circularity"]))
            * 100
        ),
    }


def create_trackbars(config: dict) -> None:
    start = initial_values(config)
    for slider in SLIDERS:
        position = min(slider.maximum, max(0, start[slider.key]))
        cv2.createTrackbar(slider.label, WINDOW, position, slider.maximum, _noop)


def read_trackbars() -> dict[str, int]:
    return {
        slider.key: cv2.getTrackbarPos(slider.label, WINDOW) for slider in SLIDERS
    }


def hsv_bounds(values: dict[str, int]) -> tuple[np.ndarray, np.ndarray]:
    """Current slider positions as inRange bounds."""
    return (
        np.array([values["h_min"], values["s_min"], values["v_min"]], dtype=np.uint8),
        np.array([values["h_max"], values["s_max"], values["v_max"]], dtype=np.uint8),
    )


def apply_to_config(config: dict, values: dict[str, int]) -> None:
    """Copy the slider values back into the config dict, in place."""
    detection = config["detection"]
    detection["hsv_ranges"][0]["lower"] = [
        values["h_min"],
        values["s_min"],
        values["v_min"],
    ]
    detection["hsv_ranges"][0]["upper"] = [
        values["h_max"],
        values["s_max"],
        values["v_max"],
    ]
    detection["blur_ksize"] = values["blur"]
    detection["open_ksize"] = values["open"]
    detection["close_ksize"] = values["close"]
    # Only write the ball filters back into a config that already had them, so
    # tuning the court's white does not sprinkle meaningless keys into it.
    if all(key in detection for key in BALL_KEYS):
        detection["min_area"] = values["min_area"]
        detection["max_area"] = values["max_area"] * AREA_STEP
        detection["min_circularity"] = values["min_circularity"] / 100.0


def largest_shape_area(mask: np.ndarray) -> float:
    """Area of the biggest blob, whatever its shape.

    Unlike the ball candidate count, this says something for a court rectangle
    too, so it is the readout that works for both configs.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return max((float(cv2.contourArea(contour)) for contour in contours), default=0.0)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    processing_width = int(config["detection"]["processing_width"])
    tunes_ball = all(key in config["detection"] for key in BALL_KEYS)

    if len(config["detection"]["hsv_ranges"]) > 1:
        print("Note : seule la première plage HSV est réglée ici.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    create_trackbars(config)

    with VideoSource.from_config(config, args.source) as source:
        key_delay = max(1, int(1000 / source.fps)) if source.is_file else 1
        last_frame = None

        while True:
            ok, frame, _ = source.read()
            if not ok or frame is None:
                if source.is_file and last_frame is not None:
                    # Hold on the last frame so the sliders can be worked on
                    # calmly once the recording has played out.
                    frame = last_frame.copy()
                else:
                    break
            else:
                last_frame = frame.copy()

            values = read_trackbars()
            small, _ = resize_for_processing(frame, processing_width)

            mask = build_mask(
                small, [hsv_bounds(values)], values["blur"], values["open"], values["close"]
            )
            candidates = find_candidates(
                mask,
                float(values["min_area"]),
                float(values["max_area"] * AREA_STEP),
                values["min_circularity"] / 100.0,
            )

            coverage = float(mask.mean()) / 255.0 * 100.0
            lines = [
                f"blanc retenu {coverage:5.1f} %"
                f"   plus grande forme {largest_shape_area(mask):7.0f} px²",
            ]
            if tunes_ball:
                lines.append(f"{len(candidates)} candidat(s) balle")
            lines.append("s enregistrer dans la config   q quitter")

            preview = np.hstack((small, mask_to_bgr(mask)))
            draw_hud(preview, lines, (255, 255, 255))
            cv2.imshow(WINDOW, preview)

            key = cv2.waitKey(key_delay) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                apply_to_config(config, values)
                path = save_config(config, args.config)
                print(f"Réglages enregistrés dans {path}")

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
