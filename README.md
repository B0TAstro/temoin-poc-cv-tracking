# Ball tracking POC

First building block of the **Témoin** project: track a ball in real time, by
its colour, **without any AI**. Computer vision and arithmetic only, so the
algorithm can later run on an iPhone.

This POC answers a single question: is the resulting trajectory clean enough to
derive bounces from it? The vertical velocity shown on screen flips sign on every
impact.

## Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage
```bash
# 1. tune the colour thresholds on the ball, under the room's own light
python src/calibrate.py
# 2. track the ball from the webcam
python src/main.py
# 3. record a run so it can be replayed unchanged
python src/main.py --record data/samples/test.mov
# 4. replay that run after changing the thresholds
python src/main.py --source data/samples/test.mov
# the tests covering the pure functions
pytest tests/ -q
```

## Calibration
1. Run `python src/calibrate.py` with the ball in frame.
2. Raise **saturation min** until the background (white wall, floor) turns black.
   This is the most effective setting by far: white has almost no saturation.
3. Adjust **teinte min** and **teinte max** around the ball's hue. For the blue
   ball, expect 85-115, with the useful hue around 100.
4. Lower **valeur min** if shadowed parts of the ball drop out of the mask.
5. Raise **ouverture** if isolated specks remain, **fermeture** if the ball is
   punched through by its printed pattern or by a specular highlight.
6. Check that the counter reads **1 candidate** when only the ball is in frame.
7. Press `s` to write the values into `config.yaml`, then `q`.

Calibration also works on a recording: `python src/calibrate.py --source
data/samples/test.mov`. Playback holds on the last frame, which leaves time to
work the sliders calmly.

The slider labels are in French, since they are part of the on-screen surface:
`teinte` is hue, `saturation` is saturation, `valeur` is value, `flou` is blur,
`ouverture` and `fermeture` are the morphological opening and closing, `surface`
is contour area and `circularité` is circularity.

## Keys

| Key | In `main.py` | In `calibrate.py` |
|-----|--------------|-------------------|
| `q` | quit | quit without saving |
| `m` | show or hide the HSV mask window | — |
| `p` | pause | — |
| `s` | screenshot into `data/samples/` | save to `config.yaml` |

## What is displayed
- a circle around the ball — **yellow** when it is detected, **orange** when the
  position comes from the filter's prediction
- the trail of recent positions, fading with age
- the velocity vector, proportional to the real speed
- a banner with the frame rate, tracker state, position, radius, velocity in
  px/s, acceleration in px/s², and the **share of frames where the ball was
  detected**

## `config.yaml` parameters

`calibrate.py` rewrites this file, and the YAML round-trip does not preserve
comments — so the parameters are documented here instead.

### `capture`

| Key | Role |
|-----|------|
| `device_index` | camera index, `0` for the Mac's built-in webcam |
| `width`, `height` | resolution requested from the camera |
| `fps` | frame rate requested from the camera |
| `warmup_frames` | frames dropped on open: the Mac webcam returns black ones at first |
| `record_fourcc` | codec used by `--record` |

### `detection`

| Key | Role |
|-----|------|
| `processing_width` | width the detection works at; display stays at full resolution. **First lever when the frame rate is too low.** |
| `blur_ksize` | Gaussian blur before thresholding, cuts sensor noise (odd) |
| `open_ksize` | morphological opening: removes isolated specks |
| `close_ksize` | morphological closing: fills holes inside the ball |
| `hsv_ranges` | list of `{lower, upper}` bounds in HSV — **H from 0 to 179**, S and V from 0 to 255. A list rather than a single range so a hue wrapping around H=0, such as red, fits in two bands without touching the code. |
| `min_area`, `max_area` | contour area, in pixels of the **processing** frame, not of the display |
| `min_circularity` | `1.0` is a perfect disc. A rasterised disc scores around `0.9`; `0.6` is a permissive threshold that still accepts a partially occluded ball. |
| `score_weights` | how a candidate is chosen among several: relative area, circularity, closeness to the tracker's prediction |

### `tracking`

| Key | Role |
|-----|------|
| `process_noise` | trust in the constant-velocity model. Higher means the filter sticks to the measurement and smooths less. |
| `measurement_noise` | trust in the measurement. Higher means more smoothing but slower reaction. |
| `max_missed_frames` | frames carried on prediction alone before the ball is declared lost |
| `history_length` | number of timestamped positions kept |
| `derivative_window` | number of points used by the regression that yields velocity |

### `visualization`

| Key | Role |
|-----|------|
| `trail_length` | number of positions drawn in the trail |
| `velocity_scale` | arrow length per pixel/second |
| `fps_window` | number of frames the frame rate is averaged over |
| `show_mask_on_start` | open the mask window right away |
| `ball_color`, `predicted_color`, `trail_color`, `velocity_color`, `hud_color` | colours in **BGR**, not RGB — the OpenCV convention |

## Layout

```
src/
├── main.py           main loop: capture, detection, tracking, display
├── capture.py        webcam or video file behind one interface
├── config.py         reading and writing config.yaml
├── detection.py      HSV mask, cleanup, contours, candidate selection
├── tracking.py       Kalman filter, history, velocity and acceleration
├── visualization.py  drawing the ball, the trail, the velocity vector
└── calibrate.py      HSV threshold tuning with sliders
tests/
└── test_geometry.py  pure functions: circularity, derivatives, selection, mask
```

The functions in `detection.py`, `tracking.py` and the derivatives are pure and
stateless: that is what will let them be ported to Swift without dragging the
capture loop along. They are also the only thing under test — a sign error in a
derivative is invisible on screen.

## Performance

The target is 30 frames per second at 720p. Measured on a synthetic 1280×720
clip, processing at 640 px:

- detection and tracking alone: **~1 ms per frame**
- full loop including display: **~230 frames per second**

If the real webcam drops below 30 FPS, lowering `processing_width` in
`config.yaml` is enough — no code change.