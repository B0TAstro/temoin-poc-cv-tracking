# Témoin — computer vision POCs

Building blocks for **Témoin**, an app that helps amateur tennis players settle a
disputed point without an umpire. Phones at the back of the court film every
point; a shared replay must eventually say whether the ball was in or out, with
a confidence score.

Everything here is done **without AI**: OpenCV and arithmetic only, so the
algorithms can later run on an iPhone.

| POC | Question it answers | Status |
| --- | --- | --- |
| **1 · ball tracking** | Can a ball be followed in real time, cleanly enough to derive bounces from? | Done, tag `poc1-ball-tracking` |
| **2 · court lines** | Once a floor rectangle is rectified, is a clicked point accurate to a few centimetres? | In progress |
| 3 · bounce in/out | Both together: was the bounce in or out? | Upcoming |

> Code, comments and documentation are in English. Everything displayed on screen
> or printed to the terminal is in French: that is the user-facing surface.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On macOS the terminal needs camera access:
**System Settings > Privacy & Security > Camera**. Without it, opening the
webcam fails with an explicit message.

Apps are run **as modules from the repo root**:

```bash
python -m apps.poc1_ball_tracking
python -m apps.poc2_court_lines
python -m apps.calibrate_hsv --config configs/court.yaml
pytest tests/ -q
```

Running one by path (`python apps/poc2_court_lines.py`) raises
`ModuleNotFoundError`: Python puts the script's own directory on `sys.path`
rather than the repo root.

## POC 1 · ball tracking

```bash
python -m apps.calibrate_hsv                                    # tune the ball colour
python -m apps.poc1_ball_tracking                               # track from the webcam
python -m apps.poc1_ball_tracking --record data/samples/test.mov  # and save raw frames
python -m apps.poc1_ball_tracking --source data/samples/test.mov  # replay that run
```

`--record` saves the **raw frames**, without the overlay. That is what makes it
possible to push the same sequence back through the detection with different
settings and compare two configurations on identical frames, instead of
comparing two different throws.

### What is displayed

- a circle around the ball — **yellow** when detected, **orange** when the
  position comes from the filter's prediction
- the trail of recent positions, fading with age
- the velocity vector, proportional to the real speed
- a banner with the frame rate, tracker state, position, radius, velocity in
  px/s, acceleration in px/s², and the **share of frames where the ball was
  detected**

The vertical velocity flips sign on every impact. That is the signal POC 3 will
build its bounce detection on, and it is why this POC displays derivatives at
all.

### Keys

| Key | Effect |
|-----|--------|
| `q` | quit |
| `m` | show or hide the HSV mask window |
| `p` | pause |
| `s` | screenshot into `data/samples/` |

## POC 2 · court lines

Lay out a rectangle of white tape on the floor, measure it, write the dimensions
into `configs/court.yaml`, then:

```bash
python -m apps.calibrate_hsv --config configs/court.yaml   # tune the white threshold
python -m apps.poc2_court_lines                            # detect and rectify
python -m apps.poc2_court_lines --record data/samples/court.mov
python -m apps.poc2_court_lines --source data/samples/court.mov
```

**Measure the rectangle to the outer edge of the tape.** Contours are traced
around the outside of the taped lines, so the detected quadrilateral follows
that edge. Measuring to the centreline instead turns half the tape width into a
systematic error — about 1 cm of the 1.3 cm seen in the measurements below.

### What is displayed

- the video with the quadrilateral and its four numbered corners, and the
  current mode: automatic, manual calibration, or locked
- a second window with the rectified **top-down view** and a grid every 10 cm
- clicking the image prints the point's real position in centimetres and marks
  it on the top-down view
- a banner with the frame rate, the mode, which method found the rectangle
  (contours or the Hough fallback), the **corner stability in pixels**, and the
  configured dimensions

Corner stability is the quality indicator here. Reprojection error cannot be
one: with exactly four points `getPerspectiveTransform` is an exact fit, so the
error is always about zero and measures nothing. A corner that dances by several
pixels between frames is what turns into centimetres of doubt in POC 3.

### Keys

| Key | Effect |
|-----|--------|
| `q` | quit |
| `m` | show or hide the white mask window |
| `c` | manual calibration: click the four corners in the order shown |
| `r` | reset to automatic detection |
| `l` | lock or unlock the current corners |
| `s` | screenshot into `data/samples/` |

Manual calibration exists because the lines of an amateur court are often worn
away. It replaces automatic detection until `r` resets it.

### Calibration procedure

1. `python -m apps.calibrate_hsv --config configs/court.yaml`, tape in frame.
2. Lower **valeur min** until the tape is white on the mask and the floor is
   black. White is low saturation and high value, so **saturation max** is the
   other lever: lower it if a coloured floor leaks in.
3. If the floor is pale and the lines barely separate, raise `tophat_ksize` in
   the config. A top-hat keeps what is smaller and brighter than its
   surroundings, which is exactly a strip of tape on a pale floor.
4. Save with `s`, then run `python -m apps.poc2_court_lines`.
5. If the rectangle is not found, press `c` and click the four corners by hand.

## Measured results

On synthetic clips, drawn through a known court-to-image homography so the real
centimetre position of every marker is known by construction:

| | POC 1 | POC 2, continuous lines | POC 2, broken lines |
|---|---|---|---|
| Detection rate | 100 % | 100 % (contours) | 100 % (Hough fallback) |
| Processing per frame | ~1.0 ms | ~0.9 ms | ~1.6 ms |
| Full loop with display | ~230 FPS | — | — |
| Corner stability | — | 1.6 px | 0.7 px |
| Mean error | — | **1.32 cm** | **0.85 cm** |
| Worst error | — | 1.96 cm | 1.07 cm |

The Hough fallback is the more accurate of the two, because `fitLine` averages
over the whole line and lands on its centreline, while the contour method
follows the outer edge of the tape.

**These numbers come from synthetic frames**: no lens distortion, no sensor
noise, no uneven lighting. Treat them as an upper bound on what a real webcam
will do, not as a measurement of it.

## Known limitations

- **Lens distortion is not corrected.** A wide-angle webcam bows straight lines.
  Near the edges of the frame that alone costs centimetres, and it is invisible
  to the eye on the overlay. Test the accuracy near a frame edge, not only in
  the middle.
- **Anything with height lands off target.** The homography maps the ground
  plane, so the top of a tall object projects to the wrong place. A validation
  marker must be flat: a cross of tape, a coin. This is the main accuracy
  problem waiting in POC 3, because the ball's centre sits one radius above the
  ground at the bounce.
- **The Hough fallback is permissive by design.** It exists to recover broken
  lines, so in a cluttered scene it can intersect the wrong lines into a
  plausible rectangle. The banner names the method that fired; if it says Hough
  when the lines are intact, distrust the result.
- **The white mask has no idea what a court is.** A white wall, a radiator or a
  sheet of paper in frame are all candidates. The area and convexity filters
  carry the whole burden of telling them apart.

## `configs/` reference

`calibrate_hsv.py` rewrites these files, and the YAML round-trip does not
preserve comments — so the parameters are documented here instead.

Both configs share the same `detection` section, which is what lets one
calibration tool serve them both: white is an ordinary HSV range, with any hue,
a low saturation ceiling and a high value floor.

### Shared · `capture`

| Key | Role |
|-----|------|
| `device_index` | camera index, `0` for the Mac's built-in webcam |
| `width`, `height` | resolution requested from the camera |
| `fps` | frame rate requested from the camera |
| `warmup_frames` | frames dropped on open: the Mac webcam returns black ones at first |
| `record_fourcc` | codec used by `--record` |

### Shared · `detection`

| Key | Role |
|-----|------|
| `processing_width` | width the detection works at; display stays at full resolution. **First lever when the frame rate is too low.** |
| `blur_ksize` | Gaussian blur before thresholding, cuts sensor noise (odd) |
| `open_ksize` | morphological opening: removes isolated specks |
| `close_ksize` | morphological closing: fills holes inside the ball or gaps in a line |
| `hsv_ranges` | list of `{lower, upper}` bounds — **H from 0 to 179**, S and V from 0 to 255. A list rather than a single range so a hue wrapping around H=0, such as red, fits in two bands without touching the code. |
| `tophat_ksize` | POC 2 only: top-hat kernel that lifts thin bright lines off a pale floor. `0` disables it. |
| `min_area`, `max_area` | POC 1 only: contour area, in pixels of the **processing** frame |
| `min_circularity` | POC 1 only: `1.0` is a perfect disc. A rasterised disc scores around `0.9`; `0.6` still accepts a partially occluded ball. |
| `score_weights` | POC 1 only: how a candidate is chosen among several — relative area, circularity, closeness to the tracker's prediction |

### `ball.yaml` · `tracking`

| Key | Role |
|-----|------|
| `process_noise` | trust in the constant-velocity model. Higher means the filter sticks to the measurement and smooths less. |
| `measurement_noise` | trust in the measurement. Higher means more smoothing but slower reaction. |
| `max_missed_frames` | frames carried on prediction alone before the ball is declared lost |
| `history_length` | number of timestamped positions kept |
| `derivative_window` | number of points used by the regression that yields velocity |

### `court.yaml` · `quad`, `court`, `smoothing`

| Key | Role |
|-----|------|
| `quad.method` | `contour` tries contours first and falls back to Hough; `hough` goes straight to the fallback |
| `quad.min_area_ratio` | smallest acceptable rectangle, as a share of the frame area |
| `quad.epsilon_ratio` | `approxPolyDP` tolerance, as a share of the contour perimeter. Raise it if the rectangle never simplifies to four points. |
| `quad.hough` | `Canny` thresholds and `HoughLinesP` parameters for the fallback. Deliberately permissive: its job is to work on degraded lines. |
| `court.width_cm`, `court.height_cm` | the real rectangle, measured to the **outer** edge of the tape |
| `court.px_per_cm` | resolution of the top-down view |
| `court.grid_step_cm` | spacing of the grid drawn on the top-down view |
| `smoothing.corner_window` | frames the corners are averaged over, and the window the stability figure is computed on |

### Shared · `visualization`

Colours are in **BGR**, not RGB — the OpenCV convention. `fps_window` is the
number of frames the frame rate is averaged over; `show_mask_on_start` opens the
mask window right away.

## Layout

```
configs/            ball.yaml, court.yaml
temoin_cv/          shared core, imported by every POC
├── capture.py           webcam or video file behind one interface
├── config.py            reading and writing the configs
├── masking.py           HSV mask and downscaling, shared by both POCs
├── ball_detection.py    contours, centre and radius of the ball
├── tracking.py          Kalman filter, history, velocity, acceleration
├── court.py             line detection, corner ordering, homography
└── visualization.py     shared drawing functions
apps/               poc1_ball_tracking.py, poc2_court_lines.py, calibrate_hsv.py
tests/              the core's pure functions only
```

The core depends on no POC; the apps import only the core. The functions in
`masking.py`, `ball_detection.py`, `tracking.py` and `court.py` are pure and
stateless, which is what will let them be ported to Swift without dragging the
capture loop along. They are also the only thing under test: a sign error in a
derivative, or a corner order that silently mirrors the top-down view, is
invisible on screen.

## Out of scope

No bounce detection on screen, no in or out decision, no sound, no second
camera, and **no AI library** — that is a project constraint, not a preference.
