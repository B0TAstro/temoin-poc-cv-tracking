# Computer vision POCs

Building blocks for **Témoin**, an app that helps amateur tennis players settle a
disputed point without an umpire. Phones at the back of the court film every
point; a shared replay must eventually say whether the ball was in or out, with
a confidence score.

Everything here is done **without AI**: OpenCV and arithmetic only, so the
algorithms can later run on an iPhone.

| POC | Question it answers | Status |
| --- | --- | --- |
| **1 · ball tracking** | Can a ball be followed in real time, cleanly enough to derive bounces from? | Done, tag `poc1-ball-tracking` |
| **2 · court lines** | From four known points, can every court line be placed to a few centimetres? | In progress |
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

A court is not one rectangle, it is a network of lines whose dimensions are
known in advance. That is what the POC exploits: **four known points are enough**.
From them comes the homography, and from the homography every remaining line
follows by projection — already named, which raw line detection could never give.

`configs/court.yaml` holds the court as four measurements plus a scale, so the
full-size ITF dimensions can stay in the file while a scaled copy is what is
actually taped on the floor.

```bash
python -m apps.calibrate_hsv --config configs/court.yaml   # tune the white threshold
python -m apps.poc2_court_lines                            # detect and rectify
python -m apps.poc2_court_lines --record data/samples/court.mov
python -m apps.poc2_court_lines --source data/samples/court.mov
```

### What is displayed

- **every line of the court**, projected from the model and coloured by how much
  of it the mask actually backs: green confirmed, orange partial, grey claimed
  but unconfirmed
- the four reference points, numbered in the order they are configured
- a second window with the rectified **top-down view** and a grid every 10 cm
- clicking the image prints the point's real position in centimetres and marks
  it on the top-down view
- a banner with the frame rate, the mode, which method found the corners, the
  **per-line score**, the weakest line, the corner stability in pixels and the
  **assumed orientation**

### Keys

| Key | Effect |
|-----|--------|
| `q` | quit |
| `m` | show or hide the white mask window |
| `c` | manual calibration: click the four reference points in the order shown |
| `o` | cycle the orientation — see the symmetry trap below |
| `r` | reset to automatic detection |
| `l` | lock or unlock the current corners |
| `s` | screenshot into `data/samples/` |

Manual calibration exists because the lines of an amateur court are often worn
away, and because the outer corners are frequently out of frame. `reference_points`
in the config names which four model points you click, so a camera that only
sees the service boxes can calibrate on those instead.

### The symmetry trap

**This is the one thing to understand before trusting a coordinate.**

A tennis court is symmetric about the net *and* along its length. Four different
corner assignments therefore map it exactly onto itself. All four project all ten
lines onto the tape, all four score a perfect 1.00, and all four look flawless on
screen — but three of them are out by most of the length of the court.

No image can resolve this. It takes knowing which end the camera stands at.

So `reference_points` must be listed **clockwise, starting from the point that
appears nearest the top-left of the image**. If a clicked point reads a position
you know is wrong, press `o` to cycle the four assignments until it reads right.
The score will not help you: it is 1.00 for all four.

The corollary matters more than the rule: **never use the score to pick the
orientation, and validate with a flat marker at a measured position.** That click
is the only honest check.

### Calibration procedure

1. `python -m apps.calibrate_hsv --config configs/court.yaml`, tape in frame.
2. Lower **valeur min** until the tape is white on the mask and the floor is
   black. White is low saturation and high value, so **saturation max** is the
   other lever: lower it if a coloured floor leaks in.
3. If the room has white walls, keep `tophat_ksize` above zero. A top-hat keeps
   only what is smaller and brighter than its surroundings, which a wall never
   is and a strip of tape always is. On one real room it took the mask from 59%
   of the frame down to 4%.
4. Save with `s`, then run `python -m apps.poc2_court_lines`.
5. If the court is not found, press `c` and click the four reference points.
6. Click a point whose real position you measured, and check the reading. If it
   is mirrored, press `o`.

## Measured results

On synthetic clips, drawn through a known court-to-image homography so the real
centimetre position of every model point is known by construction. The POC 2
scene includes white walls and a bright door, as a real room does.

| | POC 1 | POC 2 · full court |
|---|---|---|
| Detection rate | 100 % | 100 % (contours) |
| Processing per frame | ~1.0 ms | ~1.0 ms |
| Full loop with display | ~230 FPS | — |
| Corner stability | — | 1.1 px |
| Model score | — | 0.99, 10/10 lines confirmed |
| Mean position error | — | **3.77 cm** |
| Worst position error | — | 6.68 cm |

That is about 1 % of the court's length. The error is not uniform: it reaches
6.7 cm at the far corner and stays near 2 cm at the near one, because the far end
occupies far fewer pixels and each one is therefore worth more centimetres. A
bounce near the far baseline is inherently less certain than one at your feet,
and POC 3's confidence score has to say so.

**These numbers come from synthetic frames**: no lens distortion, no sensor
noise, no uneven lighting. Treat them as an upper bound on what a real webcam
will do, not as a measurement of it.

## Known limitations

- **The court's symmetry cannot be resolved from the image.** Four corner
  assignments map the court onto itself; all of them score 1.00 and only one
  gives the right coordinates. See the symmetry trap above. This is a property
  of the court, not a bug to be fixed.
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
- **Accuracy falls off towards the far end**, where the court occupies fewer
  pixels. Measure there, not only at your feet.
- **The white mask has no idea what a court is.** A white wall, a radiator or a
  sheet of paper in frame are all candidates. The top-hat removes broad bright
  areas, and mask support rejects a rectangle made of unrelated edges, but
  nothing here knows what a court looks like — that is what the model score is
  for, and it only applies once four corners are in hand.
- **The corners still have to come from somewhere.** Automatic detection finds
  the outer quadrilateral when it is closed and in frame; otherwise the four
  points are clicked. Searching for them automatically among candidate lines is
  the next step.

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

### `court.yaml` · `quad`, `court`, `scoring`, `smoothing`

| Key | Role |
|-----|------|
| `quad.method` | `contour` tries contours first and falls back to Hough; `hough` goes straight to the fallback |
| `quad.min_area_ratio` | smallest acceptable rectangle, as a share of the frame area |
| `quad.epsilon_ratio` | `approxPolyDP` tolerance, as a share of the contour perimeter |
| `quad.min_support` | share of each edge that must actually lie on mask pixels. The weakest edge decides, which is what stops four unrelated lines being reported as a court. |
| `quad.hough` | `Canny` thresholds and `HoughLinesP` parameters for the fallback. Deliberately permissive: its job is to work on degraded lines. |
| `court.length_cm`, `court.doubles_width_cm`, `court.singles_width_cm`, `court.service_line_from_net_cm` | the four measurements the whole model is derived from. Defaults are the full-size ITF dimensions. |
| `court.scale` | `1` for a real court; `0.15` for a scaled copy taped on a floor. Applies to every dimension at once. |
| `court.reference_points` | which four model points are matched to the detected or clicked corners, **clockwise from the one nearest the top-left of the image**. See the symmetry trap. |
| `court.px_per_cm` | resolution of the top-down view |
| `court.grid_step_cm` | spacing of the grid drawn on the top-down view |
| `scoring.good`, `scoring.partial` | thresholds that colour a line green, orange or grey, and decide what counts as a confirmed line |
| `smoothing.corner_window` | frames the corners are averaged over, and the window the stability figure is computed on |

The model's point names, usable in `reference_points`: `doubles_near_left`,
`doubles_near_right`, `doubles_far_right`, `doubles_far_left`,
`singles_near_left`, `singles_near_right`, `singles_far_left`,
`singles_far_right`, `service_near_left`, `service_near_right`,
`service_far_left`, `service_far_right`, `centre_near`, `centre_far`,
`net_left`, `net_right`.

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
├── court_model.py       the court as named points and lines, in centimetres
└── visualization.py     shared drawing functions
apps/               poc1_ball_tracking.py, poc2_court_lines.py, calibrate_hsv.py
tests/              the core's pure functions only
```

The core depends on no POC; the apps import only the core. The functions in
`masking.py`, `ball_detection.py`, `tracking.py`, `court.py` and
`court_model.py` are pure and
stateless, which is what will let them be ported to Swift without dragging the
capture loop along. They are also the only thing under test: a sign error in a
derivative, or a corner order that silently mirrors the top-down view, is
invisible on screen.

## Out of scope

No bounce detection on screen, no in or out decision, no sound, no second
camera, and **no AI library** — that is a project constraint, not a preference.
