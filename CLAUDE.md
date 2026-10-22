# CLAUDE.md · temoin-poc-cv-tracking

## Context

This repo holds the **computer vision POCs** of the **Témoin** project, a mobile
app that helps amateur tennis players settle a disputed point without an umpire.
Two phones at the back of the court, one on each side, film every point; when
the players disagree, each one shows the replay from its own camera.

Eventually the app must locate the ball's **bounces** and say whether it was in
or out, with a confidence score. The technical choice is to do this **without
AI**, using only OpenCV and arithmetic (colour, geometry, homography,
trajectory derivatives), so that it runs fast on an iPhone.

## The POCs in this repo

| POC | Goal | Status |
| --- | --- | --- |
| **POC 1 · ball tracking** | Detect a ball by its colour on the webcam and follow its motion | Done, frozen by the `poc1-ball-tracking` tag |
| **POC 2 · court lines** | Detect a rectangle of white lines on the floor and map it back to its real dimensions | **In progress** |
| POC 3 · bounce in/out | Put the two together: detect the bounce and say whether it was in or out | Upcoming |

## Repo layout

```
temoin-poc-cv-tracking/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── configs/
│   ├── ball.yaml              # HSV ranges, sizes, tracker parameters (POC 1)
│   └── court.yaml             # white thresholds, real rectangle dimensions (POC 2)
├── data/
│   └── samples/               # test videos and images
├── temoin_cv/                 # shared core, imported by every POC
│   ├── capture.py             # webcam or video file, one interface
│   ├── config.py              # reading and writing the YAML configs
│   ├── masking.py             # HSV mask and downscaling, shared by both POCs
│   ├── ball_detection.py      # contours, centre and radius of the ball (POC 1)
│   ├── tracking.py            # Kalman filter, history, velocity, acceleration
│   ├── court.py               # line detection, corners, homography (POC 2)
│   └── visualization.py       # shared drawing functions
├── apps/
│   ├── poc1_ball_tracking.py
│   ├── poc2_court_lines.py
│   └── calibrate_hsv.py       # colour threshold tuning with sliders
└── tests/                     # the pure functions of the core only
```

**Apps are run as modules from the repo root**, never by path:
`python -m apps.poc2_court_lines`. Running `python apps/poc2_court_lines.py`
raises `ImportError`, because Python puts the script's own directory on
`sys.path` rather than the repo root.

## POC 2 · court lines

### Goal

A Python app that, from the Mac's webcam:

1. detects a **rectangle of white lines** laid out on the floor (white tape or
   sheets of paper) standing in for the lines of a court
2. finds its **four corners**, even seen in perspective, where the rectangle
   appears as a trapezoid
3. computes the **homography** between the image and the **real dimensions** of
   the rectangle, declared in `configs/court.yaml`
4. displays a rectified **top-down view**
5. lets you **click a point in the image** to show its real position in
   centimetres inside the rectangle, to check the accuracy

The question this POC answers is accuracy, not detection: is the error between a
clicked point and its real position a matter of a few centimetres? If it is, the
in/out decision of POC 3 is viable.

### Out of scope

- No ball in this POC (putting the two together is POC 3)
- No bounce, no in or out
- No real tennis court: a rectangle of known size on the floor is enough
- **No AI or deep learning library.** A project constraint, not a preference.
- **No lens distortion correction.** A wide-angle webcam bows straight lines,
  which costs centimetres near the edges of the frame and is invisible to the
  eye on the overlay. Accepted as a known limitation and recorded in the README;
  a checkerboard calibration is a later problem.

### Detection pipeline

1. Blur the frame slightly to cut noise
2. Isolate **white**: in HSV, low saturation and high value (thresholds in
   `court.yaml`). White is an ordinary HSV range, so the mask reuses
   `temoin_cv/masking.py` unchanged. On a light floor, a top-hat transform helps
   thin lines stand out.
3. Clean the mask with morphological operations
4. **Main method**: contours, then `approxPolyDP`, keeping a convex
   quadrilateral of sufficient area
5. **Fallback**: `Canny` then `HoughLinesP`, group the segments into two
   families by angle, fit one line per family, intersect them
6. **Order the corners** stably (top-left, top-right, bottom-right, bottom-left)
   by angle around the centroid. Sorting by the sum and difference of the
   coordinates breaks as soon as the rectangle is seen at an angle, which is
   exactly the case here.
7. Smooth the corners over time (rolling mean) so they stop shaking frame to
   frame

### Manual calibration

The lines of an amateur court are often worn away, so it must always be possible
to **click the four corners by hand** (key `c`), in the order shown on screen.
Manual calibration replaces automatic detection until it is reset (key `r`).

### Homography

- Source points: the four corners in the image
- Destination points: the corners of the real rectangle, in centimetres, from
  `court.yaml`
- `cv2.getPerspectiveTransform` for the matrix, `cv2.perspectiveTransform` to
  project a point, `cv2.warpPerspective` for the top-down view
- Pure functions in `temoin_cv/court.py`: `image_to_court(point, matrix)` and
  `court_to_image(point, matrix)`, reused as they are in POC 3

**Reprojection error is not a quality indicator here.** With exactly four points
`getPerspectiveTransform` is an exact fit, so the error is always about zero and
measures nothing. The indicator to display is the **temporal stability of the
corners**, their standard deviation in pixels over the smoothing window, which
will also feed the confidence score of POC 3.

### Expected display

- The video with the lines and the four corners drawn on top, and an indicator
  of automatic detection versus manual calibration
- A second window with the rectified **top-down view** and a grid every 10 cm
- Clicking the image shows its real position in centimetres and marks it on the
  top-down view
- Key `m` for the white mask, `c` to calibrate by hand, `r` to reset, `l` to
  lock the corners, `q` to quit

### Validation

- The corners are found stably in under a second, from several viewing angles
- Clicking a point whose real position is known gives an error of **a few
  centimetres** at most. The marker must be **flat** — a cross of tape, a coin —
  never a tall object: the top of a tall object projects to the wrong place
  through a ground-plane homography, and the parallax would be blamed on the
  homography. The same effect is the main accuracy problem waiting in POC 3,
  since the ball's centre sits one radius above the ground at the bounce.
- Record in the README the conditions that defeat the detection: lighting, light
  floor, too shallow an angle

## Stack

- Python 3.11+
- `opencv-python`, `numpy`, `pyyaml`

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m apps.poc1_ball_tracking
python -m apps.poc2_court_lines
python -m apps.poc2_court_lines --source data/samples/court.mov
python -m apps.calibrate_hsv --config configs/court.yaml
pytest tests/ -q
```

On macOS the terminal needs camera access: System Settings > Privacy & Security
> Camera.

## Conventions

- **Code, identifiers, comments and documentation in English.** Everything shown
  on screen or printed to the terminal is **in French**: HUD, window titles,
  slider labels, key hints, CLI help and error messages. Keep that boundary
  explicit — map an English enum to a French label at the display layer rather
  than naming things in French.
- Type hints and small pure functions, easy to port to Swift later.
- Every tunable parameter lives in `configs/`; no magic numbers in the code.
- The `temoin_cv/` core depends on no POC; the apps import only the core.
- The video source (webcam or file) stays interchangeable, so the same test can
  be replayed.
- Only the core's pure functions are tested. A sign error in a derivative or a
  corner ordering is invisible on screen, which is what the tests are for.
- Before adding a dependency, check that it is not an AI library.
- When a POC works, tag it (`poc2-court-lines`, and so on).

## Planned next steps

1. **POC 3**: project the ball's position onto the rectangle at the moment of
   the bounce, detect the bounce from the trajectory's derivatives and the sound
   peak of the impact, and call it in or out with a confidence score built from
   the distance to the line and the error margin in centimetres
2. Marking the bounces on the replay timeline
3. Port to iPhone
