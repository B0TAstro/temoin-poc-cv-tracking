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
| **POC 2 · court lines** | Place every court line from four known points, to a few centimetres | **In progress** |
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
│   ├── court_model.py         # the court as named points and lines, in cm (POC 2)
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

A court is not one rectangle, it is a network of lines whose dimensions are known
in advance. The POC exploits exactly that: **four known points are enough**. From
them comes the homography, and from the homography every remaining line follows
by projection — already named, which raw line detection could never give.

From the Mac's webcam the app must:

1. isolate the **white lines** taped on the floor
2. obtain **four reference points**, detected automatically or clicked by hand
3. compute the **homography** to the court model's real centimetres, declared in
   `configs/court.yaml` as four measurements plus a scale
4. project and draw **every line of the court**, each scored by how much of it
   the mask actually backs
5. let a **click** report a point's real position in centimetres

The question is accuracy, not detection: is a clicked point within a few
centimetres of its real position? If so, POC 3's in/out decision is viable.

### Out of scope

- No ball in this POC (putting the two together is POC 3)
- No bounce, no in or out
- **No AI or deep learning library.** A project constraint, not a preference.
- **No lens distortion correction.** A wide-angle webcam bows straight lines,
  which costs centimetres near the edges of the frame and is invisible to the
  eye on the overlay. Accepted as a known limitation, recorded in the README.

### Non-negotiable findings

These were measured, not assumed. Do not undo them.

- **The court's symmetry cannot be resolved from the image.** A tennis court is
  symmetric about the net and along its length, so four corner assignments map
  it onto itself. All four project every line onto the tape and score a perfect
  1.00, and three of them are out by most of the length of the court. Never use
  the score to choose the orientation; `reference_points` declares it, the `o`
  key cycles it, and only a click on a measured point can confirm it.
- **A colour threshold cannot tell white tape from a white wall.** In a real
  room it kept 59% of the frame and merged walls, door and tape into one blob.
  The top-hat in `enhance_lines` takes that to 4% and must stay enabled: it
  keeps only what is smaller and brighter than its surroundings, which a wall
  never is.
- **A quadrilateral must be backed by the mask.** Four lines can always be
  intersected into one, so shape alone will report a door frame crossed with a
  skirting board. `quad_support` checks each edge and the weakest decides.
- **Reprojection error is not a quality indicator.** With exactly four points
  `getPerspectiveTransform` is an exact fit, so it is always about zero. Corner
  stability and the per-line model score are the real ones.
- **Accuracy falls off towards the far end**, where the court occupies fewer
  pixels. Measure there, not only at the near baseline.
- **A validation marker must be flat.** The homography maps the ground plane, so
  the top of a tall object projects to the wrong place. The same effect is the
  main accuracy problem waiting in POC 3: the ball's centre sits one radius
  above the ground at the bounce.

### Expected display

- The video with every model line drawn, coloured by its score, the four
  reference points numbered, and the current mode
- A second window with the rectified top-down view and a grid every 10 cm
- A click reports a position in centimetres and marks it on the top-down view
- Keys: `m` white mask, `c` manual calibration, `o` cycle orientation,
  `r` reset, `l` lock the corners, `q` quit

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

1. **Finish POC 2**: find the four reference points automatically among the
   candidate lines, instead of relying on a closed quadrilateral or a click.
   Detect line pixels structurally rather than by colour — bright with darker
   pixels on both sides, which a wall is not — run Hough over them, then search
   the correspondences and keep the homography that scores best. `segment_support`
   is already that scoring function.
2. **POC 3**: project the ball's position onto the court at the moment of
   the bounce, detect the bounce from the trajectory's derivatives and the sound
   peak of the impact, and call it in or out with a confidence score built from
   the distance to the line and the error margin in centimetres
3. Marking the bounces on the replay timeline
4. Port to iPhone
