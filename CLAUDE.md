# CLAUDE.md · temoin-poc-cv-tracking

## Context

This repo is a **POC** for **Témoin**, a mobile app that helps amateur tennis
players settle a disputed point without an umpire: phones placed at the back of
the court film every point, and a shared replay lets both players look at the
same image.

Eventually the app must locate the ball's **bounces** in the replay and say
whether it was in or out, with a confidence score. The technical choice is to do
this **without AI**, using only computer vision and arithmetic, so that it runs
fast on an iPhone.

This POC is the first building block: **tracking a ball in real time**.

## What the POC does

1. opens the Mac's webcam
2. detects a ball by its **colour**
3. **tracks it** frame by frame
4. displays the live position, the recent trajectory and the speed

## Out of scope, for now

- No bounce detection shown on screen (the trajectory only has to make it
  computable later)
- No court, no lines, no homography
- No sound
- No second camera, no Bluetooth
- No interface beyond the OpenCV windows
- **No AI or deep learning library** (no YOLO, TensorFlow, PyTorch, MediaPipe…).
  This is a project constraint, not a preference.

## Stack

- Python 3.11+
- `opencv-python` for capture, image processing and display
- `numpy` for the maths
- `pyyaml` for the configuration

See `README.md` for the file layout, the calibration procedure and the full
`config.yaml` reference.

## Non-negotiable technical choices

- **Threshold in HSV, never in RGB.** An RGB threshold falls apart as soon as the
  room light changes.
- **Constant-velocity Kalman filter** (state: x, y, vx, vy) to smooth the
  position and predict the next one. When the ball is missing from a frame, carry
  on with the prediction for a few frames before declaring it lost.
- **Keep a timestamped history** of recent positions. Velocity and acceleration
  come from it, in pixels per second. These derivatives are what the next POC
  needs: a bounce is a sharp sign change in the vertical velocity.
- **Frame rate over cleverness.** Per-frame processing stays simple and light;
  that is what will make the algorithm portable to iPhone.

## Commands

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/calibrate.py                             # tune the colour thresholds
python src/main.py                                  # track from the webcam
python src/main.py --source data/samples/test.mov   # track from a recording
pytest tests/ -q                                    # test the pure functions
```

On macOS the terminal needs camera access: System Settings > Privacy & Security
> Camera.

## Performance target

At least **30 frames per second** at 720p on a recent Mac. `processing_width` in
`config.yaml` is the lever: the detection runs on a downscaled frame while the
display stays at full resolution.

## Conventions

- **Code, identifiers, comments and documentation in English.** Everything shown
  on screen or printed to the terminal is **in French**: HUD, window titles,
  slider labels, key hints, CLI help and error messages. Keep that boundary
  explicit — map an English enum to a French label at the display layer rather
  than naming things in French.
- Type hints and small pure functions, easy to port to Swift later.
- Every tunable parameter lives in `config.yaml`; no magic numbers in the code.
- The video source (webcam or file) must stay interchangeable, so the same test
  can be replayed.
- Before adding a dependency, check that it is not an AI library.

## Planned next steps

1. Bounce detection from the trajectory's derivatives, confirmed by the sound
   peak of the impact
2. Court calibration from its four corners, and homography to real-world
   dimensions
3. In or out decision at the bounce, with a confidence score built from the
   distance to the line and the pixel error margin
4. Marking the bounces on the replay timeline
5. Port to iPhone
