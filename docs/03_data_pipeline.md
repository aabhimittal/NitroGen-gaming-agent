# 3 · Building a video-action dataset from controller overlays

**Code:** [`nitrogen/data/overlay_extraction.py`](../nitrogen/data/overlay_extraction.py),
[`nitrogen/data/synthetic.py`](../nitrogen/data/synthetic.py),
[`nitrogen/data/dataset.py`](../nitrogen/data/dataset.py)

## The core difficulty of behavior cloning at scale

Behavior cloning needs `(observation, action)` pairs. Observations are easy —
every gameplay video is a stream of frames. **Actions are the hard part**: the
raw video doesn't tell you which buttons the player pressed. Historically this
forced people to either instrument their own data collection (expensive, small)
or use an *inverse dynamics model* that infers actions from consecutive frames
(noisy, and it needs labeled data to bootstrap).

## NitroGen's insight: the actions are often *already on screen*

A large slice of gameplay footage — speedruns, tutorials, "gamepad viewer"
streams — renders a **live controller overlay** in a corner showing exactly which
buttons are held and where the sticks are pushed. That overlay is a
human-readable recording of the ground-truth action. NitroGen turns it back into
labels with a three-stage pipeline:

1. **Localize** the overlay. Controller-viewer widgets come in ~hundreds of
   visual styles, so the paper matches **SIFT / XFeat keypoints** against a
   library of ~300 controller templates to find and rectify the overlay in each
   frame.
2. **Read** the controls. A **hybrid classification + segmentation network** (a
   fine-tuned **SegFormer**) regresses the analog stick positions and classifies
   each button as pressed/released.
3. **Filter** by confidence and quality. Reported extraction quality:
   **joystick R² = 0.84**, **button accuracy = 0.96**.

The result is an *internet-scale* labeled dataset — ~40,000 hours, 1,000+ games —
without ever instrumenting a single play session.

## What this repo implements

We can't ship a SegFormer or 40,000 hours of video, but the **concept is fully
reproducible and verifiable**. [`overlay_extraction.py`](../nitrogen/data/overlay_extraction.py)
provides both directions:

- `render_overlay(frame, action)` — the *forward* direction: paint a controller
  HUD (two stick wells with dots, a row of button lights) onto a frame, exactly
  what a gamepad-viewer overlay does live while a human plays.
- `OverlayExtractor.extract(frame)` — the *inverse* direction: locate the panel
  by its colour signature (our stand-in for keypoint/template matching), read the
  stick dot offsets back into `[-1, 1]`, and classify each button light.

Round-tripping random actions through render → extract reproduces the paper's
metric framing:

```bash
python -m scripts.generate_data --overlay-demo
# joystick R^2     : ~0.99   (paper: 0.84 on real, messy video)
# button accuracy  : ~1.00   (paper: 0.96)
```

Our synthetic numbers are near-perfect because the overlay is clean; the point is
the *mechanism* — recover a dense action label from pixels alone — not the exact
score.

## From videos to a training corpus

Since we don't have real internet footage to label, we generate an analogous
corpus by rolling out **scripted expert policies** inside the toy games
([ch 9](09_benchmark.md)) and recording `(frame, action)` pairs
([`synthetic.py`](../nitrogen/data/synthetic.py)). A little action noise is
injected so the data covers states slightly off the optimal path — the same
reason real human demonstrations clone more robustly than a perfectly
deterministic script.

## Single frame → action *chunk*

The final dataset step ([`dataset.py`](../nitrogen/data/dataset.py)) is where a
key NitroGen choice appears. Each training example is:

```
input :  frame_i                                     (ONE RGB image)
target:  [action_i, action_{i+1}, ..., action_{i+15}]  (a 16-step CHUNK)
```

The model conditions on a **single frame** (no frame stack, no history) and
predicts a **chunk** of the next 16 actions. Why chunks matter for temporal
coherence and inference cost is covered in [ch 8](08_inference.md).

Continue to [**4 · The SigLIP vision encoder →**](04_vision_encoder.md)
