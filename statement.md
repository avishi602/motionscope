# Problem Statement - MotionScope

## Problem statement
Reviewing surveillance or experimental video by eye is slow and error-prone.
Users need an automatic, repeatable way to find *what moved*, *where*, *how
fast*, and *in which direction* in a video recorded by a fixed camera, and to
keep a numeric record of it.

## Scope
- Input: a video file from a **static camera** (or the built-in synthetic demo).
- Processing: background-subtraction motion detection, dense optical flow,
  multi-object tracking with persistent IDs, and analytics.
- Output: annotated video, optical-flow video, per-object CSV, JSON summary,
  motion heatmap and activity timeline.
- Fully command-line; no GUI and no internet needed.
- Out of scope: moving cameras, object classification (what the object *is*),
  and live webcam streaming.

## Target users
- Students and researchers learning classical computer-vision techniques.
- Small teams who need quick motion statistics from fixed-camera footage
  (e.g. a shop entrance, a corridor, a lab experiment).

## High-level features
1. **Motion detection** - MOG2 background subtraction with noise clean-up.
2. **Optical flow** - Farneback dense flow, colour-coded video, speed and
   compass direction per frame.
3. **Object tracking** - centroid tracker with unique IDs and trails.
4. **Analytics and reports** - heatmap, timeline chart, CSV and JSON.
5. **Synthetic demo generator** - runs out of the box with no data download.
6. **Validation, logging and tests** - clear errors, log file, 18 unit tests.
