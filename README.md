# Webcam Gaze & Reading-Pattern Detector

A small computer-vision proof of concept. It uses a standard webcam to track
where the eyes are looking, with two goals:

1. Detect whether the person is **looking at the screen** (attention).
2. Detect whether the person is **reading** (a repeating left-to-right eye pattern).

This was a personal POC to get hands-on with real-time face tracking and signal
processing. It is not a finished product.

## What works and what doesn't

- **Attention detection works.** Telling whether someone is looking at the
  screen or away from it is reliable enough in normal lighting.
- **Reading detection does not work well.** The idea was to pick up the
  rhythmic eye movement of reading as a frequency signal. The signal turned out
  to be too noisy to be dependable — head motion, blinks, and camera resolution
  all leak into it. The detection pipeline is in place and documented below,
  but it should be read as an experiment, not a working feature.

## How it works

The pipeline goes from raw webcam frames to a single stable decision.

1. **Face & iris tracking.** [MediaPipe Face Mesh](https://developers.google.com/mediapipe)
   runs with `refine_landmarks=True`, which adds iris landmarks on top of the
   468-point face mesh.

2. **Geometric normalization.** The raw iris coordinates depend on head
   position and distance to the camera, so they can't be used directly.
   Instead, the iris centre is projected onto the vector between the inner and
   outer corners of the eye. That gives a position in `[-1, 1]` that is
   independent of head pose and scale. Left and right eyes are averaged into
   one signal.

3. **Signal processing.** The normalized position over time is treated as a
   1-D signal:
   - A 4th-order **Butterworth band-pass filter** (0.5–4 Hz) isolates the
     frequency range where reading-like eye movement would show up.
   - **Welch's method** estimates the power spectral density — more stable than
     a single FFT on a short window.
   - A **confidence score** is computed as the ratio of power inside the
     reading band to total power.

4. **Temporal smoothing.** A per-frame decision is noisy, so the final state
   uses **hysteresis**: the last 10 frames are buffered and the state only
   flips when 70% of them agree. This stops the readout from flickering.

## Running it

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python eye_reading_detector.py
```

Press `q` to quit, `r` to reset the history buffer.

For best results: decent lighting, face clearly visible, fairly stable head
position.

## Limitations

- Reading detection is unreliable (see above).
- Assumes a single face and a frontal pose.
- Sensitive to lighting and webcam quality.
- No calibration step; thresholds are tuned by hand.

## Tech stack

Python · OpenCV · MediaPipe · NumPy · SciPy

## License

MIT — see [LICENSE](LICENSE).
