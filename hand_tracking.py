"""Hand-tracking worker using MediaPipe Tasks API.

Runs webcam capture + MediaPipe hand-landmark detection on a background
thread so the main render loop stays at 60 fps.

Public interface
----------------
- ``CVWorker(model_path)``  — create and start the worker thread.
- ``worker.latest()``       — returns the most-recent ``FrameData`` (non-blocking).
- ``worker.stop()``         — signal the thread to stop; join.
"""

import threading
import time
import dataclasses
import pathlib
from typing import Optional

import cv2
import numpy as np

# MediaPipe Tasks API
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)


# ── Data containers ─────────────────────────────────────────────────

@dataclasses.dataclass
class HandData:
    """Tracking results for a single hand."""
    label: str            # "Left" or "Right" (anatomical)
    openness: float       # 0 = fist, 1 = fully open
    landmarks: list       # list of 21 (x, y, z) normalised coords


@dataclasses.dataclass
class FrameData:
    """Everything produced by one capture-and-detect cycle."""
    frame: np.ndarray     # BGR image with skeleton drawn on it
    hands: list           # list[HandData] — 0, 1 or 2 entries
    timestamp: float      # time.monotonic() of capture


# ── Standard MediaPipe hand connections ──────────────────────────────
_HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger (connected to palm via 5→9)
    (9, 10), (10, 11), (11, 12),
    # Ring finger (connected to palm via 9→13)
    (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17),
]

# Skeleton colours — warm blue in BGR
_BONE_COLOR    = (220, 110, 70)    # BGR → periwinkle in RGB
_JOINT_COLOR   = (240, 150, 100)   # slightly brighter
_JOINT_RADIUS  = 4
_BONE_THICKNESS = 2


# ── Palm openness ───────────────────────────────────────────────────

def _palm_openness(landmarks) -> float:
    """Compute palm openness from 21 landmarks (normalised x, y, z).

    Method: mean of (fingertip→MCP distances) / (wrist→middle_MCP distance).
    Remap from [0.3, 0.9] → [0.0, 1.0] with clamping.
    """
    tips = [4, 8, 12, 16, 20]       # thumb-tip, index-tip, …
    mcps = [2, 5,  9, 13, 17]       # thumb-IP, index-MCP, …

    def dist(a, b):
        return ((a.x - b.x)**2 + (a.y - b.y)**2 + (a.z - b.z)**2) ** 0.5

    wrist   = landmarks[0]
    mid_mcp = landmarks[9]
    hand_size = dist(wrist, mid_mcp) + 1e-6

    tip_dists = [dist(landmarks[t], landmarks[m]) for t, m in zip(tips, mcps)]
    raw = (sum(tip_dists) / len(tip_dists)) / hand_size

    # Remap [0.3, 0.9] → [0, 1]
    openness = (raw - 0.3) / 0.6
    return max(0.0, min(1.0, openness))


# ── Skeleton drawing ────────────────────────────────────────────────

def _draw_skeleton(frame: np.ndarray, landmarks, frame_w: int, frame_h: int):
    """Draw hand skeleton on the frame using OpenCV."""
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]

    # Bones
    for a, b in _HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], _BONE_COLOR, _BONE_THICKNESS,
                 cv2.LINE_AA)

    # Joints
    for pt in pts:
        cv2.circle(frame, pt, _JOINT_RADIUS, _JOINT_COLOR, -1, cv2.LINE_AA)


# ── Worker thread ───────────────────────────────────────────────────

class CVWorker:
    """Background thread that captures webcam + runs MediaPipe.

    Thread-safe via a lock — ``latest()`` returns the most-recent
    ``FrameData`` without blocking the render loop.
    """

    def __init__(self, model_path: str = "hand_landmarker.task"):
        self._model_path = str(pathlib.Path(model_path).resolve())
        self._lock = threading.Lock()
        self._latest: Optional[FrameData] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ── public API ──────────────────────────────────────────────────

    def start(self):
        """Launch the capture thread (idempotent)."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the thread to exit and wait for it to finish."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def latest(self) -> Optional[FrameData]:
        """Return the most-recent frame data (non-blocking)."""
        with self._lock:
            return self._latest

    # ── background loop ─────────────────────────────────────────────

    def _loop(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        detector = HandLandmarker.create_from_options(options)

        ts_ms = 0
        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.01)
                continue

            ts_ms += 33  # ~30 fps monotonic timestamp for VIDEO mode
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            result = detector.detect_for_video(mp_image, ts_ms)

            hands = []
            h, w = frame.shape[:2]

            if result.hand_landmarks:
                for lms, handedness in zip(result.hand_landmarks,
                                           result.handedness):
                    label = handedness[0].category_name   # "Left" or "Right"
                    openness = _palm_openness(lms)
                    # Draw skeleton BEFORE flip (both skeleton and image
                    # get mirrored together, keeping alignment correct).
                    _draw_skeleton(frame, lms, w, h)
                    hands.append(HandData(
                        label=label,
                        openness=openness,
                        landmarks=[(l.x, l.y, l.z) for l in lms],
                    ))

            # Mirror the frame horizontally for selfie-view
            frame = cv2.flip(frame, 1)

            fd = FrameData(frame=frame, hands=hands,
                           timestamp=time.monotonic())
            with self._lock:
                self._latest = fd

            # ~30 fps pacing
            time.sleep(0.005)

        cap.release()
        detector.close()
