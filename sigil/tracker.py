"""Hand Tracking Engine (§5.1).

Wraps MediaPipe Hand Landmarker for dual-hand 21-landmark real-time detection.
Falls back to legacy ``mp.solutions.hands`` if the task API is unavailable.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

from sigil.config import TrackingConfig
from sigil.utils import ensure_model, landmark_to_array

logger = logging.getLogger(__name__)

# ── Try task-based API first, fall back to legacy ────────────────────────────
_USE_TASK_API: bool = False
try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        HandLandmarker,
        HandLandmarkerOptions,
        RunningMode,
    )

    _USE_TASK_API = True
    logger.debug("Using MediaPipe Hand Landmarker task API")
except ImportError:
    import mediapipe as mp

    logger.warning("Task API unavailable – falling back to mp.solutions.hands")


# ── Result container ─────────────────────────────────────────────────────────
@dataclass
class HandResult:
    """Per-hand tracking result for a single frame."""

    handedness: str  # "Left" | "Right"
    landmarks: np.ndarray  # (21, 3) normalised
    world_landmarks: np.ndarray | None = None  # (21, 3) metres
    score: float = 0.0


@dataclass
class FrameResult:
    """All hands detected in one frame."""

    timestamp_ms: int = 0
    hands: list[HandResult] = field(default_factory=list)
    frame: np.ndarray | None = None  # original BGR frame (for overlay / recording)

    @property
    def left(self) -> HandResult | None:
        return next((h for h in self.hands if h.handedness == "Left"), None)

    @property
    def right(self) -> HandResult | None:
        return next((h for h in self.hands if h.handedness == "Right"), None)


# ── Tracker ──────────────────────────────────────────────────────────────────
class Tracker:
    """Real-time dual-hand landmark tracker (§5.1)."""

    def __init__(self, cfg: TrackingConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._detector: Any = None
        self._frame_count: int = 0
        self._fps_actual: float = 0.0
        self._last_time: float = 0.0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def open(self) -> None:
        """Open camera and initialise MediaPipe detector."""
        self._cap = cv2.VideoCapture(self._cfg.camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._cfg.frame_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._cfg.frame_height)
        self._cap.set(cv2.CAP_PROP_FPS, self._cfg.target_fps)

        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera {self._cfg.camera_id}. "
                "Check v4l2/pipewire and permissions."
            )

        if _USE_TASK_API:
            model_path = ensure_model(self._cfg.model_asset_path)
            options = HandLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO,
                num_hands=self._cfg.num_hands,
                min_hand_detection_confidence=self._cfg.min_detection_confidence,
                min_hand_presence_confidence=self._cfg.min_detection_confidence,
                min_tracking_confidence=self._cfg.min_tracking_confidence,
            )
            self._detector = HandLandmarker.create_from_options(options)
        else:
            self._detector = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self._cfg.num_hands,
                min_detection_confidence=self._cfg.min_detection_confidence,
                min_tracking_confidence=self._cfg.min_tracking_confidence,
            )

        self._last_time = time.monotonic()
        logger.info(
            "Tracker opened – camera %d @ %dx%d",
            self._cfg.camera_id,
            self._cfg.frame_width,
            self._cfg.frame_height,
        )

    def close(self) -> None:
        """Release camera and detector resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if _USE_TASK_API and self._detector is not None:
            self._detector.close()
        self._detector = None
        logger.info("Tracker closed")

    # ── per-frame ────────────────────────────────────────────────────────────
    def read(self) -> FrameResult | None:
        """Read one frame and run hand detection.

        Returns ``None`` on camera read failure.
        """
        if self._cap is None or self._detector is None:
            raise RuntimeError("Tracker not opened – call .open() first")

        ok, frame = self._cap.read()
        if not ok:
            logger.warning("Camera read failed")
            return None

        # ── Digital Zoom (Crop & Resize) ─────────────────────────────────────
        if self._cfg.zoom > 1.0:
            h, w = frame.shape[:2]
            new_w = int(w / self._cfg.zoom)
            new_h = int(h / self._cfg.zoom)
            x = (w - new_w) // 2
            y = (h - new_h) // 2
            frame = frame[y : y + new_h, x : x + new_w]
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_LINEAR)

        now = time.monotonic()
        dt = now - self._last_time
        self._last_time = now
        self._fps_actual = 1.0 / dt if dt > 0 else 0.0
        self._frame_count += 1

        timestamp_ms = int(now * 1000)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = FrameResult(timestamp_ms=timestamp_ms, frame=frame)

        if _USE_TASK_API:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            detection = self._detector.detect_for_video(mp_image, timestamp_ms)
            for i, hand_lm in enumerate(detection.hand_landmarks):
                handedness_label = detection.handedness[i][0].category_name
                world_lm = (
                    detection.hand_world_landmarks[i]
                    if detection.hand_world_landmarks
                    else None
                )
                result.hands.append(
                    HandResult(
                        handedness=handedness_label,
                        landmarks=landmark_to_array(hand_lm),
                        world_landmarks=(
                            landmark_to_array(world_lm) if world_lm else None
                        ),
                        score=detection.handedness[i][0].score,
                    )
                )
        else:
            # Legacy API
            detection = self._detector.process(rgb)
            if detection.multi_hand_landmarks:
                for i, hand_lm in enumerate(detection.multi_hand_landmarks):
                    label = "Right"
                    if detection.multi_handedness:
                        label = detection.multi_handedness[i].classification[
                            0
                        ].label
                    result.hands.append(
                        HandResult(
                            handedness=label,
                            landmarks=landmark_to_array(hand_lm.landmark),
                            score=(
                                detection.multi_handedness[i]
                                .classification[0]
                                .score
                                if detection.multi_handedness
                                else 0.0
                            ),
                        )
                    )

        return result

    # ── introspection ────────────────────────────────────────────────────────
    @property
    def fps(self) -> float:
        return self._fps_actual

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()
