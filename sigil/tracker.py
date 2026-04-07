"""Hand Tracking Engine (§5.1) — Single-Pass Architecture.

Uses MediaPipe GestureRecognizer (Task API) for unified hand detection:
one inference call produces BOTH 21-landmark coordinates AND built-in
gesture classifications (7 classes).  The custom geometric classifier
then runs on the same landmarks with zero additional detection cost.

Falls back to legacy ``mp.solutions.hands`` if the task API is unavailable,
but loses built-in gesture scores in that mode.
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


# ── Result containers ────────────────────────────────────────────────────────
@dataclass
class BuiltinGesture:
    """A single built-in gesture prediction from MediaPipe GestureRecognizer."""

    label: str  # e.g. "Closed_Fist", "Open_Palm", "None"
    score: float  # 0.0–1.0 confidence


@dataclass
class HandResult:
    """Per-hand tracking result for a single frame."""

    handedness: str  # "Left" | "Right"
    landmarks: np.ndarray  # (21, 3) normalised
    world_landmarks: np.ndarray | None = None  # (21, 3) metres
    score: float = 0.0
    builtin_gesture: BuiltinGesture | None = None  # from GestureRecognizer


@dataclass
class FrameResult:
    """All hands detected in one frame."""

    frame_id: int = 0
    timestamp_ms: int = 0
    hands: list[HandResult] = field(default_factory=list)
    frame: np.ndarray | None = None  # original BGR frame (for overlay / recording)
    rgb_frame: np.ndarray | None = None  # RGB frame used for model inference

    @property
    def left(self) -> HandResult | None:
        return next((h for h in self.hands if h.handedness == "Left"), None)

    @property
    def right(self) -> HandResult | None:
        return next((h for h in self.hands if h.handedness == "Right"), None)


# ── Tracker ──────────────────────────────────────────────────────────────────
class Tracker:
    """Real-time dual-hand landmark + gesture tracker (§5.1).

    Single-pass architecture: one GestureRecognizer call yields both
    landmarks and built-in gesture scores.
    """

    def __init__(self, cfg: TrackingConfig) -> None:
        self._cfg = cfg
        self._cap: cv2.VideoCapture | None = None
        self._detector: Any = None
        self._is_task_api = False
        self._is_gesture_api = False  # True when using GestureRecognizer (preferred)
        self._frame_count: int = 0
        self._fps_actual: float = 0.0
        self._last_time: float = 0.0
        self._fps_alpha: float = 0.2
        self._inference_scale: float = max(0.25, min(1.0, cfg.inference_scale))

        # ── Noise Filtering State ───────────────────────────────────────────
        # Exponential Moving Average (EMA) for each hand to reduce jitter
        self._smoothed_lms: dict[str, np.ndarray] = {}  # "Left", "Right"
        self._alpha = 0.45  # Smoothing factor (0.0-1.0). Lower = more stable but laggier.

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

        # ── Try GestureRecognizer first (single-pass: landmarks + gestures) ──
        try:
            from mediapipe.tasks.python import BaseOptions
            from mediapipe.tasks.python.vision import (
                GestureRecognizer,
                GestureRecognizerOptions,
                RunningMode,
            )

            model_path = ensure_model("gesture_recognizer.task")
            options = GestureRecognizerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(model_path),
                    delegate=BaseOptions.Delegate.CPU,
                ),
                running_mode=RunningMode.VIDEO,
                num_hands=self._cfg.num_hands,
                min_hand_detection_confidence=self._cfg.min_detection_confidence,
                min_hand_presence_confidence=self._cfg.min_detection_confidence,
                min_tracking_confidence=self._cfg.min_tracking_confidence,
            )
            self._detector = GestureRecognizer.create_from_options(options)
            self._is_task_api = True
            self._is_gesture_api = True
            logger.debug("Using MediaPipe GestureRecognizer (single-pass: landmarks + gestures)")

        except (ImportError, Exception) as e:
            logger.warning(
                "GestureRecognizer unavailable (%s). Trying HandLandmarker…", e
            )

            # ── Fall back to HandLandmarker (landmarks only, no built-in gestures) ──
            try:
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import (
                    HandLandmarker,
                    HandLandmarkerOptions,
                    RunningMode,
                )

                model_path = ensure_model(self._cfg.model_asset_path)
                options = HandLandmarkerOptions(
                    base_options=BaseOptions(
                        model_asset_path=str(model_path),
                        delegate=BaseOptions.Delegate.CPU,
                    ),
                    running_mode=RunningMode.VIDEO,
                    num_hands=self._cfg.num_hands,
                    min_hand_detection_confidence=self._cfg.min_detection_confidence,
                    min_hand_presence_confidence=self._cfg.min_detection_confidence,
                    min_tracking_confidence=self._cfg.min_tracking_confidence,
                )
                self._detector = HandLandmarker.create_from_options(options)
                self._is_task_api = True
                self._is_gesture_api = False
                logger.debug("Using MediaPipe HandLandmarker (landmarks only, no built-in gestures)")

            except (ImportError, Exception) as e2:
                logger.warning(
                    "HandLandmarker also failed (%s). Falling back to legacy.", e2
                )
                import mediapipe as mp

                self._detector = mp.solutions.hands.Hands(
                    static_image_mode=False,
                    max_num_hands=self._cfg.num_hands,
                    min_detection_confidence=self._cfg.min_detection_confidence,
                    min_tracking_confidence=self._cfg.min_tracking_confidence,
                )
                self._is_task_api = False
                self._is_gesture_api = False

        self._last_time = time.monotonic()
        logger.info(
            "Tracker opened – camera %d @ %dx%d (gesture_api=%s)",
            self._cfg.camera_id,
            self._cfg.frame_width,
            self._cfg.frame_height,
            self._is_gesture_api,
        )

    def close(self) -> None:
        """Release camera and detector resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._is_task_api and self._detector is not None:
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
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        if self._fps_actual <= 0:
            self._fps_actual = inst_fps
        else:
            self._fps_actual = (
                (1.0 - self._fps_alpha) * self._fps_actual
                + self._fps_alpha * inst_fps
            )
        self._frame_count += 1

        timestamp_ms = int(now * 1000)
        inference_frame = frame
        if self._inference_scale < 0.999:
            inference_frame = cv2.resize(
                frame,
                None,
                fx=self._inference_scale,
                fy=self._inference_scale,
                interpolation=cv2.INTER_AREA,
            )

        rgb = cv2.cvtColor(inference_frame, cv2.COLOR_BGR2RGB)

        result = FrameResult(
            frame_id=self._frame_count,
            timestamp_ms=timestamp_ms,
            frame=frame,
            rgb_frame=rgb,
        )

        if self._is_gesture_api:
            # ── GestureRecognizer: single pass → landmarks + built-in gestures ──
            self._process_gesture_recognizer(rgb, timestamp_ms, result)
        elif self._is_task_api:
            # ── HandLandmarker: landmarks only ──
            self._process_hand_landmarker(rgb, timestamp_ms, result)
        else:
            # ── Legacy API ──
            self._process_legacy(rgb, result)

        # Clear smoothing state for hands not detected this frame
        detected_labels = {h.handedness for h in result.hands}
        for label in list(self._smoothed_lms.keys()):
            if label not in detected_labels:
                del self._smoothed_lms[label]

        return result

    def _process_gesture_recognizer(
        self, rgb: np.ndarray, timestamp_ms: int, result: FrameResult
    ) -> None:
        """Process frame with GestureRecognizer (single-pass architecture)."""
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self._detector.recognize_for_video(mp_image, timestamp_ms)

        for i, hand_lm in enumerate(detection.hand_landmarks):
            handedness_label = detection.handedness[i][0].category_name
            raw_lms = landmark_to_array(hand_lm)

            # Apply Smoothing Filter
            smoothed = self._apply_smoothing(handedness_label, raw_lms)

            # Extract world landmarks
            world_lm = (
                detection.hand_world_landmarks[i]
                if detection.hand_world_landmarks
                else None
            )

            # Extract built-in gesture from this same pass (zero extra cost)
            builtin = None
            if detection.gestures and i < len(detection.gestures):
                gesture_list = detection.gestures[i]
                if gesture_list:
                    top = gesture_list[0]
                    builtin = BuiltinGesture(
                        label=top.category_name,
                        score=top.score,
                    )

            result.hands.append(
                HandResult(
                    handedness=handedness_label,
                    landmarks=smoothed,
                    world_landmarks=(
                        landmark_to_array(world_lm) if world_lm else None
                    ),
                    score=detection.handedness[i][0].score,
                    builtin_gesture=builtin,
                )
            )

    def _process_hand_landmarker(
        self, rgb: np.ndarray, timestamp_ms: int, result: FrameResult
    ) -> None:
        """Process frame with HandLandmarker (landmarks only, no built-in gestures)."""
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        detection = self._detector.detect_for_video(mp_image, timestamp_ms)

        for i, hand_lm in enumerate(detection.hand_landmarks):
            handedness_label = detection.handedness[i][0].category_name
            raw_lms = landmark_to_array(hand_lm)
            smoothed = self._apply_smoothing(handedness_label, raw_lms)

            world_lm = (
                detection.hand_world_landmarks[i]
                if detection.hand_world_landmarks
                else None
            )
            result.hands.append(
                HandResult(
                    handedness=handedness_label,
                    landmarks=smoothed,
                    world_landmarks=(
                        landmark_to_array(world_lm) if world_lm else None
                    ),
                    score=detection.handedness[i][0].score,
                    builtin_gesture=None,  # No gesture info from HandLandmarker
                )
            )

    def _process_legacy(self, rgb: np.ndarray, result: FrameResult) -> None:
        """Process frame with legacy mp.solutions.hands."""
        detection = self._detector.process(rgb)
        if detection.multi_hand_landmarks:
            for i, hand_lm in enumerate(detection.multi_hand_landmarks):
                label = "Right"
                if detection.multi_handedness:
                    label = detection.multi_handedness[i].classification[0].label

                raw_lms = landmark_to_array(hand_lm.landmark)
                smoothed = self._apply_smoothing(label, raw_lms)

                result.hands.append(
                    HandResult(
                        handedness=label,
                        landmarks=smoothed,
                        score=(
                            detection.multi_handedness[i]
                            .classification[0]
                            .score
                            if detection.multi_handedness
                            else 0.0
                        ),
                        builtin_gesture=None,  # No gesture info from legacy API
                    )
                )

    def _apply_smoothing(self, label: str, current: np.ndarray) -> np.ndarray:
        """Apply Exponential Moving Average to landmarks."""
        if label not in self._smoothed_lms:
            self._smoothed_lms[label] = current
            return current

        # EMA: Smoothed = (1 - alpha) * Prev + alpha * Current
        prev = self._smoothed_lms[label]
        smoothed = (1.0 - self._alpha) * prev + self._alpha * current
        self._smoothed_lms[label] = smoothed
        return smoothed

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
