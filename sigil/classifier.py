"""Gesture Classification Engine (§5.2).

Three classifier types:
  - Instant:     Single-frame pose via MediaPipe Gesture Recognizer or custom .tflite.
  - Gradual:     Continuous derived features (pinch distance, velocity, curl).
  - Sequential:  Multi-frame time-series via DTW / state-machine on landmark diffs.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

from sigil.config import ExecutionConfig, GestureMapping
from sigil.tracker import FrameResult, HandResult
from sigil.utils import (
    ensure_model,
    euclidean,
    finger_curl_angles,
    hand_velocity,
    monotonic_ms,
    palm_center,
    thumb_index_distance,
)

logger = logging.getLogger(__name__)

# ── Try MediaPipe Gesture Recognizer task API ────────────────────────────────
_HAS_GESTURE_RECOGNIZER: bool = False
try:
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions
    from mediapipe.tasks.python.vision import (
        GestureRecognizer,
        GestureRecognizerOptions,
        RunningMode,
    )

    _HAS_GESTURE_RECOGNIZER = True
except ImportError:
    pass


# ── Result container ─────────────────────────────────────────────────────────
@dataclass
class ClassificationResult:
    """Output of a classifier for one frame / event."""

    gesture_name: str  # matched config name or "" if none
    gesture_type: str  # "instant" | "gradual" | "sequential"
    confidence: float = 0.0
    value: float = 0.0  # for gradual: continuous 0–1 value
    position: tuple[float, float] | None = None  # (x, y) coordinates if applicable
    deltas: tuple[float, float] | None = None  # (dx, dy) for movement gestures
    hand: str = ""  # "Left" | "Right" | "Both"
    raw_label: str = ""  # underlying model label
    timestamp_ms: int = 0
    current_mode: str = "touchpad"  # active mode when this result was produced


# ── Abstract base ────────────────────────────────────────────────────────────
class BaseClassifier(ABC):
    """Common interface for all classifier types."""

    @abstractmethod
    def classify(self, frame_result: FrameResult) -> list[ClassificationResult]: ...

    @abstractmethod
    def close(self) -> None: ...


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Instant Classifier (§5.2 – single-frame pose)
# ═════════════════════════════════════════════════════════════════════════════
class InstantClassifier(BaseClassifier):
    """Uses MediaPipe Gesture Recognizer to classify single-frame hand poses.

    Maps recognised labels to config entries of ``type: instant``.
    """

    def __init__(
        self,
        mappings: list[GestureMapping],
        custom_model: str | None = None,
    ) -> None:
        self._mappings = [m for m in mappings if m.type == "instant" and m.enabled]
        self._recognizer: Any = None

        if _HAS_GESTURE_RECOGNIZER:
            model_file = custom_model or "gesture_recognizer.task"
            model_path = ensure_model(model_file)
            options = GestureRecognizerOptions(
                base_options=BaseOptions(model_asset_path=str(model_path)),
                running_mode=RunningMode.VIDEO,
                num_hands=2,
                min_hand_detection_confidence=0.7,
                min_tracking_confidence=0.6,
            )
            self._recognizer = GestureRecognizer.create_from_options(options)
            logger.info("InstantClassifier ready (task API, model=%s)", model_file)
        else:
            logger.warning("Gesture Recognizer task unavailable – instant classification disabled")

    def classify(self, frame_result: FrameResult) -> list[ClassificationResult]:
        results: list[ClassificationResult] = []
        if self._recognizer is None or frame_result.frame is None:
            return results

        import cv2

        rgb = cv2.cvtColor(frame_result.frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        recognition = self._recognizer.recognize_for_video(mp_image, frame_result.timestamp_ms)

        for i, gestures in enumerate(recognition.gestures):
            if not gestures:
                continue
            top = gestures[0]
            label = top.category_name
            score = top.score
            handedness = (
                recognition.handedness[i][0].category_name if recognition.handedness else "Right"
            )

            logger.debug(
                "MediaPipe gesture detected: label=%s, score=%.2f, hand=%s",
                label,
                score,
                handedness,
            )

            # Match against configured mappings
            for mapping in self._mappings:
                pose = mapping.condition.get("pose", "")
                if pose and pose == label and self._hand_matches(mapping.hand, handedness):
                    hand = frame_result.left if handedness == "Left" else frame_result.right
                    pos = None
                    if hand is not None:
                        tip = hand.landmarks[8]
                        pos = (float(tip[0]), float(tip[1]))

                    logger.info(
                        "Instant gesture matched: %s (pose=%s, hand=%s, conf=%.2f)",
                        mapping.name,
                        label,
                        handedness,
                        score,
                    )

                    results.append(
                        ClassificationResult(
                            gesture_name=mapping.name,
                            gesture_type="instant",
                            confidence=score,
                            position=pos,
                            hand=handedness,
                            raw_label=label,
                            timestamp_ms=frame_result.timestamp_ms,
                        )
                    )
                    break  # first match priority

        return results

    @staticmethod
    def _hand_matches(config_hand: str, detected: str) -> bool:
        if config_hand == "both":
            return True
        return config_hand.lower() == detected.lower()

    def close(self) -> None:
        if self._recognizer is not None:
            self._recognizer.close()
            self._recognizer = None


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Gradual Classifier (§5.2 – continuous derived features)
# ═════════════════════════════════════════════════════════════════════════════
_FEATURE_EXTRACTORS: dict[str, Any] = {}


def _register_feature(name: str):  # type: ignore[no-untyped-def]
    def decorator(fn):  # type: ignore[no-untyped-def]
        _FEATURE_EXTRACTORS[name] = fn
        return fn

    return decorator


@_register_feature("thumb_index_distance")
def _feat_thumb_index(hand: HandResult, **_: Any) -> float:
    return thumb_index_distance(hand.landmarks)


@_register_feature("finger_curl_mean")
def _feat_curl(hand: HandResult, **_: Any) -> float:
    return float(finger_curl_angles(hand.landmarks).mean())


@_register_feature("palm_velocity")
def _feat_velocity(
    hand: HandResult,
    prev: np.ndarray | None = None,
    dt: float = 0.033,
    **_: Any,
) -> float:
    return hand_velocity(prev, hand.landmarks, dt)


@_register_feature("two_hand_distance")
def _feat_two_hand_dist(
    _hand: HandResult,
    left: HandResult | None = None,
    right: HandResult | None = None,
    **_kw: Any,
) -> float:
    if left is None or right is None:
        return 0.0
    return euclidean(palm_center(left.landmarks), palm_center(right.landmarks))


@_register_feature("pointer_position")
def _feat_pointer_pos(hand: HandResult, **_: Any) -> tuple[float, float]:
    """Returns (x, y) of index finger tip (landmark 8)."""
    tip = hand.landmarks[8]
    return (float(tip[0]), float(tip[1]))


@_register_feature("finger_count")
def _feat_finger_count(hand: HandResult, **_: Any) -> float:
    """Returns number of extended fingers."""
    from sigil.utils import count_extended_fingers

    return float(count_extended_fingers(hand.landmarks))


@_register_feature("finger_deltas")
def _feat_finger_deltas(
    hand: HandResult, prev: np.ndarray | None = None, **_: Any
) -> tuple[float, float]:
    """Returns (dx, dy) of palm center movement."""
    if prev is None:
        return (0.0, 0.0)

    curr_palm = palm_center(hand.landmarks)
    prev_palm = palm_center(prev)
    dx = curr_palm[0] - prev_palm[0]
    dy = curr_palm[1] - prev_palm[1]
    return (float(dx), float(dy))


@_register_feature("touchpad_data")
def _feat_touchpad(hand: HandResult, **_: Any) -> tuple[float, float, float]:
    """Returns (x, y, finger_count)."""
    from sigil.utils import count_extended_fingers

    tip = hand.landmarks[8]  # index tip for position
    count = count_extended_fingers(hand.landmarks)
    return (float(tip[0]), float(tip[1]), float(count))


class GradualClassifier(BaseClassifier):
    """Derives continuous features from landmarks and triggers on threshold changes."""

    def __init__(self, mappings: list[GestureMapping]) -> None:
        self._mappings = [m for m in mappings if m.type == "gradual" and m.enabled]
        self._prev_values: dict[str, float | tuple[float, float, float] | tuple[float, float]] = {}
        self._prev_landmarks: dict[str, np.ndarray] = {}  # per hand
        self._prev_time: float = 0.0

    def classify(self, frame_result: FrameResult) -> list[ClassificationResult]:
        results: list[ClassificationResult] = []
        now = frame_result.timestamp_ms / 1000.0
        dt = now - self._prev_time if self._prev_time > 0 else 0.033
        self._prev_time = now

        left = frame_result.left
        right = frame_result.right

        for mapping in self._mappings:
            feat_name = mapping.condition.get("feature", "")
            extractor = _FEATURE_EXTRACTORS.get(feat_name)
            if extractor is None:
                continue

            # Determine which hand(s) to pass
            hand = self._pick_hand(mapping.hand, left, right)
            if hand is None:
                continue

            prev_lm = self._prev_landmarks.get(mapping.name)
            value = extractor(hand, prev=prev_lm, dt=dt, left=left, right=right)

            prev_val = self._prev_values.get(mapping.name, value)

            # Special case for coordinate features (tuples)
            if isinstance(value, tuple):
                # For positions, we don't trigger on delta by default,
                # but we emit the result if the hand is detected.
                triggered = True
                pos = (value[0], value[1]) if len(value) >= 2 else None
                deltas = value if "deltas" in feat_name else None
                scalar_val = value[2] if len(value) == 3 else 0.0
            else:
                delta = value - prev_val  # type: ignore[operator]
                min_delta = mapping.condition.get("min_delta", 0.05)
                direction = mapping.condition.get("direction", "both")
                triggered = (
                    (direction == "increase" and delta >= min_delta)
                    or (direction == "decrease" and delta <= -min_delta)
                    or (direction == "both" and abs(delta) >= min_delta)
                )
                pos = None
                deltas = None
                scalar_val = float(value)

            self._prev_values[mapping.name] = value
            self._prev_landmarks[mapping.name] = hand.landmarks.copy()

            if triggered:
                results.append(
                    ClassificationResult(
                        gesture_name=mapping.name,
                        gesture_type="gradual",
                        confidence=1.0,
                        value=scalar_val,
                        position=pos,
                        deltas=deltas,
                        hand=mapping.hand,
                        raw_label=feat_name,
                        timestamp_ms=frame_result.timestamp_ms,
                    )
                )

        return results

    @staticmethod
    def _pick_hand(
        config_hand: str, left: HandResult | None, right: HandResult | None
    ) -> HandResult | None:
        if config_hand == "both":
            return left or right  # extractor receives both via kwargs
        if config_hand == "left":
            return left
        return right

    def close(self) -> None:
        self._prev_values.clear()
        self._prev_landmarks.clear()


# ═════════════════════════════════════════════════════════════════════════════
# 3.  Sequential Classifier (§5.2 – temporal multi-frame series)
# ═════════════════════════════════════════════════════════════════════════════


@dataclass
class _SeqState:
    """Internal tracking state for one sequential mapping."""

    step_index: int = 0
    frame_counter: int = 0


class SequentialClassifier(BaseClassifier):
    """State-machine based sequential gesture detector.

    Monitors a buffer of recent instant labels and checks if the required
    sequence occurs within ``window_frames``.
    """

    def __init__(
        self,
        mappings: list[GestureMapping],
        buffer_size: int = 30,
    ) -> None:
        self._mappings = [m for m in mappings if m.type == "sequential" and m.enabled]
        self._buffer: deque[str] = deque(maxlen=buffer_size)
        self._states: dict[str, _SeqState] = {m.name: _SeqState() for m in self._mappings}
        # Also track the DTW / raw landmark diff buffer for advanced matching
        self._landmark_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)

    def feed_instant_label(self, label: str) -> None:
        """Feed an instant-classifier label into the sequence buffer.

        Called by the daemon after instant classification.
        """
        if label:
            self._buffer.append(label)

    def feed_landmarks(self, landmarks: np.ndarray) -> None:
        """Feed raw landmarks for DTW-based matching (future use)."""
        self._landmark_buffer.append(landmarks.copy())

    def classify(self, frame_result: FrameResult) -> list[ClassificationResult]:
        results: list[ClassificationResult] = []

        for mapping in self._mappings:
            seq = mapping.condition.get("sequence", [])
            window = mapping.condition.get("window_frames", 20)
            if not seq:
                continue

            state = self._states[mapping.name]
            state.frame_counter += 1

            # Check the recent buffer tail for the next expected label
            if state.step_index < len(seq):
                expected = seq[state.step_index]
                # Search recent buffer entries
                recent = list(self._buffer)[-window:]
                if expected in recent:
                    state.step_index += 1
                    state.frame_counter = 0

            # Check for timeout
            if state.frame_counter > window and state.step_index > 0:
                state.step_index = 0
                state.frame_counter = 0

            # Sequence complete?
            if state.step_index >= len(seq):
                results.append(
                    ClassificationResult(
                        gesture_name=mapping.name,
                        gesture_type="sequential",
                        confidence=1.0,
                        hand=mapping.hand,
                        raw_label="→".join(seq),
                        timestamp_ms=frame_result.timestamp_ms,
                    )
                )
                # Reset
                state.step_index = 0
                state.frame_counter = 0
                self._buffer.clear()

        return results

    def close(self) -> None:
        self._buffer.clear()
        self._landmark_buffer.clear()
        self._states.clear()


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Unified Classifier Facade
# ═════════════════════════════════════════════════════════════════════════════
class GestureClassifier:
    """Unified facade that runs all three sub-classifiers (§5.2 hybrid).

    Instant gesture execution model:
      - A gesture must be detected for ``confirm_frames`` consecutive frames
        (each above ``confidence_threshold``) before it fires.
      - Once an instant action fires, ALL classification is suppressed for
        ``blanking_ms`` to prevent hand-transition noise from triggering
        unintended actions.

    Mode system:
      - Two operational modes: "touchpad" (cursor/scroll) and "keybind" (gestures).
      - Gestures declare which modes they are active in via ``active_modes``.
      - Gestures with ``mode_toggle: true`` switch between modes and are
        immune to blanking so the user can always switch back.
    """

    def __init__(
        self,
        mappings: list[GestureMapping],
        execution: ExecutionConfig | None = None,
        custom_instant_model: str | None = None,
    ) -> None:
        self.instant = InstantClassifier(mappings, custom_instant_model)
        self.gradual = GradualClassifier(mappings)
        self.sequential = SequentialClassifier(mappings)
        self._last_fire: dict[str, int] = {}  # per-gesture cooldown tracking

        # Execution policy
        exec_cfg = execution or ExecutionConfig()
        self._confirm_frames = max(1, exec_cfg.confirm_frames)
        self._blanking_ms = exec_cfg.blanking_ms
        self._confidence_threshold = exec_cfg.confidence_threshold

        # Confirmation state for instant gestures:
        # tracks consecutive frames a specific gesture has been seen
        self._instant_streak: dict[str, int] = {}

        # Global blanking timestamp – 0 means inactive
        self._blanking_until: int = 0

        # Mode state
        self._current_mode: str = "touchpad"  # "touchpad" | "keybind"
        self._mappings = mappings  # keep reference for mode lookups

    @property
    def current_mode(self) -> str:
        return self._current_mode

    def _is_mode_active(self, gesture_name: str) -> bool:
        """Check if a gesture is active in the current mode."""
        for m in self._mappings:
            if m.name == gesture_name:
                modes = m.active_modes
                return "both" in modes or self._current_mode in modes
        return False

    def _is_mode_toggle(self, gesture_name: str) -> bool:
        """Check if a gesture is a mode toggle."""
        for m in self._mappings:
            if m.name == gesture_name:
                return m.mode_toggle
        return False

    def classify(self, frame_result: FrameResult) -> list[ClassificationResult]:
        """Run all classifiers and return de-duplicated, cooldown-filtered results."""
        now = monotonic_ms()

        # ── Global blanking: suppress EVERYTHING except continuous actions ─────
        blanking_active = now < self._blanking_until

        all_results: list[ClassificationResult] = []

        # 1. Instant — apply confirmation logic
        instant_results = self.instant.classify(frame_result)
        for r in instant_results:
            self.sequential.feed_instant_label(r.raw_label)
            logger.debug(
                "Instant gesture detected: %s (label=%s, conf=%.2f, hand=%s)",
                r.gesture_name,
                r.raw_label,
                r.confidence,
                r.hand,
            )
        confirmed_instants = self._apply_instant_confirmation(instant_results, now)
        all_results.extend(confirmed_instants)

        # 2. Gradual (only if no instant action fired this frame)
        if not confirmed_instants:
            all_results.extend(self.gradual.classify(frame_result))

        # 3. Sequential
        all_results.extend(self.sequential.classify(frame_result))

        # Feed landmarks for sequential DTW
        for hand in frame_result.hands:
            self.sequential.feed_landmarks(hand.landmarks)

        # Filter results based on blanking, cooldowns, and mode
        filtered: list[ClassificationResult] = []
        for r in all_results:
            is_toggle = self._is_mode_toggle(r.gesture_name)
            is_continuous = self._is_continuous(r.gesture_name)

            # ── Mode filter: skip gestures not active in current mode ────────
            # This MUST apply to ALL gestures (including continuous) so touchpad
            # cursor gestures don't fire in keybind mode
            if not is_toggle and not self._is_mode_active(r.gesture_name):
                logger.debug(
                    "Gesture '%s' filtered out (not active in %s mode)",
                    r.gesture_name,
                    self._current_mode,
                )
                continue

            # ── Blanking filter ─────────────────────────────────────────────
            if blanking_active and not is_continuous and not is_toggle:
                continue

            if is_continuous:
                # Continuous gestures bypass cooldown and don't trigger blanking
                # but mode filter was already applied above
                filtered.append(r)
            else:
                last = self._last_fire.get(r.gesture_name, 0)
                cooldown = self._lookup_cooldown(r.gesture_name)
                if now - last >= cooldown:
                    self._last_fire[r.gesture_name] = now
                    filtered.append(r)

        # If any NON-CONTINUOUS instant action fired → activate global blanking
        for r in filtered:
            if r.gesture_type == "instant" and not self._is_continuous(r.gesture_name):
                self._blanking_until = now + self._blanking_ms
                self._instant_streak.clear()
                logger.info(
                    "Instant action '%s' confirmed – blanking for %d ms",
                    r.gesture_name,
                    self._blanking_ms,
                )
                break  # one trigger is enough to blank

        # Apply current_mode to all results so daemon/overlay can read it
        for r in filtered:
            r.current_mode = self._current_mode

        # Process mode toggles (after filtering, before returning)
        for r in filtered:
            if self._is_mode_toggle(r.gesture_name):
                logger.info(
                    "Mode toggle gesture '%s' triggered (current_mode before=%s)",
                    r.gesture_name,
                    self._current_mode,
                )
                self._toggle_mode()
                r.current_mode = self._current_mode
                break  # one toggle per frame

        return filtered

    def _toggle_mode(self) -> None:
        """Switch between touchpad and keybind modes."""
        old_mode = self._current_mode
        self._current_mode = "keybind" if self._current_mode == "touchpad" else "touchpad"
        logger.info(
            "Mode toggled: %s → %s (current_mode=%s)",
            old_mode,
            self._current_mode,
            self._current_mode,
        )

    def _is_continuous(self, gesture_name: str) -> bool:
        """Check if a gesture is marked as continuous in config."""
        for m in (
            list(self.instant._mappings)
            + list(self.gradual._mappings)
            + list(self.sequential._mappings)
        ):
            if m.name == gesture_name:
                return m.continuous
        return False

    # ── Instant confirmation helpers ─────────────────────────────────────────
    def _apply_instant_confirmation(
        self,
        raw_results: list[ClassificationResult],
        now: int,
    ) -> list[ClassificationResult]:
        """Require N consecutive frames of the same gesture before emitting it."""
        if not raw_results:
            # No gesture detected this frame → reset all streaks
            self._instant_streak.clear()
            return []

        confirmed: list[ClassificationResult] = []
        current_gestures = set()

        for r in raw_results:
            # Skip low-confidence detections
            if r.confidence < self._confidence_threshold:
                logger.debug(
                    "Gesture '%s' below confidence threshold (%.2f < %.2f)",
                    r.gesture_name,
                    r.confidence,
                    self._confidence_threshold,
                )
                continue

            name = r.gesture_name
            current_gestures.add(name)

            # Increment streak for this gesture independently
            self._instant_streak[name] = self._instant_streak.get(name, 0) + 1
            logger.debug(
                "Gesture '%s' streak: %d/%d", name, self._instant_streak[name], self._confirm_frames
            )

            if self._instant_streak[name] >= self._confirm_frames:
                logger.info(
                    "Gesture '%s' CONFIRMED (streak=%d >= %d)",
                    name,
                    self._instant_streak[name],
                    self._confirm_frames,
                )
                confirmed.append(r)

        # Drop streaks for gestures not seen in this frame
        for name in list(self._instant_streak.keys()):
            if name not in current_gestures:
                logger.debug("Gesture '%s' streak reset (not detected this frame)", name)
                del self._instant_streak[name]

        return confirmed

    def _lookup_cooldown(self, gesture_name: str) -> int:
        """Find the per-gesture cooldown_ms from mappings."""
        for m in (
            list(self.instant._mappings)
            + list(self.gradual._mappings)
            + list(self.sequential._mappings)
        ):
            if m.name == gesture_name:
                return m.cooldown_ms
        return 300

    def close(self) -> None:
        self.instant.close()
        self.gradual.close()
        self.sequential.close()
