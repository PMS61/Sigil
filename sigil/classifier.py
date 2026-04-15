"""Gesture Classification Engine (§5.2) — Single-Pass Architecture.

Three classifier types:
  - Instant:     Reads built-in gesture scores from the tracker's FrameResult
                 (no separate GestureRecognizer — zero extra inference cost).
  - Gradual:     Continuous derived features (pinch distance, curl).
  - Sequential:  Multi-frame time-series via DTW / state-machine on landmark diffs.

Plus:
  - Geometric:   Custom Random Forest on 96-dim features from the same landmarks.

The GestureClassifier facade merges built-in and geometric results with
priority logic: custom wins if confident, falls back to built-in.
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
    count_extended_fingers,
    euclidean,
    finger_curl_angles,
    monotonic_ms,
    palm_center,
    thumb_index_distance,
)

logger = logging.getLogger(__name__)


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
    continuous: bool = False  # if True, this gesture triggers every frame
    source: str = ""  # "builtin" | "geometric" for priority resolution


# ── Abstract base ────────────────────────────────────────────────────────────
class BaseClassifier(ABC):
    """Common interface for all classifier types."""

    @abstractmethod
    def classify(self, frame_result: FrameResult) -> list[ClassificationResult]: ...

    @abstractmethod
    def close(self) -> None: ...


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Instant Classifier (§5.2 – reads built-in gestures from tracker)
# ═════════════════════════════════════════════════════════════════════════════
class InstantClassifier(BaseClassifier):
    """Reads built-in gesture scores directly from FrameResult.

    The tracker's GestureRecognizer already produced gesture classifications
    alongside landmarks — this classifier just maps those to config entries.
    Zero additional inference cost.
    """

    def __init__(
        self,
        mappings: list[GestureMapping],
        execution: ExecutionConfig | None = None,
    ) -> None:
        self._mappings = [m for m in mappings if m.type == "instant" and m.enabled]
        logger.info(
            "InstantClassifier ready (reads built-in gestures from tracker, %d mappings)",
            len(self._mappings),
        )

    def classify(self, frame_result: FrameResult) -> list[ClassificationResult]:
        if not self._mappings:
            return []

        best_per_hand: dict[str, ClassificationResult] = {}

        for hand in frame_result.hands:
            builtin = hand.builtin_gesture
            if builtin is None:
                continue

            label = builtin.label
            score = builtin.score
            handedness = hand.handedness

            logger.debug(
                "Built-in gesture: label=%s, score=%.2f, hand=%s",
                label,
                score,
                handedness,
            )

            # Match against configured mappings
            for mapping in self._mappings:
                pose = mapping.condition.get("pose", "")
                finger_count = mapping.condition.get("finger_count")

                pose_match = not pose or pose == label
                finger_match = True
                if finger_count is not None:
                    actual_count = count_extended_fingers(hand.landmarks, hand.handedness)
                    finger_match = actual_count == finger_count

                if pose_match and finger_match and self._hand_matches(mapping.hand, handedness):
                    pos = (float(hand.landmarks[8][0]), float(hand.landmarks[8][1]))

                    logger.debug(
                        "Instant gesture matched: %s (pose=%s, hand=%s, conf=%.2f, source=builtin)",
                        mapping.name,
                        label,
                        handedness,
                        score,
                    )

                    candidate = ClassificationResult(
                        gesture_name=mapping.name,
                        gesture_type="instant",
                        confidence=score,
                        position=pos,
                        hand=handedness,
                        raw_label=label,
                        timestamp_ms=frame_result.timestamp_ms,
                        continuous=mapping.continuous,
                        source="builtin",
                    )
                    prev = best_per_hand.get(handedness)
                    if prev is None or candidate.confidence > prev.confidence:
                        best_per_hand[handedness] = candidate
                    break

        return list(best_per_hand.values())

    @staticmethod
    def _hand_matches(config_hand: str, detected: str) -> bool:
        if config_hand == "both":
            return True
        return config_hand.lower() == detected.lower()

    def close(self) -> None:
        pass  # No resources to release — tracker owns the recognizer


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


@_register_feature("pinch_ratio")
def _feat_pinch_ratio(hand: HandResult, **_: Any) -> float:
    """Thumb-index distance normalized by wrist-index MCP distance."""
    base = euclidean(hand.landmarks[0], hand.landmarks[5]) + 1e-7
    return thumb_index_distance(hand.landmarks) / base


@_register_feature("finger_curl_mean")
def _feat_curl(hand: HandResult, **_: Any) -> float:
    return float(finger_curl_angles(hand.landmarks).mean())


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
    return float(count_extended_fingers(hand.landmarks, hand.handedness))


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


class GradualClassifier(BaseClassifier):
    """Derives continuous features from landmarks and triggers on threshold changes."""

    def __init__(
        self, mappings: list[GestureMapping], execution: ExecutionConfig | None = None
    ) -> None:
        self._mappings = [m for m in mappings if m.type == "gradual" and m.enabled]
        self._prev_values: dict[str, float | tuple[float, float, float] | tuple[float, float]] = {}
        self._prev_landmarks: dict[str, np.ndarray] = {}  # per hand
        self._prev_time: float = 0.0
        self._finger_count_history: dict[str, list[int]] = {}
        self._hysteresis_state: dict[str, bool] = {}

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

            # Check finger_count condition if specified (Per-hand check)
            finger_count = mapping.condition.get("finger_count")
            if finger_count is not None:
                actual_count = count_extended_fingers(hand.landmarks, hand.handedness)
                # Use history to require consistent finger count for stability
                count_history = self._finger_count_history.get(mapping.name, [])
                count_history.append(actual_count)
                if len(count_history) > 3:
                    count_history = count_history[-3:]
                self._finger_count_history[mapping.name] = count_history

                # Check if majority of recent frames match
                matches = sum(1 for c in count_history if c == finger_count)
                if matches < 2:  # require at least 2 of last 3 frames to match
                    continue

                # SOLIDITY FIX: If this is a continuous gesture stop immediately
                if mapping.continuous and actual_count != finger_count:
                    self._finger_count_history[mapping.name] = []
                    continue

            prev_lm = self._prev_landmarks.get(mapping.name)
            value = extractor(hand, prev=prev_lm, dt=dt, left=left, right=right)

            prev_val = self._prev_values.get(mapping.name, value)

            # Special case for coordinate features (tuples)
            if isinstance(value, tuple):
                if feat_name == "finger_deltas":
                    dx, dy = float(value[0]), float(value[1])
                    min_delta = float(mapping.condition.get("min_delta", 0.05))
                    component = str(mapping.condition.get("delta_component", "both")).lower()

                    if component == "dx":
                        metric = abs(dx)
                    elif component == "dy":
                        metric = abs(dy)
                    else:
                        metric = max(abs(dx), abs(dy))

                    triggered = metric >= min_delta
                    pos = None
                    deltas = (dx, dy)
                    scalar_val = metric
                else:
                    triggered = True
                    pos = (value[0], value[1]) if len(value) >= 2 else None
                    deltas = value if "deltas" in feat_name else None
                    scalar_val = value[2] if len(value) == 3 else 0.0
            else:
                if (
                    "enter_below" in mapping.condition
                    or "exit_above" in mapping.condition
                    or mapping.condition.get("edge") in ("press", "release", "both")
                ):
                    enter_below = mapping.condition.get("enter_below")
                    exit_above = mapping.condition.get("exit_above", enter_below)
                    edge = str(mapping.condition.get("edge", "both")).lower()

                    was_active = self._hysteresis_state.get(mapping.name, False)
                    is_active = was_active

                    if enter_below is not None and value <= float(enter_below):
                        is_active = True
                    elif exit_above is not None and value >= float(exit_above):
                        is_active = False

                    triggered = (
                        (not was_active and is_active and edge in ("press", "both"))
                        or (was_active and not is_active and edge in ("release", "both"))
                    )
                    self._hysteresis_state[mapping.name] = is_active
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
                        continuous=mapping.continuous,
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
# 3.  Geometric Classifier (§5.2 – custom landmark-based pose classification)
# ═════════════════════════════════════════════════════════════════════════════
class GeometricClassifier(BaseClassifier):
    """Uses Random Forest to classify 101-dimensional geometric features.

    Loads .pkl artifacts from MODELS_DIR trained via the landmark architecture.
    Runs on the SAME landmarks from the tracker — zero extra detection cost.
    """

    def __init__(
        self,
        mappings: list[GestureMapping],
        execution: ExecutionConfig | None = None,
        model_name: str = "custom_gesture.pkl",
    ) -> None:
        from sigil.config import MODELS_DIR

        self._mappings = [m for m in mappings if m.type == "instant" and m.enabled]
        self._artifact: dict[str, Any] | None = None
        self._clf: Any = None
        self._scaler: Any = None
        self._class_names: list[str] = []

        model_path = MODELS_DIR / model_name
        if model_path.exists():
            try:
                import joblib

                self._artifact = joblib.load(model_path)
                self._clf = self._artifact["model"]
                self._scaler = self._artifact["scaler"]
                self._class_names = self._artifact["class_names"]
                logger.info(
                    "GeometricClassifier loaded model: %s (%d classes)",
                    model_name,
                    len(self._class_names),
                )
            except Exception as e:
                logger.error("Failed to load geometric model %s: %s", model_name, e)

    def classify(self, frame_result: FrameResult) -> list[ClassificationResult]:
        results: list[ClassificationResult] = []
        if self._clf is None:
            return results

        from sigil.utils import extract_96_features

        for hand in frame_result.hands:
            try:
                features = extract_96_features(hand.landmarks)
                X = np.array([features], dtype=np.float32)
                X_scaled = self._scaler.transform(X)

                probs = self._clf.predict_proba(X_scaled)[0]
                idx = np.argmax(probs)
                label = self._class_names[idx]
                score = float(probs[idx])
                
                # Improved confidence: consider second-best prediction
                sorted_probs = np.sort(probs)[::-1]
                confidence_margin = sorted_probs[0] - sorted_probs[1] if len(sorted_probs) > 1 else sorted_probs[0]
                adjusted_score = score * (1 + confidence_margin * 0.5)

                if label == "None":
                    continue

                for mapping in self._mappings:
                    pose = mapping.condition.get("pose", "")
                    if pose == label and self._hand_matches(mapping.hand, hand.handedness):
                        pos = (float(hand.landmarks[8][0]), float(hand.landmarks[8][1]))
                        results.append(
                            ClassificationResult(
                                gesture_name=mapping.name,
                                gesture_type="instant",
                                confidence=min(adjusted_score, 1.0),
                                position=pos,
                                hand=hand.handedness,
                                raw_label=f"geometric:{label}",
                                timestamp_ms=frame_result.timestamp_ms,
                                continuous=mapping.continuous,
                                source="geometric",
                            )
                        )
                        break
            except Exception as e:
                logger.debug("Geometric classification error: %s", e)

        return results

    @staticmethod
    def _hand_matches(config_hand: str, detected: str) -> bool:
        if config_hand == "both":
            return True
        return config_hand.lower() == detected.lower()

    def close(self) -> None:
        self._clf = None
        self._scaler = None


# ═════════════════════════════════════════════════════════════════════════════
# 4.  Sequential Classifier (§5.2 – temporal multi-frame series)
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
        """Buffer landmarks for sequential analysis."""
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
                        continuous=mapping.continuous,
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
# 5.  Unified Classifier Facade — Single-Pass Dual Classification
# ═════════════════════════════════════════════════════════════════════════════
class GestureClassifier:
    """Unified facade: reads built-in gestures + runs geometric model on same landmarks.

    Single-Pass Architecture:
    - Built-in (7 classes): Read from tracker's FrameResult (zero cost)
    - Geometric (custom RF): Runs on same landmarks (~1–2ms)

    Priority / Merge Logic:
    - Custom gesture wins if confident — it's why you trained it
    - Fall back to built-in if custom is 'none' or below threshold
    - 'none' only if both are silent

    Smoothing: 7-frame buffer for stable output display.
    """

    # Minimum confidence for geometric to override builtin
    GEOMETRIC_MIN_CONFIDENCE = 0.45
    # Buffer size for smoothing
    SMOOTHING_BUFFER_SIZE = 5

    def __init__(
        self,
        mappings: list[GestureMapping],
        execution: ExecutionConfig | None = None,
        custom_instant_model: str | None = None,
        instant_inference_interval_ms: int = 0,
    ) -> None:
        # Built-in gesture reader (reads from FrameResult, no separate recognizer)
        self.instant = InstantClassifier(mappings, execution)
        # Custom geometric classifier (RF on 96-dim features)
        self.geometric = GeometricClassifier(mappings, execution)
        # Continuous feature classifiers
        self.gradual = GradualClassifier(mappings, execution)
        self.sequential = SequentialClassifier(mappings)
        self._last_fire: dict[str, int] = {}

        exec_cfg = execution or ExecutionConfig()
        self._confirm_frames = max(1, exec_cfg.confirm_frames)
        self._blanking_ms = exec_cfg.blanking_ms
        self._confidence_threshold = exec_cfg.confidence_threshold

        self._instant_streak: dict[str, int] = {}
        self._blanking_left: int = 0
        self._blanking_right: int = 0
        # Backward-compatible aggregate blanking timestamp used by existing tests.
        self._blanking_until: int = 0
        self._mappings = mappings
        self._instant_inference_interval_ms = max(0, instant_inference_interval_ms)
        self._last_instant_inference_ts = -1

        # Smoothing buffer per hand: stores recent gesture labels
        self._smoothing_buffer: dict[str, list[str]] = {
            "left": [],
            "right": [],
            "unknown": [],
        }

    @staticmethod
    def _normalize_hand_key(hand: str) -> str:
        lowered = hand.lower()
        if lowered in ("left", "right"):
            return lowered
        return "unknown"

    def _apply_priority_and_smoothing(
        self,
        builtin_results: list[ClassificationResult],
        geometric_results: list[ClassificationResult],
    ) -> list[ClassificationResult]:
        """Merge built-in and geometric results with priority logic.

        Priority rules:
        1. Custom gesture wins if it fired — it's why you trained it
        2. Fall back to built-in if custom is absent
        3. Apply 7-frame smoothing buffer for display stability
        """
        if not builtin_results and not geometric_results:
            return []

        resolved: list[ClassificationResult] = []

        for hand_key in ["left", "right", "unknown"]:
            hand_builtins = [
                r for r in builtin_results if self._normalize_hand_key(r.hand) == hand_key
            ]
            hand_geometrics = [
                r for r in geometric_results if self._normalize_hand_key(r.hand) == hand_key
            ]

            if not hand_builtins and not hand_geometrics:
                continue

            best: ClassificationResult | None = None

            if hand_geometrics:
                geo_best = max(hand_geometrics, key=lambda r: r.confidence)
                geo_conf = geo_best.confidence

                if hand_builtins:
                    builtin_best = max(hand_builtins, key=lambda r: r.confidence)
                    bp_conf = builtin_best.confidence

                    # Custom wins if above threshold AND beats built-in
                    if geo_conf >= self.GEOMETRIC_MIN_CONFIDENCE and geo_conf > bp_conf:
                        best = geo_best
                        logger.debug(
                            "Geometric wins: %.2f >= %.2f AND > built-in %.2f",
                            geo_conf,
                            self.GEOMETRIC_MIN_CONFIDENCE,
                            bp_conf,
                        )
                    else:
                        best = builtin_best
                        logger.debug(
                            "Built-in wins: Geometric %.2f < %.2f threshold or <= built-in %.2f",
                            geo_conf,
                            self.GEOMETRIC_MIN_CONFIDENCE,
                            bp_conf,
                        )
                else:
                    if geo_conf >= self.GEOMETRIC_MIN_CONFIDENCE:
                        best = geo_best
                        logger.debug(
                            "Geometric wins (no built-in): %.2f >= %.2f",
                            geo_conf,
                            self.GEOMETRIC_MIN_CONFIDENCE,
                        )
                    else:
                        logger.debug(
                            "No gesture wins: geometric %.2f below threshold %.2f",
                            geo_conf,
                            self.GEOMETRIC_MIN_CONFIDENCE,
                        )
            elif hand_builtins:
                best = max(hand_builtins, key=lambda r: r.confidence)
                logger.debug("Built-in wins (no geometric): %.2f", best.confidence)

            if best:
                buffer = self._smoothing_buffer[hand_key]
                buffer.append(best.gesture_name)
                if len(buffer) > self.SMOOTHING_BUFFER_SIZE:
                    buffer.pop(0)

                # Avoid early-frame inertia: before buffer is full, keep current label.
                if len(buffer) < self.SMOOTHING_BUFFER_SIZE:
                    smoothed_gesture = best.gesture_name
                else:
                    smoothed_gesture = max(set(buffer), key=buffer.count)

                smoothed_result = ClassificationResult(
                    gesture_name=smoothed_gesture,
                    gesture_type=best.gesture_type,
                    confidence=best.confidence,
                    position=best.position,
                    hand=best.hand,
                    raw_label=best.raw_label,
                    timestamp_ms=best.timestamp_ms,
                    continuous=best.continuous,
                    source=best.source,
                )
                resolved.append(smoothed_result)

        return resolved

    def classify(self, frame_result: FrameResult) -> list[ClassificationResult]:
        """Run all classifiers and return de-duplicated, cooldown-filtered results."""
        now = monotonic_ms()

        # 1. Read built-in gestures (zero cost — already in FrameResult)
        #    and run geometric classifier on same landmarks (~1–2ms)
        run_instant = True
        if self._instant_inference_interval_ms > 0:
            if (
                self._last_instant_inference_ts >= 0
                and frame_result.timestamp_ms - self._last_instant_inference_ts
                < self._instant_inference_interval_ms
            ):
                run_instant = False

        if run_instant:
            builtin_results = self.instant.classify(frame_result)
            self._last_instant_inference_ts = frame_result.timestamp_ms
        else:
            builtin_results = []

        geometric_results = self.geometric.classify(frame_result)

        # 2. Resolve conflicts with priority logic + smoothing
        resolved_instants = self._apply_priority_and_smoothing(builtin_results, geometric_results)

        for r in resolved_instants:
            if run_instant and r.source == "builtin":
                self.sequential.feed_instant_label(r.raw_label)
            logger.debug(
                "%s gesture: %s (label=%s, conf=%.2f, source=%s)",
                r.gesture_type.title(),
                r.gesture_name,
                r.raw_label,
                r.confidence,
                r.source,
            )

        confirmed_instants = self._apply_instant_confirmation(resolved_instants, now)
        all_results: list[ClassificationResult] = []
        all_results.extend(confirmed_instants)

        # 3. Gradual
        all_results.extend(self.gradual.classify(frame_result))

        # 4. Sequential
        all_results.extend(self.sequential.classify(frame_result))

        # Feed landmarks for sequential DTW
        for hand in frame_result.hands:
            self.sequential.feed_landmarks(hand.landmarks)

        # Filter results based on blanking and cooldowns
        filtered: list[ClassificationResult] = []

        for r in all_results:
            is_continuous = self._is_continuous(r.gesture_name)

            # ── Hand-specific Blanking filter ────────────────────────────────
            if not is_continuous:
                if r.hand.lower() == "left" and now < self._blanking_left:
                    continue
                if r.hand.lower() == "right" and now < self._blanking_right:
                    continue
                if r.hand.lower() == "both" and (
                    now < self._blanking_left or now < self._blanking_right
                ):
                    continue
                if not r.hand and now < self._blanking_until:
                    continue

            if is_continuous:
                # Continuous gestures bypass cooldown and don't trigger blanking
                filtered.append(r)
            else:
                last = self._last_fire.get(r.gesture_name, 0)
                cooldown = self._lookup_cooldown(r.gesture_name)
                if now - last >= cooldown:
                    self._last_fire[r.gesture_name] = now
                    filtered.append(r)

        # Update per-hand blanking if a discrete action fired
        for r in filtered:
            if r.gesture_type in ("instant", "sequential") and not self._is_continuous(
                r.gesture_name
            ):
                blanking_ms = self._lookup_blanking(r.gesture_name)

                # Apply blanking to the hand that triggered it
                if r.hand.lower() in ("left", "both"):
                    self._blanking_left = now + blanking_ms
                if r.hand.lower() in ("right", "both"):
                    self._blanking_right = now + blanking_ms

                # Unknown hand (e.g. mocked tests): preserve legacy global blanking behavior.
                if not r.hand:
                    self._blanking_until = now + blanking_ms
                else:
                    self._blanking_until = max(self._blanking_left, self._blanking_right)

                # We clear streaks for the triggering hand to avoid immediate repeat
                for name in list(self._instant_streak.keys()):
                    # Finding the mapping to check which hand it belongs to
                    m = next((m for m in self._mappings if m.name == name), None)
                    if m and (m.hand.lower() == r.hand.lower() or r.hand.lower() == "both"):
                        self._instant_streak[name] = 0

                logger.debug(
                    "%s action '%s' confirmed on %s hand – blanking for %d ms",
                    r.gesture_type.title(),
                    r.gesture_name,
                    r.hand,
                    blanking_ms,
                )

        return filtered

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
            required = self._lookup_confirm_frames(name)
            logger.debug(
                "Gesture '%s' streak: %d/%d", name, self._instant_streak[name], required
            )

            if self._instant_streak[name] >= required:
                logger.debug(
                    "Gesture '%s' CONFIRMED (streak=%d >= %d)",
                    name,
                    self._instant_streak[name],
                    required,
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
        for m in self._mappings:
            if m.name == gesture_name:
                return m.cooldown_ms
        return 300

    def _lookup_confirm_frames(self, gesture_name: str) -> int:
        """Find per-gesture confirm_frames override, else fallback to global."""
        for m in self._mappings:
            if m.name == gesture_name:
                if m.confirm_frames is None:
                    return self._confirm_frames
                return max(1, m.confirm_frames)
        return self._confirm_frames

    def _lookup_blanking(self, gesture_name: str) -> int:
        """Find per-gesture blanking override, else fallback to global."""
        for m in self._mappings:
            if m.name == gesture_name:
                return self._blanking_ms if m.blanking_ms is None else m.blanking_ms
        return self._blanking_ms

    def reload(
        self, mappings: list[GestureMapping], execution: ExecutionConfig | None = None
    ) -> None:
        """Update classifier mappings and execution policy."""
        self._mappings = mappings
        exec_cfg = execution or ExecutionConfig()
        self._confirm_frames = max(1, exec_cfg.confirm_frames)
        self._blanking_ms = exec_cfg.blanking_ms
        self._confidence_threshold = exec_cfg.confidence_threshold

        # Propagate to sub-classifiers
        self.instant._mappings = [m for m in mappings if m.type == "instant" and m.enabled]
        self.geometric._mappings = [m for m in mappings if m.type == "instant" and m.enabled]
        self.gradual._mappings = [m for m in mappings if m.type == "gradual" and m.enabled]
        self.sequential._mappings = [m for m in mappings if m.type == "sequential" and m.enabled]

        logger.info(
            "GestureClassifier reloaded (blanking=%dms, confirm=%df)",
            self._blanking_ms,
            self._confirm_frames,
        )

    def close(self) -> None:
        self.instant.close()
        self.geometric.close()
        self.gradual.close()
        self.sequential.close()
