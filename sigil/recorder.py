"""Gesture Recording Mode (§5.3).

Captures training data for custom gestures:
  - Instant: cropped RGB frames saved to class folders (Model Maker compatible).
  - Gradual/Sequential: landmark time-series + derived features to CSV/JSON.

Hotkey-toggled, with UI feedback via overlay callbacks.
"""

from __future__ import annotations

import csv
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import cv2

from sigil.config import RECORDINGS_DIR, RecordingConfig
from sigil.tracker import FrameResult, HandResult
from sigil.utils import finger_curl_angles, palm_center, thumb_index_distance

logger = logging.getLogger(__name__)


class RecordMode(Enum):
    INSTANT = auto()
    GRADUAL = auto()
    SEQUENTIAL = auto()


@dataclass
class RecordingSession:
    """State for an active recording session."""

    class_name: str = ""
    mode: RecordMode = RecordMode.INSTANT
    hand: str = "right"  # "left" | "right" | "both"
    sample_count: int = 0
    target_count: int = 100
    active: bool = False
    start_time: float = 0.0
    output_dir: Path = field(default_factory=lambda: RECORDINGS_DIR)

    # Buffers for sequential mode
    _seq_buffer: list[dict[str, Any]] = field(default_factory=list)
    _seq_recording: bool = False


class Recorder:
    """Manages gesture recording sessions (§5.3).

    Usage::

        recorder = Recorder(cfg)
        recorder.start("my_pinch", RecordMode.INSTANT, hand="right")
        # In frame loop:
        recorder.capture(frame_result)
        # When done:
        recorder.stop()
    """

    def __init__(self, cfg: RecordingConfig) -> None:
        self._cfg = cfg
        self._session = RecordingSession()
        self._on_status: Callable[[str], None] | None = None  # overlay callback

    # ── Properties ───────────────────────────────────────────────────────────
    @property
    def is_active(self) -> bool:
        return self._session.active

    @property
    def session(self) -> RecordingSession:
        return self._session

    def set_status_callback(self, cb: Callable[[str], None]) -> None:
        self._on_status = cb

    # ── Session lifecycle ────────────────────────────────────────────────────
    def start(
        self,
        class_name: str,
        mode: RecordMode,
        hand: str = "right",
        target_count: int | None = None,
    ) -> None:
        """Begin a new recording session for *class_name*."""
        out = Path(self._cfg.output_dir) / class_name
        out.mkdir(parents=True, exist_ok=True)

        self._session = RecordingSession(
            class_name=class_name,
            mode=mode,
            hand=hand,
            sample_count=_count_existing(out, mode),
            target_count=target_count or self._cfg.samples_per_class,
            active=True,
            start_time=time.monotonic(),
            output_dir=out,
        )
        self._emit(
            f"Recording '{class_name}' ({mode.name}) – "
            f"{self._session.sample_count}/{self._session.target_count}"
        )
        logger.info(
            "Recording started: class=%s mode=%s hand=%s",
            class_name,
            mode.name,
            hand,
        )

    def stop(self) -> None:
        """End the current recording session."""
        if not self._session.active:
            return
        # Flush sequential buffer if any
        if self._session.mode == RecordMode.SEQUENTIAL and self._session._seq_buffer:
            self._save_sequence()
        self._session.active = False
        self._emit(
            f"Recording stopped: {self._session.sample_count} samples saved"
        )
        logger.info(
            "Recording stopped: class=%s samples=%d",
            self._session.class_name,
            self._session.sample_count,
        )

    def discard_last(self) -> None:
        """Delete the most recent sample (§5.3 discard/redo)."""
        s = self._session
        if not s.active or s.sample_count <= 0:
            return

        if s.mode == RecordMode.INSTANT:
            # Remove last image
            images = sorted(s.output_dir.glob("*.jpg"))
            if images:
                images[-1].unlink()
                s.sample_count -= 1
                self._emit(f"Discarded last sample ({s.sample_count}/{s.target_count})")
        elif s.mode in (RecordMode.GRADUAL, RecordMode.SEQUENTIAL):
            # Remove last JSON sequence
            jsons = sorted(s.output_dir.glob("*.json"))
            if jsons:
                jsons[-1].unlink()
                s.sample_count -= 1
                self._emit(f"Discarded last sample ({s.sample_count}/{s.target_count})")

    # ── Per-frame capture ────────────────────────────────────────────────────
    def capture(self, frame_result: FrameResult) -> bool:
        """Capture one frame of data. Returns True if sample saved."""
        s = self._session
        if not s.active:
            return False
        if s.sample_count >= s.target_count:
            self._emit(f"Target reached ({s.target_count}). Stop or increase target.")
            return False

        hand = self._pick_hand(frame_result)
        if hand is None:
            return False

        if s.mode == RecordMode.INSTANT:
            return self._capture_instant(frame_result, hand)
        elif s.mode == RecordMode.GRADUAL:
            return self._capture_gradual(frame_result, hand)
        elif s.mode == RecordMode.SEQUENTIAL:
            return self._capture_sequential(frame_result, hand)
        return False

    # ── Instant capture ──────────────────────────────────────────────────────
    def _capture_instant(self, fr: FrameResult, hand: HandResult) -> bool:
        """Save annotated RGB frame to class folder (Model Maker compatible §5.3)."""
        if fr.frame is None:
            return False

        s = self._session
        # Crop around hand bounding box with margin
        h, w = fr.frame.shape[:2]
        xs = hand.landmarks[:, 0] * w
        ys = hand.landmarks[:, 1] * h
        margin = 40
        x1 = max(0, int(xs.min()) - margin)
        y1 = max(0, int(ys.min()) - margin)
        x2 = min(w, int(xs.max()) + margin)
        y2 = min(h, int(ys.max()) + margin)

        crop = fr.frame[y1:y2, x1:x2]
        if crop.size == 0:
            return False

        fname = s.output_dir / f"{s.sample_count:05d}.jpg"
        cv2.imwrite(str(fname), crop)
        s.sample_count += 1
        self._emit(f"[instant] {s.sample_count}/{s.target_count}")
        return True

    # ── Gradual capture ──────────────────────────────────────────────────────
    def _capture_gradual(self, fr: FrameResult, hand: HandResult) -> bool:
        """Save normalised landmark features for one frame + derived metrics."""
        s = self._session
        features = self._extract_features(fr, hand)

        fname = s.output_dir / f"{s.sample_count:05d}.json"
        with open(fname, "w") as fh:
            json.dump(features, fh)

        # Also append to CSV for Model Maker / sklearn convenience
        csv_path = s.output_dir / "features.csv"
        is_new = not csv_path.exists()
        with open(csv_path, "a", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=sorted(features.keys()))
            if is_new:
                writer.writeheader()
            writer.writerow(features)

        s.sample_count += 1
        self._emit(f"[gradual] {s.sample_count}/{s.target_count}")
        return True

    # ── Sequential capture ───────────────────────────────────────────────────
    def _capture_sequential(self, fr: FrameResult, hand: HandResult) -> bool:
        """Buffer landmark frames; save full sequence on stop or target hit."""
        s = self._session
        features = self._extract_features(fr, hand)
        features["frame_index"] = len(s._seq_buffer)
        s._seq_buffer.append(features)

        # Auto-save every 30 frames as one sequence sample
        if len(s._seq_buffer) >= 30:
            self._save_sequence()
            return True
        return False

    def _save_sequence(self) -> None:
        s = self._session
        if not s._seq_buffer:
            return
        fname = s.output_dir / f"seq_{s.sample_count:05d}.json"
        with open(fname, "w") as fh:
            json.dump(s._seq_buffer, fh)
        s._seq_buffer.clear()
        s.sample_count += 1
        self._emit(f"[sequential] {s.sample_count}/{s.target_count}")

    # ── Feature extraction ───────────────────────────────────────────────────
    @staticmethod
    def _extract_features(fr: FrameResult, hand: HandResult) -> dict[str, Any]:
        lm = hand.landmarks
        curls = finger_curl_angles(lm)
        center = palm_center(lm)
        return {
            "timestamp_ms": fr.timestamp_ms,
            "handedness": hand.handedness,
            "thumb_index_dist": float(thumb_index_distance(lm)),
            "palm_x": float(center[0]),
            "palm_y": float(center[1]),
            "curl_thumb": float(curls[0]),
            "curl_index": float(curls[1]),
            "curl_middle": float(curls[2]),
            "curl_ring": float(curls[3]),
            "curl_pinky": float(curls[4]),
            # Flattened normalised landmarks (63 values)
            **{f"lm_{i}_{c}": float(lm[i][j]) for i in range(21) for j, c in enumerate("xyz")},
        }

    # ── Helpers ──────────────────────────────────────────────────────────────
    def _pick_hand(self, fr: FrameResult) -> HandResult | None:
        s = self._session
        if s.hand == "left":
            return fr.left
        if s.hand == "right":
            return fr.right
        # both – prefer right
        return fr.right or fr.left

    def _emit(self, msg: str) -> None:
        logger.debug("Recorder: %s", msg)
        if self._on_status:
            self._on_status(msg)


def _count_existing(directory: Path, mode: RecordMode) -> int:
    """Count already-saved samples in directory."""
    if mode == RecordMode.INSTANT:
        return len(list(directory.glob("*.jpg")))
    return len(list(directory.glob("*.json")))
