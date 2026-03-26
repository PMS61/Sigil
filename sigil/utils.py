"""Utility helpers for Sigil."""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np

from sigil.config import MODELS_DIR

logger = logging.getLogger(__name__)

# ── Model download URLs (MediaPipe official) ─────────────────────────────────
MODEL_URLS: dict[str, str] = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "gesture_recognizer.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
    ),
}


def ensure_model(filename: str) -> Path:
    """Download a MediaPipe model to MODELS_DIR on first run (§7)."""
    dest = MODELS_DIR / filename
    if dest.exists():
        return dest

    url = MODEL_URLS.get(filename)
    if url is None:
        raise FileNotFoundError(
            f"No download URL for '{filename}'. Place it manually in {MODELS_DIR}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s …", filename, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 – trusted Google domain
    logger.info("Download complete: %s (%d bytes)", filename, dest.stat().st_size)
    return dest


# ── Landmark math ────────────────────────────────────────────────────────────
def landmark_to_array(landmarks: list) -> np.ndarray:  # type: ignore[type-arg]
    """Convert MediaPipe NormalizedLandmarkList to (21, 3) float32 array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def thumb_index_distance(landmarks: np.ndarray) -> float:
    """Normalised thumb-tip (4) to index-tip (8) distance."""
    return euclidean(landmarks[4], landmarks[8])


def palm_center(landmarks: np.ndarray) -> np.ndarray:
    """Average of wrist (0) and middle-finger MCP (9) as palm center."""
    return (landmarks[0] + landmarks[9]) / 2.0  # type: ignore[no-any-return]


def finger_curl_angles(landmarks: np.ndarray) -> np.ndarray:
    """Curl angle proxy for each finger (5 values, 0=extended, 1=fully curled)."""
    wrist = landmarks[0]
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [1, 5, 9, 13, 17]
    curls = np.zeros(5, dtype=np.float32)
    for i, (tip, mcp) in enumerate(zip(finger_tips, finger_mcps, strict=False)):
        mcp_to_wrist = euclidean(landmarks[mcp], wrist) + 1e-7
        tip_to_mcp = euclidean(landmarks[tip], landmarks[mcp])
        ratio = tip_to_mcp / mcp_to_wrist
        curls[i] = max(0.0, min(1.0, 1.0 - ratio))
    return curls


def count_extended_fingers(landmarks: np.ndarray) -> int:
    """Returns number of fingers extended (0-4, excludes thumb).

    Only counts index, middle, ring, and pinky fingers.
    Thumb detection is unreliable for touchpad gestures.
    """
    curls = finger_curl_angles(landmarks)
    # Only count fingers 1-4 (index, middle, ring, pinky), skip thumb (index 0)
    # Threshold: curl < 0.5 is roughly extended
    extended = curls[1:] < 0.5  # Skip thumb
    return int(np.sum(extended))


def hand_velocity(prev: np.ndarray | None, curr: np.ndarray, dt: float) -> float:
    """Palm velocity in normalised-coord units / second."""
    if prev is None or dt <= 0:
        return 0.0
    return euclidean(palm_center(curr), palm_center(prev)) / dt


# ── Timing helpers ───────────────────────────────────────────────────────────
def monotonic_ms() -> int:
    """Current monotonic time in milliseconds."""
    return int(time.monotonic() * 1000)


def file_hash(path: Path) -> str:
    """SHA-256 hex digest of a file (for cache checks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
