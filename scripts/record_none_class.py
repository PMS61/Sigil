#!/usr/bin/env python3
"""Quick utility to record 'None' (background/noise) gesture samples.

This script opens your webcam and lets you record background hand samples
with a single keypress. These samples teach the model what NOT to classify
as a gesture, reducing false positives.

Usage:
    python scripts/record_none_class.py

Controls:
    SPACE    - Record current frame as a 'None' sample
    ESC / q  - Quit
    s        - Stop recording session

Samples are saved to ~/.local/share/sigil/recordings/None/
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sigil.config import RECORDINGS_DIR, RecordingConfig, TrackingConfig, load_config
from sigil.tracker import FrameResult, Tracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

NONE_DIR = RECORDINGS_DIR / "None"
TARGET_SAMPLES = 150  # Recommended: 100-200 for good coverage


def count_existing() -> int:
    """Count already-recorded None samples."""
    if not NONE_DIR.exists():
        return 0
    return len(list(NONE_DIR.glob("*.jpg"))) + len(list(NONE_DIR.glob("*.json"))) // 2


def save_sample(frame_result: FrameResult, sample_idx: int) -> Path | None:
    """Save a cropped hand image + landmarks JSON for the 'None' class."""
    if frame_result.frame is None:
        return None

    # Pick whichever hand is visible
    hand = frame_result.right or frame_result.left
    if hand is None:
        return None

    NONE_DIR.mkdir(parents=True, exist_ok=True)

    h, w = frame_result.frame.shape[:2]
    xs = hand.landmarks[:, 0] * w
    ys = hand.landmarks[:, 1] * h
    margin = 40
    x1 = max(0, int(xs.min()) - margin)
    y1 = max(0, int(ys.min()) - margin)
    x2 = min(w, int(xs.max()) + margin)
    y2 = min(h, int(ys.max()) + margin)

    crop = frame_result.frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    # Save cropped image
    img_path = NONE_DIR / f"{sample_idx:05d}.jpg"
    cv2.imwrite(str(img_path), crop)

    # Save raw landmarks
    landmarks_data = {
        "timestamp_ms": frame_result.timestamp_ms,
        "handedness": hand.handedness,
        "landmarks": hand.landmarks.tolist(),
    }
    json_path = NONE_DIR / f"{sample_idx:05d}.json"
    with open(json_path, "w") as fh:
        json.dump(landmarks_data, fh)

    return img_path


def main() -> None:
    # Load config
    try:
        cfg = load_config()
    except Exception:
        logger.warning("Could not load config, using defaults")
        cfg = type("MockCfg", (), {
            "tracking": TrackingConfig(),
            "recording": RecordingConfig(),
        })()

    existing = count_existing()
    print(f"\n{'=' * 60}")
    print(f"  'None' Class Recorder — Background/Noise Samples")
    print(f"{'=' * 60}")
    print(f"  Existing samples: {existing}")
    print(f"  Target:           {TARGET_SAMPLES}")
    print(f"  Remaining:        {max(0, TARGET_SAMPLES - existing)}")
    print(f"  Output dir:       {NONE_DIR}")
    print(f"\n  Controls:")
    print(f"    SPACE  – Record sample")
    print(f"    ESC/q  – Quit")
    print(f"    s      – Save session and exit")
    print(f"{'=' * 60}\n")

    # Initialise tracker
    tracker = Tracker(cfg.tracking)
    tracker.open()

    sample_idx = existing
    recorded_this_session = 0

    print("Camera opened. Press SPACE to record samples.")

    try:
        while True:
            # Run tracking (read() handles camera capture internally)
            frame_result = tracker.read()
            if frame_result is None:
                logger.error("Camera read failed")
                break

            display_frame = frame_result.frame if frame_result.frame is not None else frame_result.rgb_frame
            if display_frame is None:
                continue

            # Draw landmarks if hands detected
            for hand in frame_result.hands:
                for lm in hand.landmarks:
                    x, y = int(lm[0] * display_frame.shape[1]), int(lm[1] * display_frame.shape[0])
                    cv2.circle(display_frame, (x, y), 3, (0, 255, 0), -1)

            # Overlay HUD
            status_color = (0, 255, 0) if frame_result.hands else (0, 0, 255)
            status_text = "HAND DETECTED — Press SPACE" if frame_result.hands else "No hand detected"
            cv2.putText(
                display_frame,
                status_text,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.putText(
                display_frame,
                f"Samples: {sample_idx} / {TARGET_SAMPLES}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Record 'None' Class — SPACE to capture, ESC/q to quit", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord("q"):
                print(f"\nQuit. Recorded {recorded_this_session} new samples this session.")
                break
            elif key == ord("s"):
                print(f"\nSession saved. Recorded {recorded_this_session} samples.")
                break
            elif key == 32:  # SPACE
                if not frame_result.hands:
                    print("⚠  No hand detected — show your hand and try again")
                    continue

                path = save_sample(frame_result, sample_idx)
                if path:
                    sample_idx += 1
                    recorded_this_session += 1
                    remaining = max(0, TARGET_SAMPLES - sample_idx)
                    print(f"✓ Sample {sample_idx}/{TARGET_SAMPLES} saved ({remaining} remaining)")

                    # Flash feedback
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 40), (0, 0, 255), -1)
                    cv2.putText(
                        display_frame,
                        f"SAVED #{sample_idx}",
                        (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Record 'None' Class — SPACE to capture, ESC/q to quit", display_frame)
                    cv2.waitKey(100)  # Brief flash
                else:
                    print("⚠  Could not save sample (no hand crop)")

    except KeyboardInterrupt:
        print(f"\nInterrupted. Recorded {recorded_this_session} samples.")
    finally:
        tracker.close()
        cv2.destroyAllWindows()

    if sample_idx >= TARGET_SAMPLES:
        print(f"\n✅ Target reached! {sample_idx} 'None' samples recorded.")
        print("   You can now train your model with: sigil train")
    elif sample_idx > existing:
        print(f"\n📊 {sample_idx} total 'None' samples ({recorded_this_session} new).")
        print("   Record more or train with: sigil train")
    else:
        print(f"\n⚠️  No new samples recorded. Need {TARGET_SAMPLES} total.")


if __name__ == "__main__":
    main()
