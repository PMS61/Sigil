"""Visualization Overlay (§5.6) – facade.

Auto-selects the best available backend:
  1. GTK4 + gtk4-layer-shell (native Wayland overlay)
  2. GTK4 regular window (Wayland/X11 with no layer-shell)
  3. OpenCV cv2.imshow fallback (XWayland)

The public API (``draw``, ``close``, ``set_recording_status``) is identical
regardless of the active backend.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

from sigil.ui import get_backend

if TYPE_CHECKING:
    from sigil.classifier import ClassificationResult
    from sigil.config import OverlayConfig
    from sigil.tracker import FrameResult, HandResult

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  OpenCV fallback (legacy)
# ══════════════════════════════════════════════════════════════════════════════
_HAND_COLORS = {
    "Left": (255, 128, 0),   # orange
    "Right": (0, 200, 255),  # cyan
}
_LANDMARK_RADIUS = 3
_CONNECTION_THICKNESS = 2

_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


class _OpenCVOverlay:
    """OpenCV-backed overlay (legacy fallback)."""

    WINDOW_NAME = "Sigil"

    def __init__(self) -> None:
        self._recording_status: str = ""

    def set_recording_status(self, status: str) -> None:
        self._recording_status = status

    def draw(
        self,
        frame_result: FrameResult,
        classifications: list[ClassificationResult] | None = None,
        fps: float = 0.0,
    ) -> bool:
        if frame_result.frame is None:
            return True

        canvas = frame_result.frame.copy()
        for hand in frame_result.hands:
            self._draw_hand(canvas, hand)

        h, w = canvas.shape[:2]
        cv2.rectangle(canvas, (0, 0), (w, 80), (20, 20, 20), -1)

        cv2.putText(canvas, f"FPS: {fps:.1f}", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)

        hands_str = ", ".join(f"{hd.handedness}({hd.score:.2f})" for hd in frame_result.hands)
        cv2.putText(canvas, f"Hands: {hands_str or 'none'}", (10, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        if classifications:
            label_str = " | ".join(
                f"{c.gesture_name} ({c.confidence:.2f})" for c in classifications
            )
            cv2.putText(canvas, label_str, (10, 75),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        if self._recording_status:
            cv2.putText(canvas, f"REC: {self._recording_status}", (w - 400, 25),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        cv2.imshow(self.WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        return key != 27

    def _draw_hand(self, canvas: np.ndarray, hand: HandResult) -> None:
        h, w = canvas.shape[:2]
        color = _HAND_COLORS.get(hand.handedness, (255, 255, 255))
        lm = hand.landmarks
        pts = np.zeros((21, 2), dtype=np.int32)
        for i in range(21):
            pts[i] = (int(lm[i][0] * w), int(lm[i][1] * h))
        for a, b in _CONNECTIONS:
            cv2.line(canvas, tuple(pts[a]), tuple(pts[b]), color, _CONNECTION_THICKNESS)
        for i in range(21):
            cv2.circle(canvas, tuple(pts[i]), _LANDMARK_RADIUS, color, -1)
        cv2.putText(canvas, hand.handedness, (pts[0][0] - 20, pts[0][1] - 15),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    def close(self) -> None:
        cv2.destroyAllWindows()


# ══════════════════════════════════════════════════════════════════════════════
#  Public facade
# ══════════════════════════════════════════════════════════════════════════════
class Overlay:
    """Auto-selecting overlay facade.

    Uses the best available backend (GTK4-layer-shell → GTK4 → OpenCV).
    When using the Wayland backend, ``draw()`` is a no-op because frame
    processing is driven by the GTK main loop instead.  Call
    ``get_wayland_overlay()`` to obtain the underlying ``WaylandOverlay``
    for direct integration.
    """

    def __init__(
        self,
        enabled: bool = True,
        *,
        force_backend: str | None = None,
        overlay_cfg: OverlayConfig | None = None,
    ) -> None:
        self._enabled = enabled

        # Resolve backend
        self._backend_name = force_backend or get_backend()

        # Import defaults for overlay settings
        from sigil.config import OverlayConfig

        ocfg = overlay_cfg or OverlayConfig()

        self._wayland: Any = None
        self._opencv: _OpenCVOverlay | None = None

        if self._backend_name in ("gtk4-layer-shell", "gtk4") and self._enabled:
            try:
                from sigil.ui.wayland_overlay import WaylandOverlay

                self._wayland = WaylandOverlay(
                    enabled=enabled,
                    cam_width=ocfg.width,
                    cam_height=ocfg.height,
                    position=ocfg.position,
                    margin=ocfg.margin,
                    opacity=ocfg.opacity,
                    show_camera=ocfg.camera_preview,
                )
                logger.info("Overlay backend: %s", self._backend_name)
            except Exception:
                logger.warning("GTK4 overlay init failed – falling back to OpenCV", exc_info=True)
                self._backend_name = "opencv"
                self._wayland = None

        if self._wayland is None:
            self._opencv = _OpenCVOverlay()
            self._backend_name = "opencv"
            if self._enabled:
                logger.info("Overlay backend: opencv (fallback)")

    @property
    def backend(self) -> str:
        return self._backend_name

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def is_wayland(self) -> bool:
        return self._wayland is not None

    def get_wayland_overlay(self) -> Any:
        """Return the underlying WaylandOverlay, or None."""
        return self._wayland

    def set_enabled(self, val: bool) -> None:
        self._enabled = val
        if self._wayland:
            self._wayland.set_enabled(val)
        if not val and self._opencv:
            cv2.destroyAllWindows()

    def set_recording_status(self, status: str) -> None:
        if self._wayland:
            self._wayland.set_recording_status(status)
        elif self._opencv:
            self._opencv.set_recording_status(status)

    def draw(
        self,
        frame_result: FrameResult,
        classifications: list[ClassificationResult] | None = None,
        fps: float = 0.0,
    ) -> bool:
        """Draw one frame.  Returns False if the user closed the window.

        When using the Wayland backend, this is a passthrough (returns True)
        because frame updates are driven by ``WaylandOverlay.update()`` from
        the GLib timer.
        """
        if not self._enabled:
            return True

        if self._wayland:
            # GTK path: overlay.update() is called from the GLib tick in daemon
            return True

        if self._opencv:
            return self._opencv.draw(frame_result, classifications, fps)

        return True

    def close(self) -> None:
        if self._wayland:
            self._wayland.close()
            self._wayland = None
        if self._opencv:
            self._opencv.close()
            self._opencv = None
