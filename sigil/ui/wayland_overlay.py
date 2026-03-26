"""Wayland-native overlay using GTK4 + gtk4-layer-shell.

Creates a compact, semi-transparent panel anchored to a screen corner.
Shows live camera feed with hand landmarks (Cairo), gesture status,
recording progress, and action toasts.

Falls back to a regular GTK4 window if layer-shell is unavailable.

System packages (Arch):
  pacman -S gtk4 gtk4-layer-shell python-gobject python-cairo
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

# Now safe to import cv2/numpy
import cv2
import numpy as np

# Import sigil.ui FIRST to ensure gtk4-layer-shell is preloaded before cv2/GTK
from sigil.ui import HAS_CAIRO, HAS_GTK4, HAS_LAYER_SHELL
from sigil.ui.drawing import (
    COLORS,
    draw_hand,
    draw_rounded_rect,
)

if TYPE_CHECKING:
    from sigil.classifier import ClassificationResult
    from sigil.tracker import FrameResult

logger = logging.getLogger(__name__)

if not HAS_GTK4:
    raise ImportError("GTK4 is required for wayland_overlay – install python-gobject and gtk4")

import gi  # type: ignore[import-untyped] # noqa: E402

gi.require_version("Gtk", "4.0")  # noqa: E402
gi.require_version("Gdk", "4.0")  # noqa: E402
from gi.repository import Gdk, GLib, Gtk  # type: ignore[import-untyped] # noqa: E402

if HAS_LAYER_SHELL:
    gi.require_version("Gtk4LayerShell", "1.0")
    from gi.repository import Gtk4LayerShell

if HAS_CAIRO:
    import cairo


# ── Constants ────────────────────────────────────────────────────────────────
_STYLES_CSS = Path(__file__).parent / "styles.css"

# Edge name → Gtk4LayerShell.Edge pair
_ANCHOR_MAP: dict[str, list[str]] = {
    "bottom-right": ["BOTTOM", "RIGHT"],
    "bottom-left": ["BOTTOM", "LEFT"],
    "top-right": ["TOP", "RIGHT"],
    "top-left": ["TOP", "LEFT"],
}


# ── Toast entry ──────────────────────────────────────────────────────────────
class _Toast:
    """A transient status message."""

    __slots__ = ("text", "expire_time", "color")

    def __init__(self, text: str, duration_s: float = 2.0, color: str = "green") -> None:
        self.text = text
        self.expire_time = time.monotonic() + duration_s
        self.color = color

    @property
    def alive(self) -> bool:
        return time.monotonic() < self.expire_time

    @property
    def alpha(self) -> float:
        remaining = self.expire_time - time.monotonic()
        if remaining > 0.5:
            return 1.0
        return max(0.0, remaining / 0.5)


# ══════════════════════════════════════════════════════════════════════════════
#  Overlay Window
# ══════════════════════════════════════════════════════════════════════════════
class WaylandOverlay:
    """GTK4 + layer-shell overlay window for Sigil.

    Parameters
    ----------
    enabled : bool
        Whether the overlay is initially visible.
    cam_width, cam_height : int
        Size of the camera preview in the overlay.
    position : str
        Anchor position: "bottom-right", "bottom-left", "top-right", "top-left".
    margin : int
        Pixel margin from screen edge.
    opacity : float
        Base window opacity (0.0–1.0), layered on top of CSS alpha.
    show_camera : bool
        Whether to show the camera feed preview.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        cam_width: int = 320,
        cam_height: int = 180,
        position: str = "bottom-right",
        margin: int = 16,
        opacity: float = 1.0,
        show_camera: bool = True,
    ) -> None:
        self._enabled = enabled
        self._cam_w = cam_width
        self._cam_h = cam_height
        self._position = position
        self._margin = margin
        self._opacity = opacity
        self._show_camera = show_camera and HAS_CAIRO

        # State
        self._current_frame: FrameResult | None = None
        self._classifications: list[ClassificationResult] = []
        self._fps: float = 0.0
        self._recording_status: str = ""
        self._recording_progress: float = 0.0
        self._toasts: list[_Toast] = []
        self._is_recording: bool = False
        self._current_mode: str = "touchpad"  # "touchpad" | "keybind"

        # Cairo surface for camera (reused across frames)
        self._cam_surface: Any = None
        if HAS_CAIRO:
            self._cam_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, self._cam_w, self._cam_h)

        # GTK widgets (built in _build_ui)
        self._app: Gtk.Application | None = None
        self._window: Gtk.Window | None = None
        self._camera_area: Gtk.DrawingArea | None = None
        self._lbl_fps: Gtk.Label | None = None
        self._lbl_hands: Gtk.Label | None = None
        self._lbl_gesture: Gtk.Label | None = None
        self._lbl_action: Gtk.Label | None = None
        self._recording_box: Gtk.Box | None = None
        self._lbl_rec: Gtk.Label | None = None
        self._lbl_rec_count: Gtk.Label | None = None
        self._progress_bar: Gtk.ProgressBar | None = None
        self._toast_box: Gtk.Box | None = None
        self._mode_badge: Gtk.Label | None = None

    # ── Properties ───────────────────────────────────────────────────────────
    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, val: bool) -> None:
        self._enabled = val
        if self._window:
            self._window.set_visible(val)

    def set_recording_status(self, status: str) -> None:
        """Update recording-mode status (called by Recorder)."""
        self._recording_status = status
        self._is_recording = bool(status)

    def set_recording_progress(self, current: int, total: int) -> None:
        """Update recording progress bar."""
        self._recording_progress = current / max(total, 1)

    def add_toast(self, text: str, duration: float = 2.0, color: str = "green") -> None:
        """Show a transient notification in the overlay."""
        self._toasts.append(_Toast(text, duration, color))
        # Keep at most 3
        if len(self._toasts) > 3:
            self._toasts = self._toasts[-3:]

    # ── GTK Application ──────────────────────────────────────────────────────
    def create_window(self, app: Gtk.Application) -> Gtk.Window:
        """Build the overlay window and all widgets. Called during app activation."""
        self._app = app
        self._window = Gtk.Window(application=app)
        self._window.set_title("Sigil")
        self._window.set_default_size(self._cam_w + 24, self._cam_h + 200)
        self._window.set_resizable(False)
        self._window.set_decorated(False)
        logger.debug("Created GTK window with size %dx%d", self._cam_w + 24, self._cam_h + 200)

        # ── Layer-shell setup ────────────────────────────────────────────────
        self._layer_shell_active = False
        if HAS_LAYER_SHELL:
            try:
                Gtk4LayerShell.init_for_window(self._window)
                # Verify the window was actually initialized as a layer surface
                if Gtk4LayerShell.is_layer_window(self._window):
                    Gtk4LayerShell.set_layer(self._window, Gtk4LayerShell.Layer.OVERLAY)
                    # Use "gtk4-layer-shell" namespace to match Hyprland rules
                    # This ensures proper blur and alpha handling
                    Gtk4LayerShell.set_namespace(self._window, "gtk4-layer-shell")
                    Gtk4LayerShell.set_exclusive_zone(self._window, -1)

                    # Keyboard interactivity: NONE means click-through
                    Gtk4LayerShell.set_keyboard_mode(self._window, Gtk4LayerShell.KeyboardMode.NONE)

                    # Anchor to configured corner
                    anchors = _ANCHOR_MAP.get(self._position, ["BOTTOM", "RIGHT"])
                    for edge_name in anchors:
                        edge = getattr(Gtk4LayerShell.Edge, edge_name)
                        Gtk4LayerShell.set_anchor(self._window, edge, True)
                        Gtk4LayerShell.set_margin(self._window, edge, self._margin)

                    self._layer_shell_active = True
                    logger.info("Layer-shell overlay initialised on %s", self._position)
                else:
                    logger.warning(
                        "gtk4-layer-shell: init_for_window did not create a layer surface. "
                        "Falling back to regular GTK4 window. "
                        "Try: LD_PRELOAD=/usr/lib/libgtk4-layer-shell.so sigil run"
                    )
            except Exception:
                logger.warning(
                    "Layer-shell setup failed – falling back to regular window",
                    exc_info=True,
                )

        if not self._layer_shell_active:
            # Regular GTK4 window fallback – position near screen corner
            self._window.set_decorated(False)
            # GTK4 doesn't have set_keep_above, but we can use set_keep_above
            # via the native Wayland interface if needed
            logger.warning("Using regular GTK4 window (not layer-shell)")
            logger.warning(
                "Note: Window may appear behind other applications."
                " Use layer-shell for best experience."
            )
            logger.warning(
                "For better results, run with: LD_PRELOAD=/usr/lib/libgtk4-layer-shell.so sigil run"
            )

        # ── CSS ──────────────────────────────────────────────────────────────
        self._load_css()
        self._window.add_css_class("sigil-window")

        # ── Widget tree ──────────────────────────────────────────────────────
        self._build_ui()

        if self._enabled:
            logger.info("Presenting overlay window (enabled=%s)", self._enabled)
            self._window.present()
        else:
            logger.debug("Overlay window created but not visible (enabled=False)")

        return self._window

    def _load_css(self) -> None:
        """Load the stylesheet."""
        provider = Gtk.CssProvider()
        if _STYLES_CSS.exists():
            provider.load_from_path(str(_STYLES_CSS))
        else:
            logger.warning("CSS file not found: %s", _STYLES_CSS)
            provider.load_from_string(
                ".sigil-panel { background: rgba(30,30,46,0.88); "
                "border-radius: 16px; padding: 12px; }"
            )
        Gtk.StyleContext.add_provider_for_display(
            Gdk.Display.get_default(),
            provider,
            Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
        )

    def _build_ui(self) -> None:
        """Construct the widget hierarchy."""
        assert self._window is not None

        # Root box
        panel = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        panel.add_css_class("sigil-panel")
        self._window.set_child(panel)

        # ── Camera preview ───────────────────────────────────────────────────
        if self._show_camera:
            self._camera_area = Gtk.DrawingArea()
            self._camera_area.set_content_width(self._cam_w)
            self._camera_area.set_content_height(self._cam_h)
            self._camera_area.add_css_class("sigil-camera")
            self._camera_area.set_draw_func(self._on_camera_draw)
            panel.append(self._camera_area)

        # ── Status row (FPS + hands) ─────────────────────────────────────────
        status_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        status_row.add_css_class("sigil-status")
        panel.append(status_row)

        # Mode badge
        self._mode_badge = Gtk.Label(label="TRACK")
        self._mode_badge.add_css_class("sigil-mode-badge")
        status_row.append(self._mode_badge)

        self._lbl_fps = Gtk.Label(label="0 fps")
        self._lbl_fps.add_css_class("sigil-fps")
        self._lbl_fps.set_hexpand(True)
        self._lbl_fps.set_halign(Gtk.Align.END)
        status_row.append(self._lbl_fps)

        # Hands label
        self._lbl_hands = Gtk.Label(label="No hands")
        self._lbl_hands.add_css_class("sigil-hands")
        self._lbl_hands.set_halign(Gtk.Align.START)
        panel.append(self._lbl_hands)

        # ── Separator ────────────────────────────────────────────────────────
        sep = Gtk.Separator()
        sep.add_css_class("sigil-separator")
        panel.append(sep)

        # ── Gesture info ─────────────────────────────────────────────────────
        self._lbl_gesture = Gtk.Label(label="Waiting for gesture…")
        self._lbl_gesture.add_css_class("sigil-gesture-none")
        self._lbl_gesture.set_halign(Gtk.Align.START)
        self._lbl_gesture.set_wrap(True)
        panel.append(self._lbl_gesture)

        self._lbl_action = Gtk.Label(label="")
        self._lbl_action.add_css_class("sigil-action")
        self._lbl_action.set_halign(Gtk.Align.START)
        self._lbl_action.set_visible(False)
        panel.append(self._lbl_action)

        # ── Recording section (hidden by default) ────────────────────────────
        self._recording_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._recording_box.set_visible(False)
        panel.append(self._recording_box)

        rec_sep = Gtk.Separator()
        rec_sep.add_css_class("sigil-separator")
        self._recording_box.append(rec_sep)

        rec_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._recording_box.append(rec_header)

        self._lbl_rec = Gtk.Label(label="● REC")
        self._lbl_rec.add_css_class("sigil-recording-label")
        rec_header.append(self._lbl_rec)

        self._lbl_rec_count = Gtk.Label(label="0/100")
        self._lbl_rec_count.add_css_class("sigil-recording-count")
        self._lbl_rec_count.set_hexpand(True)
        self._lbl_rec_count.set_halign(Gtk.Align.END)
        rec_header.append(self._lbl_rec_count)

        self._progress_bar = Gtk.ProgressBar()
        self._progress_bar.set_fraction(0.0)
        self._progress_bar.add_css_class("sigil-progress-trough")
        self._recording_box.append(self._progress_bar)

        # ── Toast area ───────────────────────────────────────────────────────
        self._toast_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        panel.append(self._toast_box)

    # ── Drawing (Cairo) ──────────────────────────────────────────────────────
    def _on_camera_draw(
        self,
        area: Gtk.DrawingArea,
        cr: Any,  # cairo.Context
        width: int,
        height: int,
    ) -> None:
        """Draw the camera feed + landmarks onto the DrawingArea."""
        if not HAS_CAIRO:
            return

        fr = self._current_frame

        # Background if no frame
        if fr is None or fr.frame is None:
            cr.set_source_rgba(0.067, 0.067, 0.106, 0.95)
            draw_rounded_rect(cr, 0, 0, width, height, 12)
            cr.fill()
            # "No camera" text
            cr.set_source_rgba(*COLORS["subtext0"][:3], 0.6)
            cr.select_font_face("Sans", 0, 0)
            cr.set_font_size(13)
            ext = cr.text_extents("No camera feed")
            cr.move_to((width - ext.width) / 2, (height + ext.height) / 2)
            cr.show_text("No camera feed")
            return

        # Clip to rounded rect
        draw_rounded_rect(cr, 0, 0, width, height, 12)
        cr.clip()

        # Draw camera feed
        if self._cam_surface:
            cr.set_source_surface(self._cam_surface, 0, 0)
            cr.paint()

        # ── Dim overlay for better landmark visibility ───────────────────────
        cr.set_source_rgba(0, 0, 0, 0.15)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        # ── Draw hand landmarks ──────────────────────────────────────────────
        for hand in fr.hands:
            draw_hand(cr, hand, width, height)

        # ── Recording border pulse ───────────────────────────────────────────
        if self._is_recording:
            pulse = 0.5 + 0.5 * math.sin(time.monotonic() * 4)
            cr.set_source_rgba(0.953, 0.545, 0.659, 0.3 + 0.4 * pulse)
            cr.set_line_width(3)
            draw_rounded_rect(cr, 1.5, 1.5, width - 3, height - 3, 11)
            cr.stroke()

    def _update_cam_surface(self, frame: np.ndarray, target_w: int, target_h: int) -> None:
        """Convert an OpenCV BGR frame into a Cairo ImageSurface.

        Always creates a new surface to avoid 'assertion ! _cairo_surface_has_snapshots'
        when updating a surface that GTK is currently rendering.
        """
        # Create a fresh surface for this frame
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, target_w, target_h)

        # Prepare the frame data
        resized = cv2.resize(frame, (target_w, target_h))
        # BGR → BGRA (Cairo ARGB32 is BGRA in memory on little-endian)
        bgra = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = 255

        # Write into Cairo surface buffer
        buf = surface.get_data()
        stride = surface.get_stride()

        if stride == target_w * 4:
            arr = np.ndarray(shape=(target_h, target_w, 4), dtype=np.uint8, buffer=buf)
            np.copyto(arr, bgra)
        else:
            for y in range(target_h):
                row_start = y * stride
                buf_arr = np.ndarray(shape=(stride,), dtype=np.uint8, buffer=buf, offset=row_start)
                buf_arr[: target_w * 4] = bgra[y].ravel()

        surface.mark_dirty()
        self._cam_surface = surface

    # ── Frame update (called from daemon tick) ───────────────────────────────
    def update(
        self,
        frame_result: FrameResult | None,
        classifications: list[ClassificationResult] | None = None,
        fps: float = 0.0,
        current_mode: str = "touchpad",
    ) -> None:
        """Update overlay state and queue a redraw.

        This is the main entry point called ~30× / second from the daemon loop.
        """
        if not self._enabled or self._window is None:
            return

        self._current_frame = frame_result
        self._classifications = classifications or []
        self._fps = fps
        self._current_mode = current_mode

        # ── Update camera surface if frame present ───────────────────────────
        if self._show_camera and frame_result and frame_result.frame is not None:
            # Use DrawingArea size if available, else configured size
            w = self._camera_area.get_width() if self._camera_area else self._cam_w
            h = self._camera_area.get_height() if self._camera_area else self._cam_h
            if w <= 1:
                w = self._cam_w
            if h <= 1:
                h = self._cam_h
            self._update_cam_surface(frame_result.frame, w, h)

        # ── Update labels ────────────────────────────────────────────────────
        if self._lbl_fps:
            self._lbl_fps.set_text(f"{fps:.0f} fps")

        if self._lbl_hands and frame_result:
            n = len(frame_result.hands)
            if n == 0:
                self._lbl_hands.set_text("No hands detected")
            else:
                parts = [f"{h.handedness} ({h.score:.0%})" for h in frame_result.hands]
                self._lbl_hands.set_text("  ".join(parts))

        # Gesture label
        if self._lbl_gesture:
            if self._classifications:
                top = self._classifications[0]
                self._lbl_gesture.set_text(f"✋ {top.gesture_name}  ({top.confidence:.0%})")
                self._lbl_gesture.remove_css_class("sigil-gesture-none")
                self._lbl_gesture.add_css_class("sigil-gesture")
            else:
                self._lbl_gesture.set_text("Waiting for gesture…")
                self._lbl_gesture.remove_css_class("sigil-gesture")
                self._lbl_gesture.add_css_class("sigil-gesture-none")

        # Action label (last fired)
        if self._lbl_action:
            if self._classifications:
                # Show raw_label or gesture action hint
                self._lbl_action.set_visible(True)
                labels = " │ ".join(c.raw_label or c.gesture_name for c in self._classifications)
                self._lbl_action.set_text(labels)
            else:
                self._lbl_action.set_visible(False)

        # Recording section
        if self._recording_box:
            self._recording_box.set_visible(self._is_recording)
        if self._is_recording and self._recording_status:
            if self._lbl_rec:
                self._lbl_rec.set_text("● REC")
            if self._lbl_rec_count:
                self._lbl_rec_count.set_text(self._recording_status)
            if self._progress_bar:
                self._progress_bar.set_fraction(self._recording_progress)
            if self._mode_badge:
                self._mode_badge.set_text("REC")
                self._mode_badge.remove_css_class("sigil-mode-badge")
                self._mode_badge.remove_css_class("sigil-mode-badge-touchpad")
                self._mode_badge.remove_css_class("sigil-mode-badge-keybind")
                self._mode_badge.add_css_class("sigil-mode-badge-recording")
            if self._camera_area:
                self._camera_area.remove_css_class("sigil-camera")
                self._camera_area.add_css_class("sigil-camera-recording")
        else:
            if self._mode_badge:
                if self._current_mode == "keybind":
                    self._mode_badge.set_text("KEYS")
                    self._mode_badge.remove_css_class("sigil-mode-badge")
                    self._mode_badge.remove_css_class("sigil-mode-badge-touchpad")
                    self._mode_badge.remove_css_class("sigil-mode-badge-recording")
                    self._mode_badge.add_css_class("sigil-mode-badge-keybind")
                else:
                    self._mode_badge.set_text("TOUCH")
                    self._mode_badge.remove_css_class("sigil-mode-badge")
                    self._mode_badge.remove_css_class("sigil-mode-badge-keybind")
                    self._mode_badge.remove_css_class("sigil-mode-badge-recording")
                    self._mode_badge.add_css_class("sigil-mode-badge-touchpad")
            if self._camera_area:
                self._camera_area.remove_css_class("sigil-camera-recording")
                self._camera_area.add_css_class("sigil-camera")

        # Update toasts
        self._update_toasts()

        # Queue camera area redraw
        if self._camera_area:
            self._camera_area.queue_draw()

    def _update_toasts(self) -> None:
        """Refresh toast notification widgets."""
        if not self._toast_box:
            return

        # Remove dead toasts
        self._toasts = [t for t in self._toasts if t.alive]

        # Clear existing children
        child = self._toast_box.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._toast_box.remove(child)
            child = next_child

        # Add live toasts
        for toast in self._toasts:
            lbl = Gtk.Label(label=f"⚡ {toast.text}")
            lbl.add_css_class("sigil-toast-text")
            lbl.set_opacity(toast.alpha)
            lbl.set_halign(Gtk.Align.START)

            box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
            box.add_css_class("sigil-toast")
            box.set_opacity(toast.alpha)
            box.append(lbl)
            self._toast_box.append(box)

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def close(self) -> None:
        """Destroy the overlay window."""
        if self._window:
            self._window.close()
            self._window = None
        self._cam_surface = None


# ══════════════════════════════════════════════════════════════════════════════
#  GTK Application wrapper
# ══════════════════════════════════════════════════════════════════════════════
class SigilGtkApp(Gtk.Application):
    """GTK4 Application that hosts the overlay and drives the frame loop.

    Parameters
    ----------
    overlay : WaylandOverlay
    frame_callback : callable
        Called every ~33 ms (30 fps). Should read tracker, classify, execute,
        and call ``overlay.update()``. Return ``True`` to keep running.
    shutdown_callback : callable
        Called when the application is closing.
    """

    def __init__(
        self,
        overlay: WaylandOverlay,
        frame_callback: Any,
        shutdown_callback: Any,
        target_fps: int = 30,
    ) -> None:
        super().__init__(application_id="dev.sigil.overlay")
        self._overlay = overlay
        self._frame_cb = frame_callback
        self._shutdown_cb = shutdown_callback
        self._target_fps = target_fps
        self._timer_id: int | None = None

    def do_activate(self) -> None:
        """Create window and start frame timer if callback provided."""
        self._overlay.create_window(self)

        # Start the frame processing timer if we have a callback (legacy/sync mode)
        if self._frame_cb:
            interval_ms = max(1, 1000 // self._target_fps)
            self._timer_id = GLib.timeout_add(interval_ms, self._on_tick)
            logger.info("GTK overlay active – sync frame timer at %d ms", interval_ms)
        else:
            logger.info("GTK overlay active – driving updates from background thread")

    def _on_tick(self) -> bool:
        """Process one frame. Returning False stops the timer."""
        if not self._frame_cb:
            return False
        try:
            keep_going = self._frame_cb()
            if not keep_going:
                self.quit()
                return False
            return True
        except Exception:
            logger.exception("Frame processing error")
            return True  # keep going despite errors

    def do_shutdown(self) -> None:
        """Clean up on exit."""
        if self._timer_id is not None:
            GLib.source_remove(self._timer_id)
            self._timer_id = None
        self._overlay.close()
        if self._shutdown_cb:
            self._shutdown_cb()
        Gtk.Application.do_shutdown(self)
