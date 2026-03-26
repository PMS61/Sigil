"""Main daemon / event loop (§5.5, §6).

Orchestrates: Tracker → Classifier → Mapper → Executor → Overlay.

Two execution paths:
  • **GTK4 path** (Wayland) – GLib.timeout_add drives the frame loop inside
    a Gtk.Application; no asyncio needed.
  • **Asyncio path** (fallback) – original ``asyncio.run`` loop with
    OpenCV overlay.
"""

from __future__ import annotations

import asyncio
import logging
import signal
import time

from sigil.classifier import GestureClassifier
from sigil.config import SigilConfig, load_config
from sigil.executor import ActionMapper, Executor
from sigil.overlay import Overlay
from sigil.recorder import Recorder, RecordMode
from sigil.tracker import Tracker

logger = logging.getLogger(__name__)


class Daemon:
    """Sigil background daemon (§5.5).

    Lifecycle:
      1. Load config
      2. Open tracker (camera + MediaPipe)
      3. Initialise classifier, mapper, executor, overlay
      4. Enter frame loop (GTK4 or asyncio)
      5. Clean shutdown on SIGINT / SIGTERM / ESC
    """

    def __init__(
        self,
        cfg: SigilConfig | None = None,
        *,
        force_backend: str | None = None,
    ) -> None:
        self._cfg = cfg or load_config()
        self._running = False
        self._force_backend = force_backend

        # Components
        self._tracker = Tracker(self._cfg.tracking)
        self._classifier = GestureClassifier(self._cfg.gestures, self._cfg.execution)
        self._mapper = ActionMapper(self._cfg)
        self._executor = Executor()
        self._overlay = Overlay(
            enabled=self._cfg.overlay,
            force_backend=force_backend,
            overlay_cfg=self._cfg.overlay_settings,
        )
        self._recorder = Recorder(self._cfg.recording)

        # Wire recorder status to overlay
        self._recorder.set_status_callback(self._overlay.set_recording_status)

        # Stats
        self._frames_processed: int = 0
        self._actions_fired: int = 0

    # ── Public API ───────────────────────────────────────────────────────────
    def run(self) -> None:
        """Start the daemon (blocking).

        Chooses GTK4 path when the overlay has a Wayland backend, otherwise
        falls back to the asyncio loop.
        """
        if self._overlay.is_wayland:
            self._run_gtk()
        else:
            asyncio.run(self._main_loop())

    # ── GTK4 path ────────────────────────────────────────────────────────────
    def _run_gtk(self) -> None:
        """Run using GLib main loop via the GTK4 application."""
        from sigil.ui.wayland_overlay import SigilGtkApp
        import threading

        wayland_ov = self._overlay.get_wayland_overlay()
        if wayland_ov is None:
            logger.error("Wayland overlay not available – falling back to asyncio path")
            asyncio.run(self._main_loop())
            return

        self._tracker.open()
        self._running = True

        logger.info("Sigil daemon starting (GTK4 thread path) …")
        logger.info("Overlay backend: %s", self._overlay.backend)
        logger.info("Overlay enabled: %s", self._overlay.enabled)

        # Start processing loop in a background thread
        self._worker_thread = threading.Thread(
            target=self._gtk_worker_loop, name="SigilWorker", daemon=True
        )
        self._worker_thread.start()

        app = SigilGtkApp(
            overlay=wayland_ov,
            frame_callback=None,  # Not used in threaded mode
            shutdown_callback=self._shutdown,
            target_fps=self._cfg.tracking.target_fps,
        )
        app.run()

    def _gtk_worker_loop(self) -> None:
        """Background thread for tracking and classification."""
        from gi.repository import GLib

        target_interval = 1.0 / self._cfg.tracking.target_fps
        prev_mode = self._classifier.current_mode

        try:
            while self._running:
                t0 = time.monotonic()

                frame = self._tracker.read()
                if frame is None:
                    time.sleep(0.01)
                    continue

                self._frames_processed += 1

                # Recording
                if self._recorder.is_active:
                    self._recorder.capture(frame)

                # Classify
                results = self._classifier.classify(frame)

                # Detect mode transition
                cur_mode = self._classifier.current_mode
                if cur_mode != prev_mode:
                    mode_label = "TOUCHPAD" if cur_mode == "touchpad" else "KEYBIND"
                    wayland_ov = self._overlay.get_wayland_overlay()
                    if wayland_ov:
                        GLib.idle_add(
                            wayland_ov.add_toast,
                            f"Mode → {mode_label}",
                            2.0,
                            "blue",
                        )
                    prev_mode = cur_mode

                # Execute (synchronous in worker thread)
                for result in results:
                    action = self._mapper.resolve(result)
                    if action:
                        logger.info(
                            "Gesture '%s' → %s (conf=%.2f)",
                            result.gesture_name,
                            action,
                            result.confidence,
                        )
                        success = self._executor.execute_sync(action, result)
                        if success:
                            self._actions_fired += 1
                            # Schedule toast update on UI thread
                            wayland_ov = self._overlay.get_wayland_overlay()
                            if wayland_ov:
                                GLib.idle_add(
                                    wayland_ov.add_toast, f"{result.gesture_name} → fired"
                                )

                # Update the Wayland overlay on the UI thread
                wayland_ov = self._overlay.get_wayland_overlay()
                if wayland_ov:
                    GLib.idle_add(
                        wayland_ov.update,
                        frame,
                        results,
                        self._tracker.fps,
                        self._classifier.current_mode,
                    )

                # Control FPS
                elapsed = time.monotonic() - t0
                wait = max(0.001, target_interval - elapsed)
                time.sleep(wait)

        except Exception:
            logger.exception("Fatal error in worker thread")
            self._running = False

    def _process_frame_sync(self) -> bool:
        """Process one frame synchronously (called from GLib timer).

        Returns True to keep running, False to stop.
        """
        if not self._running:
            return False

        frame = self._tracker.read()
        if frame is None:
            return True  # no frame yet, keep polling

        self._frames_processed += 1

        # Recording
        if self._recorder.is_active:
            self._recorder.capture(frame)

        # Classify
        prev_mode = self._classifier.current_mode
        results = self._classifier.classify(frame)
        cur_mode = self._classifier.current_mode

        # Detect mode transition
        if cur_mode != prev_mode:
            mode_label = "TOUCHPAD" if cur_mode == "touchpad" else "KEYBIND"
            wayland_ov = self._overlay.get_wayland_overlay()
            if wayland_ov:
                wayland_ov.add_toast(f"Mode → {mode_label}", 2.0, "blue")

        # Execute (synchronous)
        for result in results:
            action = self._mapper.resolve(result)
            if action:
                logger.info(
                    "Gesture '%s' → %s (conf=%.2f)",
                    result.gesture_name,
                    action,
                    result.confidence,
                )
                success = self._executor.execute_sync(action, result)
                if success:
                    self._actions_fired += 1
                    # Show toast on the Wayland overlay
                    wayland_ov = self._overlay.get_wayland_overlay()
                    if wayland_ov:
                        wayland_ov.add_toast(f"{result.gesture_name} → fired")

        # Update the Wayland overlay
        wayland_ov = self._overlay.get_wayland_overlay()
        if wayland_ov:
            wayland_ov.update(
                frame,
                results,
                fps=self._tracker.fps,
                current_mode=self._classifier.current_mode,
            )

        return True

    async def _main_loop(self) -> None:
        """Async main loop."""
        loop = asyncio.get_running_loop()

        # Register signal handlers for graceful shutdown
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, self._request_stop)

        logger.info("Sigil daemon starting …")
        self._tracker.open()
        self._running = True

        target_interval = 1.0 / self._cfg.tracking.target_fps
        low_fps_interval = 1.0 / self._cfg.tracking.low_fps_fallback

        try:
            while self._running:
                t0 = time.monotonic()

                # 1. Read frame
                frame = self._tracker.read()
                if frame is None:
                    await asyncio.sleep(0.01)
                    continue

                self._frames_processed += 1

                # 2. Recording mode capture
                if self._recorder.is_active:
                    self._recorder.capture(frame)

                # 3. Classify gestures
                prev_mode = self._classifier.current_mode
                results = self._classifier.classify(frame)

                # Log mode transitions
                if self._classifier.current_mode != prev_mode:
                    logger.info("Mode switched to '%s'", self._classifier.current_mode)

                # 4. Execute matched actions
                for result in results:
                    action = self._mapper.resolve(result)
                    if action:
                        logger.info(
                            "Gesture '%s' → %s (conf=%.2f)",
                            result.gesture_name,
                            action,
                            result.confidence,
                        )
                        asyncio.create_task(self._safe_execute(action, result))
                        self._actions_fired += 1

                # 5. Overlay
                if self._overlay.enabled:
                    alive = self._overlay.draw(frame, results, fps=self._tracker.fps)
                    if not alive:
                        self._request_stop()

                # 6. Frame pacing – graceful degradation (§6)
                elapsed = time.monotonic() - t0
                fps_low = (
                    self._tracker.fps > 0
                    and self._tracker.fps < self._cfg.tracking.low_fps_fallback
                )
                if fps_low:
                    wait = max(0, low_fps_interval - elapsed)
                else:
                    wait = max(0, target_interval - elapsed)

                if wait > 0:
                    await asyncio.sleep(wait)

        except Exception:
            logger.exception("Fatal error in main loop")
        finally:
            self._shutdown()

    async def _safe_execute(self, action: str, result: ClassificationResult | None = None) -> None:
        """Execute with error isolation."""
        try:
            await self._executor.execute(action, result)
        except Exception:
            logger.exception("Executor error for '%s'", action)

    def _request_stop(self) -> None:
        logger.info("Shutdown requested")
        self._running = False

    def _shutdown(self) -> None:
        """Clean up all resources."""
        logger.info(
            "Shutting down – processed %d frames, fired %d actions",
            self._frames_processed,
            self._actions_fired,
        )
        self._classifier.close()
        self._overlay.close()
        self._tracker.close()

    # ── Recording control (called by CLI / hotkey) ───────────────────────────
    def start_recording(
        self,
        class_name: str,
        mode: str = "instant",
        hand: str = "right",
    ) -> None:
        mode_map = {
            "instant": RecordMode.INSTANT,
            "gradual": RecordMode.GRADUAL,
            "sequential": RecordMode.SEQUENTIAL,
        }
        self._recorder.start(class_name, mode_map[mode], hand=hand)

    def stop_recording(self) -> None:
        self._recorder.stop()

    def discard_last_sample(self) -> None:
        self._recorder.discard_last()

    # ── Config hot-reload ────────────────────────────────────────────────────
    def reload_config(self) -> None:
        """Re-read config and update mapper without restarting."""
        self._cfg = load_config()
        self._mapper.reload(self._cfg)
        logger.info("Config reloaded")

    # ── Mode control ─────────────────────────────────────────────────────────
    def toggle_mode(self) -> str:
        """Manually toggle between touchpad and keybind modes.

        Returns the new mode name.
        """
        self._classifier._toggle_mode()
        return self._classifier.current_mode

    def get_mode(self) -> str:
        """Return the current operational mode."""
        return self._classifier.current_mode
