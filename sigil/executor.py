"""Gesture → Hyprland Mapping & Execution (§5.5).

Maps classified gestures to ``hyprctl dispatch`` commands via config.
Executes via hyprctl UNIX socket for lowest latency, with CLI fallback.

Provides both async (``execute``) and sync (``execute_sync``) methods so
the executor works in both asyncio and GLib main loop contexts.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
import json
import socket
import subprocess
from pathlib import Path
from typing import Any

from sigil.classifier import ClassificationResult
from sigil.config import GestureMapping, SigilConfig

logger = logging.getLogger(__name__)


# ── Hyprland socket discovery ────────────────────────────────────────────────
def _find_hyprctl_socket() -> Path | None:
    """Locate the hyprctl UNIX socket for the current Hyprland instance."""
    his = os.environ.get("HYPRLAND_INSTANCE_SIGNATURE")
    if his:
        # Modern Hyprland uses XDG_RUNTIME_DIR/hypr/<signature>/.socket.sock
        runtime = os.environ.get("XDG_RUNTIME_DIR", f"/run/user/{os.getuid()}")
        candidates = [
            Path(runtime) / "hypr" / his / ".socket.sock",
            Path(runtime) / "hypr" / his / ".socket2.sock",
            Path(f"/tmp/hypr/{his}/.socket.sock"),
        ]
        for p in candidates:
            if p.exists():
                return p
    return None


def _has_hyprctl() -> bool:
    return shutil.which("hyprctl") is not None


# ── Action resolver ──────────────────────────────────────────────────────────
class ActionMapper:
    """Resolves gesture names to configured hyprctl dispatch commands (§5.5).

    Priority: first matching gesture in config fires.
    """

    def __init__(self, cfg: SigilConfig) -> None:
        self._mappings: dict[str, GestureMapping] = {}
        for g in cfg.gestures:
            if g.enabled and g.name not in self._mappings:
                self._mappings[g.name] = g
        logger.info("ActionMapper loaded %d gesture mappings", len(self._mappings))

    def resolve(self, result: ClassificationResult) -> str | None:
        """Return the hyprctl action string for a classification result, or None."""
        mapping = self._mappings.get(result.gesture_name)
        if mapping is None:
            return None
        
        return mapping.action or None

    def reload(self, cfg: SigilConfig) -> None:
        """Hot-reload mappings from updated config."""
        self._mappings.clear()
        for g in cfg.gestures:
            if g.enabled and g.name not in self._mappings:
                self._mappings[g.name] = g
        logger.info("ActionMapper reloaded – %d mappings", len(self._mappings))


# ── Executor ─────────────────────────────────────────────────────────────────
class Executor:
    """Executes hyprctl dispatch commands (§5.5).

    Prefers UNIX socket for <5 ms latency; falls back to CLI subprocess.
    """

    def __init__(self) -> None:
        self._socket_path = _find_hyprctl_socket()
        self._use_socket = self._socket_path is not None
        self._use_cli = _has_hyprctl()
        self._screen_res: tuple[int, int] = (1920, 1080)  # default fallback
        
        # Cursor smoothing state
        self._last_norm_pos: tuple[float, float] | None = None
        self._smoothing_alpha = 0.3  # lower = smoother but more lag
        self._sensitivity = 2.0      # higher = less hand movement needed
        
        self._refresh_screen_res()

        if self._use_socket:
            logger.info("Executor: using Hyprland socket %s", self._socket_path)
        elif self._use_cli:
            logger.info("Executor: using hyprctl CLI fallback")
        else:
            logger.warning(
                "Executor: hyprctl not found – actions will be logged but not executed"
            )

    def _refresh_screen_res(self) -> None:
        """Fetch primary monitor resolution from hyprctl."""
        if not self._use_cli:
            return
        
        try:
            res = subprocess.run(
                ["hyprctl", "monitors", "-j"],
                capture_output=True, text=True, check=True
            )
            monitors = json.loads(res.stdout)
            if monitors:
                # Find primary or first monitor
                primary = next((m for m in monitors if m.get("focused")), monitors[0])
                self._screen_res = (primary["width"], primary["height"])
                logger.info("Executor: screen resolution detected as %dx%d", *self._screen_res)
        except Exception as exc:
            logger.warning("Failed to detect screen resolution (%s), using %dx%d", exc, *self._screen_res)

    def _interpolate_action(self, action: str, result: ClassificationResult | None) -> str:
        """Replace placeholders in action string with actual values."""
        if not result:
            return action
        
        # Base value interpolation
        action = action.replace("{val}", f"{result.value:.4f}")
        
        # Position interpolation (scaled to screen)
        if result.position:
            norm_x, norm_y = result.position
            
            # Apply sensitivity (map center of camera to full screen)
            norm_x = (norm_x - 0.5) * self._sensitivity + 0.5
            norm_y = (norm_y - 0.5) * self._sensitivity + 0.5
            
            # Clamp to 0..1 range
            norm_x = max(0.0, min(1.0, norm_x))
            norm_y = max(0.0, min(1.0, norm_y))

            # Apply exponential smoothing
            if self._last_norm_pos is None:
                self._last_norm_pos = (norm_x, norm_y)
            else:
                lx, ly = self._last_norm_pos
                norm_x = lx + self._smoothing_alpha * (norm_x - lx)
                norm_y = ly + self._smoothing_alpha * (norm_y - ly)
                self._last_norm_pos = (norm_x, norm_y)

            # MediaPipe X is 0 (left) to 1 (right). 
            # Hyprland movecursor uses screen coordinates.
            screen_x = int(norm_x * self._screen_res[0])
            screen_y = int(norm_y * self._screen_res[1])
            
            action = action.replace("{x}", str(screen_x))
            action = action.replace("{y}", str(screen_y))
            action = action.replace("{nx}", f"{norm_x:.4f}")
            action = action.replace("{ny}", f"{norm_y:.4f}")
        else:
            # Reset smoothing if no position detected this frame
            self._last_norm_pos = None
            
        # Delta interpolation
        if result.deltas:
            dx, dy = result.deltas
            # Scale deltas to screen resolution for scrolling/movement
            sdx = int(dx * self._screen_res[0])
            sdy = int(dy * self._screen_res[1])
            action = action.replace("{dx}", str(sdx))
            action = action.replace("{dy}", str(sdy))
            action = action.replace("{ndx}", f"{dx:.4f}")
            action = action.replace("{ndy}", f"{dy:.4f}")

        # Value/Count interpolation
        action = action.replace("{count}", str(int(result.value)))
        
        return action

    async def execute(self, action: str, result: ClassificationResult | None = None) -> bool:
        """Execute a hyprctl action string asynchronously.

        Returns True on success, False on failure.
        """
        if not action:
            return False

        action = self._interpolate_action(action, result)
        logger.info("Executing: %s", action)

        # Try socket first
        if self._use_socket and self._socket_path:
            try:
                return await self._execute_socket(action)
            except OSError as exc:
                logger.warning(
                    "Socket execution failed (%s) – disabling socket, using CLI fallback",
                    exc,
                )
                self._use_socket = False
                self._socket_path = None
            except Exception as exc:
                logger.warning("Socket execution failed (%s), trying CLI", exc)

        # CLI fallback
        if self._use_cli:
            return await self._execute_cli(action)

        logger.warning("No executor available – skipping: %s", action)
        return False

    async def _execute_socket(self, action: str) -> bool:
        """Send command via Hyprland UNIX socket."""
        assert self._socket_path is not None

        # hyprctl dispatch expects: /dispatch <dispatcher> <args>
        # But the action string from config is already "hyprctl dispatch ..."
        # We need to strip "hyprctl " prefix for socket commands.
        cmd = action.strip()
        if cmd.startswith("hyprctl "):
            cmd = cmd[len("hyprctl "):]

        reader, writer = await asyncio.open_unix_connection(str(self._socket_path))
        try:
            # Hyprland socket protocol: send "/<command>\n"
            writer.write(f"/{cmd}".encode())
            await writer.drain()
            response = await asyncio.wait_for(reader.read(4096), timeout=2.0)
            resp_str = response.decode().strip()
            logger.debug("Socket response: %s", resp_str)
            return resp_str == "ok" or "ok" in resp_str.lower()
        finally:
            writer.close()
            await writer.wait_closed()

    async def _execute_cli(self, action: str) -> bool:
        """Execute via subprocess (higher latency, ~10-20 ms)."""
        # Action is a full shell command like "hyprctl dispatch killactive"
        proc = await asyncio.create_subprocess_shell(
            action,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error(
                "CLI execution failed (rc=%d): %s\nstderr: %s",
                proc.returncode,
                action,
                stderr.decode(),
            )
            return False
        logger.debug("CLI output: %s", stdout.decode().strip())
        return True

    # ── Synchronous API (for GLib main loop) ─────────────────────────────────
    def execute_sync(self, action: str, result: ClassificationResult | None = None) -> bool:
        """Execute a hyprctl action string synchronously.

        Used by the GTK4/GLib path where async is not available.
        """
        if not action:
            return False

        action = self._interpolate_action(action, result)
        logger.info("Executing (sync): %s", action)

        if self._use_socket and self._socket_path:
            try:
                return self._execute_socket_sync(action)
            except OSError as exc:
                logger.warning("Socket failed (%s) – disabling, using CLI", exc)
                self._use_socket = False
                self._socket_path = None

        if self._use_cli:
            return self._execute_cli_sync(action)

        logger.warning("No executor available – skipping: %s", action)
        return False

    def _execute_socket_sync(self, action: str) -> bool:
        """Send command via synchronous UNIX socket."""
        assert self._socket_path is not None

        cmd = action.strip()
        if cmd.startswith("hyprctl "):
            cmd = cmd[len("hyprctl "):]

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(2.0)
        try:
            sock.connect(str(self._socket_path))
            sock.sendall(f"/{cmd}".encode())
            resp = sock.recv(4096).decode().strip()
            logger.debug("Socket response (sync): %s", resp)
            return "ok" in resp.lower()
        finally:
            sock.close()

    def _execute_cli_sync(self, action: str) -> bool:
        """Execute via subprocess.run (blocking)."""
        result = subprocess.run(
            action,
            shell=True,
            capture_output=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.error(
                "CLI execution failed (rc=%d): %s\nstderr: %s",
                result.returncode,
                action,
                result.stderr.decode(),
            )
            return False
        logger.debug("CLI output (sync): %s", result.stdout.decode().strip())
        return True
