"""Sigil UI module – Wayland-native overlay with graceful fallback.

Backend selection priority:
  1. GTK4 + gtk4-layer-shell  → Layer-shell overlay (best Hyprland experience)
  2. GTK4 without layer-shell → Regular Wayland window (decent)
  3. OpenCV ``cv2.imshow``     → XWayland / X11 fallback (legacy)

System packages required for each tier:
  Tier 1: python-gobject, gtk4, gtk4-layer-shell
  Tier 2: python-gobject, gtk4
  Tier 3: opencv-python  (always available via pip)
"""

from __future__ import annotations

import ctypes
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Preload gtk4-layer-shell before libwayland-client ────────────────────────
# gtk4-layer-shell intercepts Wayland protocol calls and MUST be loaded before
# libwayland-client.so.  When Python loads gi → Gtk → libwayland first, the
# layer-shell init silently fails.  Preloading via ctypes (or LD_PRELOAD)
# ensures correct loading order.
# See: https://github.com/wmww/gtk4-layer-shell/blob/main/linking.md

_LAYER_SHELL_PRELOADED: bool = False

_LAYER_SHELL_SEARCH_PATHS = [
    "/usr/lib/libgtk4-layer-shell.so",
    "/usr/lib64/libgtk4-layer-shell.so",
    "/usr/local/lib/libgtk4-layer-shell.so",
    "/usr/lib/x86_64-linux-gnu/libgtk4-layer-shell.so",
    "/usr/lib/aarch64-linux-gnu/libgtk4-layer-shell.so",
]


def _preload_layer_shell() -> bool:
    """Preload libgtk4-layer-shell.so via ctypes to fix loading order."""
    # Check LD_PRELOAD first – user may have already set it
    ld_preload = os.environ.get("LD_PRELOAD", "")
    if "gtk4-layer-shell" in ld_preload:
        logger.debug("gtk4-layer-shell already in LD_PRELOAD")
        return True

    for path in _LAYER_SHELL_SEARCH_PATHS:
        if Path(path).exists():
            try:
                ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                logger.debug("Preloaded gtk4-layer-shell from %s", path)
                return True
            except OSError as exc:
                logger.debug("Failed to preload %s: %s", path, exc)

    # Try generic name (relies on ldconfig)
    try:
        ctypes.CDLL("libgtk4-layer-shell.so", mode=ctypes.RTLD_GLOBAL)
        logger.debug("Preloaded gtk4-layer-shell via generic name")
        return True
    except OSError:
        pass

    logger.debug("gtk4-layer-shell library not found for preloading")
    return False


_LAYER_SHELL_PRELOADED = _preload_layer_shell()

# ── Backend feature flags ────────────────────────────────────────────────────
HAS_GTK4: bool = False
HAS_LAYER_SHELL: bool = False
HAS_CAIRO: bool = False

try:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Gdk", "4.0")
    from gi.repository import Gdk, GLib, Gtk  # noqa: F401

    HAS_GTK4 = True
    logger.debug("GTK4 available")

    try:
        gi.require_version("Gtk4LayerShell", "1.0")
        from gi.repository import Gtk4LayerShell  # noqa: F401

        HAS_LAYER_SHELL = True
        logger.debug("gtk4-layer-shell available (preloaded=%s)", _LAYER_SHELL_PRELOADED)
    except (ValueError, ImportError):
        logger.debug("gtk4-layer-shell not found – falling back to regular GTK4 window")

    try:
        import cairo  # noqa: F401

        HAS_CAIRO = True
    except ImportError:
        logger.debug("pycairo not found – camera preview will be disabled")

except (ValueError, ImportError):
    logger.debug("GTK4 not available – will use OpenCV fallback overlay")


def get_backend() -> str:
    """Return the name of the best available UI backend."""
    if HAS_GTK4 and HAS_LAYER_SHELL:
        return "gtk4-layer-shell"
    elif HAS_GTK4:
        return "gtk4"
    return "opencv"


__all__ = [
    "HAS_GTK4",
    "HAS_LAYER_SHELL",
    "HAS_CAIRO",
    "get_backend",
]
