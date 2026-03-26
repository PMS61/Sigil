"""Cairo drawing utilities for hand landmarks and UI elements.

Renders MediaPipe hand landmarks, connections, and status indicators
onto Cairo contexts for the Wayland overlay.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cairo

    from sigil.tracker import HandResult

# ── Catppuccin Mocha palette (RGBA 0–1) ─────────────────────────────────────
COLORS = {
    "base": (0.118, 0.118, 0.180, 0.88),
    "surface0": (0.192, 0.196, 0.267, 1.0),
    "surface1": (0.271, 0.278, 0.353, 1.0),
    "overlay0": (0.427, 0.443, 0.537, 1.0),
    "text": (0.804, 0.839, 0.957, 1.0),
    "subtext0": (0.651, 0.678, 0.784, 1.0),
    "blue": (0.537, 0.706, 0.980, 1.0),
    "green": (0.651, 0.890, 0.631, 1.0),
    "red": (0.953, 0.545, 0.659, 1.0),
    "peach": (0.980, 0.702, 0.529, 1.0),
    "mauve": (0.796, 0.651, 0.969, 1.0),
    "teal": (0.596, 0.878, 0.816, 1.0),
    "yellow": (0.976, 0.886, 0.686, 1.0),
    "left_hand": (0.980, 0.702, 0.529, 1.0),   # peach
    "right_hand": (0.537, 0.706, 0.980, 1.0),  # blue
}

# ── Landmark connections (MediaPipe hand topology) ───────────────────────────
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (5, 9), (9, 10), (10, 11), (11, 12),     # middle
    (9, 13), (13, 14), (14, 15), (15, 16),   # ring
    (13, 17), (17, 18), (18, 19), (19, 20),  # pinky
    (0, 17),
]

FINGER_TIPS = [4, 8, 12, 16, 20]


def draw_hand(
    cr: cairo.Context,
    hand: HandResult,
    width: int,
    height: int,
    *,
    landmark_radius: float = 3.0,
    connection_width: float = 2.0,
    glow: bool = True,
) -> None:
    """Draw a single hand's landmarks and connections onto a Cairo context.

    Parameters
    ----------
    cr : cairo.Context
    hand : HandResult with .landmarks (21, 3) normalised and .handedness
    width, height : pixel dimensions of the drawing area
    """
    color = COLORS.get(f"{hand.handedness.lower()}_hand", COLORS["blue"])
    lm = hand.landmarks  # (21, 3) normalised

    # Convert normalised → pixel
    pts = np.zeros((21, 2), dtype=np.float64)
    for i in range(21):
        pts[i] = (lm[i][0] * width, lm[i][1] * height)

    # ── Connections ──────────────────────────────────────────────────────────
    cr.set_line_width(connection_width)
    cr.set_source_rgba(color[0], color[1], color[2], 0.6)
    for a, b in HAND_CONNECTIONS:
        cr.move_to(pts[a][0], pts[a][1])
        cr.line_to(pts[b][0], pts[b][1])
    cr.stroke()

    # ── Landmarks ────────────────────────────────────────────────────────────
    for i in range(21):
        x, y = pts[i]

        # Glow effect for fingertips
        if glow and i in FINGER_TIPS:
            cr.set_source_rgba(color[0], color[1], color[2], 0.15)
            cr.arc(x, y, landmark_radius * 3, 0, 2 * 3.14159)
            cr.fill()

        # Solid landmark dot
        cr.set_source_rgba(color[0], color[1], color[2], 0.95)
        cr.arc(x, y, landmark_radius, 0, 2 * 3.14159)
        cr.fill()

    # ── Handedness label near wrist ──────────────────────────────────────────
    wrist = pts[0]
    cr.set_source_rgba(*color[:3], 0.8)
    cr.select_font_face("Sans", 0, 1)  # type: ignore[arg-type]
    cr.set_font_size(11)
    cr.move_to(wrist[0] - 12, wrist[1] - 12)
    cr.show_text(hand.handedness[0])  # "L" or "R"


def draw_rounded_rect(
    cr: cairo.Context,
    x: float,
    y: float,
    w: float,
    h: float,
    radius: float = 12,
) -> None:
    """Add a rounded rectangle path to the Cairo context."""
    import math

    cr.new_sub_path()
    cr.arc(x + w - radius, y + radius, radius, -math.pi / 2, 0)
    cr.arc(x + w - radius, y + h - radius, radius, 0, math.pi / 2)
    cr.arc(x + radius, y + h - radius, radius, math.pi / 2, math.pi)
    cr.arc(x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
    cr.close_path()


def draw_progress_bar(
    cr: cairo.Context,
    x: float,
    y: float,
    w: float,
    h: float,
    progress: float,
    *,
    bg_color: tuple[float, ...] = COLORS["surface0"],
    fg_color: tuple[float, ...] = COLORS["blue"],
    radius: float = 4,
) -> None:
    """Draw a rounded progress bar."""
    # Background
    draw_rounded_rect(cr, x, y, w, h, radius)
    cr.set_source_rgba(*bg_color)
    cr.fill()

    # Filled portion
    fill_w = max(0, min(w, w * progress))
    if fill_w > radius * 2:
        draw_rounded_rect(cr, x, y, fill_w, h, radius)
        cr.set_source_rgba(*fg_color)
        cr.fill()


def draw_status_pill(
    cr: cairo.Context,
    x: float,
    y: float,
    text: str,
    *,
    bg_color: tuple[float, ...] = COLORS["surface1"],
    text_color: tuple[float, ...] = COLORS["text"],
    font_size: float = 11,
    padding_x: float = 8,
    padding_y: float = 4,
) -> float:
    """Draw a rounded pill / badge and return its width."""
    cr.select_font_face("Sans", 0, 0)  # type: ignore[arg-type]
    cr.set_font_size(font_size)
    extents = cr.text_extents(text)
    w = extents.width + padding_x * 2
    h = extents.height + padding_y * 2

    draw_rounded_rect(cr, x, y, w, h, h / 2)
    cr.set_source_rgba(*bg_color)
    cr.fill()

    cr.set_source_rgba(*text_color)
    cr.move_to(x + padding_x, y + padding_y + extents.height)
    cr.show_text(text)

    return w
