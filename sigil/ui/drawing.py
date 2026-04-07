"""Cairo drawing utilities for hand landmarks and UI elements.

Renders MediaPipe hand landmarks, connections, and status indicators
onto Cairo contexts for the Wayland overlay.

Depth-aware rendering: uses z-coordinate to modulate opacity/thickness.
Joint hierarchy: wrist > MCP > PIP/DIP > fingertips (with glow).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import cairo

    from sigil.tracker import HandResult

# ── Catppuccin Mocha palette (RGBA 0–1) ─────────────────────────────────────
COLORS = {
    "base": (0.118, 0.118, 0.180, 0.88),
    "crust": (0.067, 0.067, 0.106, 1.0),
    "mantle": (0.094, 0.094, 0.145, 1.0),
    "surface0": (0.192, 0.196, 0.267, 1.0),
    "surface1": (0.271, 0.278, 0.353, 1.0),
    "surface2": (0.345, 0.353, 0.431, 1.0),
    "overlay0": (0.427, 0.443, 0.537, 1.0),
    "overlay1": (0.502, 0.518, 0.612, 1.0),
    "text": (0.804, 0.839, 0.957, 1.0),
    "subtext0": (0.651, 0.678, 0.784, 1.0),
    "subtext1": (0.729, 0.757, 0.871, 1.0),
    "blue": (0.537, 0.706, 0.980, 1.0),
    "green": (0.651, 0.890, 0.631, 1.0),
    "red": (0.953, 0.545, 0.659, 1.0),
    "peach": (0.980, 0.702, 0.529, 1.0),
    "mauve": (0.796, 0.651, 0.969, 1.0),
    "teal": (0.596, 0.878, 0.816, 1.0),
    "yellow": (0.976, 0.886, 0.686, 1.0),
    "lavender": (0.702, 0.745, 0.996, 1.0),
    "sapphire": (0.455, 0.780, 0.925, 1.0),
    "sky": (0.537, 0.820, 0.922, 1.0),
    "flamingo": (0.949, 0.604, 0.620, 1.0),
    "rosewater": (0.961, 0.757, 0.757, 1.0),
    "left_hand": (0.980, 0.702, 0.529, 1.0),   # peach
    "right_hand": (0.537, 0.706, 0.980, 1.0),   # blue
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
MCP_JOINTS = [1, 5, 9, 13, 17]   # metacarpophalangeal joints
PIP_JOINTS = [2, 6, 10, 14, 18]  # proximal interphalangeal joints
PALM_POLYGON = [0, 5, 9, 13, 17]  # palm region vertices

# ── Gesture emoji mapping ────────────────────────────────────────────────────
GESTURE_EMOJIS: dict[str, str] = {
    "Closed_Fist": "✊",
    "Open_Palm": "🖐️",
    "Pointing_Up": "☝️",
    "Thumb_Up": "👍",
    "Thumb_Down": "👎",
    "Victory": "✌️",
    "ILoveYou": "🤟",
    # Derived / custom
    "pinch": "🤏",
    "two_hand_distance": "↔️",
    "pointer_position": "👆",
    "finger_count": "🖐️",
    "finger_deltas": "↗️",
    "finger_curl_mean": "✊",
    "thumb_index_distance": "🤏",
}


def _depth_factor(z: float) -> float:
    """Map z-coordinate to a 0–1 factor (closer = 1.0, farther = 0.5)."""
    # MediaPipe z values are roughly -0.1 to 0.1 relative to wrist
    # Positive z = away from camera, negative = toward camera
    return max(0.5, min(1.0, 0.75 - z * 3.0))


def draw_hand(
    cr: "cairo.Context",
    hand: "HandResult",
    width: int,
    height: int,
    *,
    glow: bool = True,
    confidence: float = 1.0,
    fast_mode: bool = False,
) -> None:
    """Draw a single hand's landmarks and connections onto a Cairo context.

    Features:
    - Depth-aware line thickness and opacity (z-coordinate)
    - Joint hierarchy (wrist > MCP > PIP > tips)
    - Multi-layer fingertip glow
    - Semi-transparent palm polygon fill
    - Confidence arc around wrist

    When ``fast_mode`` is enabled, uses a simplified draw path (flat lines + dots)
    to keep rendering cost low on software compositors.
    """
    color = COLORS.get(f"{hand.handedness.lower()}_hand", COLORS["blue"])
    lm = hand.landmarks  # (21, 3) normalised

    # Convert normalised → pixel
    pts = np.zeros((21, 2), dtype=np.float64)
    z_vals = np.zeros(21, dtype=np.float64)
    for i in range(21):
        pts[i] = (lm[i][0] * width, lm[i][1] * height)
        z_vals[i] = lm[i][2] if lm.shape[1] > 2 else 0.0

    # Enable anti-aliasing
    cr.set_antialias(1)  # CAIRO_ANTIALIAS_GOOD

    if fast_mode:
        # Single-stroke connection pass.
        cr.set_line_width(1.6)
        cr.set_source_rgba(color[0], color[1], color[2], 0.55)
        for a, b in HAND_CONNECTIONS:
            cr.move_to(pts[a][0], pts[a][1])
            cr.line_to(pts[b][0], pts[b][1])
        cr.stroke()

        # Lightweight landmark pass.
        for i in range(21):
            x, y = pts[i]
            radius = 2.7 if i in FINGER_TIPS else 2.0
            alpha = 0.9 if i in FINGER_TIPS else 0.8
            cr.set_source_rgba(color[0], color[1], color[2], alpha)
            cr.arc(x, y, radius, 0, 2 * math.pi)
            cr.fill()
        return

    # ── Palm polygon fill (semi-transparent) ─────────────────────────────────
    cr.set_source_rgba(color[0], color[1], color[2], 0.08)
    cr.move_to(pts[PALM_POLYGON[0]][0], pts[PALM_POLYGON[0]][1])
    for idx in PALM_POLYGON[1:]:
        cr.line_to(pts[idx][0], pts[idx][1])
    cr.close_path()
    cr.fill()

    # ── Connections (depth-aware thickness and opacity) ───────────────────────
    for a, b in HAND_CONNECTIONS:
        avg_z = (z_vals[a] + z_vals[b]) / 2.0
        depth = _depth_factor(avg_z)
        line_w = 1.5 + depth * 1.5  # 1.5 – 3.0 px
        alpha = 0.35 + depth * 0.35  # 0.35 – 0.70

        cr.set_line_width(line_w)
        cr.set_source_rgba(color[0], color[1], color[2], alpha)
        cr.move_to(pts[a][0], pts[a][1])
        cr.line_to(pts[b][0], pts[b][1])
        cr.stroke()

    # ── Landmarks (hierarchical sizing) ──────────────────────────────────────
    for i in range(21):
        x, y = pts[i]
        depth = _depth_factor(z_vals[i])

        if i == 0:
            # Wrist: largest, anchor point
            radius = 4.0
            alpha = 0.95
        elif i in MCP_JOINTS:
            # MCP joints: medium
            radius = 3.0
            alpha = 0.85
        elif i in FINGER_TIPS:
            # Fingertips: medium with glow
            radius = 2.8
            alpha = 0.95
        elif i in PIP_JOINTS:
            # PIP joints: small
            radius = 2.2
            alpha = 0.75
        else:
            # DIP and other joints: smallest
            radius = 2.0
            alpha = 0.70

        # Scale radius by depth
        radius *= (0.7 + depth * 0.3)

        # Fingertip multi-layer glow
        if glow and i in FINGER_TIPS:
            # Outer glow (very diffuse)
            cr.set_source_rgba(color[0], color[1], color[2], 0.06 * depth)
            cr.arc(x, y, radius * 4.5, 0, 2 * math.pi)
            cr.fill()
            # Mid glow
            cr.set_source_rgba(color[0], color[1], color[2], 0.12 * depth)
            cr.arc(x, y, radius * 2.8, 0, 2 * math.pi)
            cr.fill()
            # Inner glow
            cr.set_source_rgba(color[0], color[1], color[2], 0.22 * depth)
            cr.arc(x, y, radius * 1.6, 0, 2 * math.pi)
            cr.fill()

        # Solid landmark dot
        cr.set_source_rgba(color[0], color[1], color[2], alpha * depth)
        cr.arc(x, y, radius, 0, 2 * math.pi)
        cr.fill()

    # ── Confidence arc around wrist ──────────────────────────────────────────
    if confidence > 0:
        wrist = pts[0]
        arc_radius = 8.0
        # Draw partial arc proportional to confidence (0° = top, clockwise)
        start_angle = -math.pi / 2
        end_angle = start_angle + (2 * math.pi * confidence)

        # Background track
        cr.set_source_rgba(color[0], color[1], color[2], 0.12)
        cr.set_line_width(1.5)
        cr.arc(wrist[0], wrist[1], arc_radius, 0, 2 * math.pi)
        cr.stroke()

        # Confidence fill
        if confidence > 0.5:
            arc_alpha = 0.6
        else:
            arc_alpha = 0.4
        cr.set_source_rgba(color[0], color[1], color[2], arc_alpha)
        cr.set_line_width(2.0)
        cr.arc(wrist[0], wrist[1], arc_radius, start_angle, end_angle)
        cr.stroke()

    # ── Handedness label near wrist ──────────────────────────────────────────
    wrist = pts[0]
    cr.set_source_rgba(*color[:3], 0.7)
    cr.select_font_face("Sans", 0, 1)  # type: ignore[arg-type]
    cr.set_font_size(10)
    cr.move_to(wrist[0] - 8, wrist[1] - 18)
    cr.show_text(hand.handedness[0])  # "L" or "R"


def draw_rounded_rect(
    cr: "cairo.Context",
    x: float,
    y: float,
    w: float,
    h: float,
    radius: float = 12,
) -> None:
    """Add a rounded rectangle path to the Cairo context."""
    cr.new_sub_path()
    cr.arc(x + w - radius, y + radius, radius, -math.pi / 2, 0)
    cr.arc(x + w - radius, y + h - radius, radius, 0, math.pi / 2)
    cr.arc(x + radius, y + h - radius, radius, math.pi / 2, math.pi)
    cr.arc(x + radius, y + radius, radius, math.pi, 3 * math.pi / 2)
    cr.close_path()


def draw_progress_bar(
    cr: "cairo.Context",
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
    cr: "cairo.Context",
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


def draw_tracking_quality_bar(
    cr: "cairo.Context",
    x: float,
    y: float,
    width: float,
    num_hands: int,
    max_hands: int = 2,
) -> None:
    """Draw a thin tracking quality indicator bar at the bottom of the camera preview.

    Shows hand detection status: 0 = no tracking, 1 = single hand, 2 = both hands.
    """
    bar_h = 3.0
    radius = 1.5

    # Background track
    draw_rounded_rect(cr, x, y, width, bar_h, radius)
    cr.set_source_rgba(*COLORS["surface0"][:3], 0.5)
    cr.fill()

    if num_hands > 0:
        fill_ratio = num_hands / max(max_hands, 1)
        fill_w = max(bar_h, width * fill_ratio)

        # Color: 1 hand = blue, 2 hands = green
        if num_hands >= max_hands:
            bar_color = COLORS["green"]
        else:
            bar_color = COLORS["blue"]

        draw_rounded_rect(cr, x, y, fill_w, bar_h, radius)
        cr.set_source_rgba(*bar_color[:3], 0.7)
        cr.fill()
