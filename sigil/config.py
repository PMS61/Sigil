"""Configuration loading and management for Sigil.

Handles YAML config at ~/.config/sigil/config.yaml with gesture-to-action mappings,
tracking parameters, and recording settings.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
CONFIG_DIR = Path.home() / ".config" / "sigil"
CONFIG_FILE = CONFIG_DIR / "config.yaml"
DATA_DIR = Path.home() / ".local" / "share" / "sigil"
MODELS_DIR = DATA_DIR / "models"
RECORDINGS_DIR = DATA_DIR / "recordings"
LOG_DIR = DATA_DIR / "logs"
DEFAULT_CONFIG = Path(__file__).parent / "default_config.yaml"


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class TrackingConfig:
    """MediaPipe Hand Landmarker parameters (§5.1)."""

    num_hands: int = 2
    min_detection_confidence: float = 0.7
    min_tracking_confidence: float = 0.6
    model_asset_path: str = "hand_landmarker.task"
    camera_id: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: int = 30
    low_fps_fallback: int = 15  # §6 graceful degradation
    zoom: float = 1.0  # digital zoom level (1.0 = no zoom, >1.0 = cropped)


@dataclass
class RecordingConfig:
    """Recording-mode parameters (§5.3)."""

    hotkey: str = "Super+Alt+G"
    samples_per_class: int = 100
    min_samples: int = 50
    max_samples: int = 200
    output_dir: str = str(RECORDINGS_DIR)


@dataclass
class GestureMapping:
    """Single gesture→action mapping (§5.5)."""

    name: str
    type: str  # "instant" | "gradual" | "sequential"
    hand: str  # "left" | "right" | "both"
    condition: dict[str, Any] = field(default_factory=dict)
    action: str = ""  # hyprctl dispatch string
    cooldown_ms: int = 300  # debounce
    continuous: bool = False  # If True, fires every frame without suppression
    enabled: bool = True
    active_modes: list[str] = field(default_factory=lambda: ["both"])  # "touchpad"|"keybind"|"both"
    mode_toggle: bool = False  # If True, toggles between touchpad/keybind modes
    # Optional per-gesture execution policy overrides (primarily for touchpad reliability).
    # If unset, global ExecutionConfig applies.
    confirm_frames: int | None = None
    blanking_ms: int | None = None
    confidence_threshold: float | None = None


@dataclass
class ExecutionConfig:
    """Controls how instant gestures are confirmed and cursor movement (§5.5)."""

    confirm_frames: int = 3  # consecutive frames required to confirm an instant gesture
    blanking_ms: int = 1500  # global suppression after an instant action fires
    confidence_threshold: float = 0.75  # minimum confidence to count a frame towards confirmation


@dataclass
class OverlayConfig:
    """UI overlay appearance settings (§5.6)."""

    position: str = "bottom-right"  # bottom-right | bottom-left | top-right | top-left
    width: int = 320  # camera preview width
    height: int = 180  # camera preview height
    opacity: float = 1.0  # base window opacity (0.0–1.0)
    camera_preview: bool = True  # show camera feed in overlay
    margin: int = 16  # pixel margin from screen edge


@dataclass
class SigilConfig:
    """Top-level Sigil configuration."""

    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    recording: RecordingConfig = field(default_factory=RecordingConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    overlay_settings: OverlayConfig = field(default_factory=OverlayConfig)
    gestures: list[GestureMapping] = field(default_factory=list)
    log_level: str = "info"
    overlay: bool = True
    daemon: bool = True


# ── Loader ─────────────────────────────────────────────────────────────────────
def _ensure_dirs() -> None:
    """Create required directories if missing."""
    for d in (CONFIG_DIR, DATA_DIR, MODELS_DIR, RECORDINGS_DIR, LOG_DIR):
        d.mkdir(parents=True, exist_ok=True)


def _install_default_config() -> None:
    """Copy bundled default config to user config dir on first run."""
    if not CONFIG_FILE.exists() and DEFAULT_CONFIG.exists():
        shutil.copy2(DEFAULT_CONFIG, CONFIG_FILE)
        logger.info("Installed default config → %s", CONFIG_FILE)


def _parse_tracking(raw: dict[str, Any]) -> TrackingConfig:
    fields = TrackingConfig.__dataclass_fields__
    return TrackingConfig(**{k: v for k, v in raw.items() if k in fields})


def _parse_recording(raw: dict[str, Any]) -> RecordingConfig:
    return RecordingConfig(
        **{k: v for k, v in raw.items() if k in RecordingConfig.__dataclass_fields__}
    )


def _parse_execution(raw: dict[str, Any]) -> ExecutionConfig:
    fields = ExecutionConfig.__dataclass_fields__
    return ExecutionConfig(**{k: v for k, v in raw.items() if k in fields})


def _parse_overlay(raw: dict[str, Any]) -> OverlayConfig:
    fields = OverlayConfig.__dataclass_fields__
    return OverlayConfig(**{k: v for k, v in raw.items() if k in fields})


def _parse_gestures(raw: list[dict[str, Any]]) -> list[GestureMapping]:
    mappings: list[GestureMapping] = []
    for entry in raw:
        try:
            raw_modes = entry.get("active_modes", ["both"])
            # Normalise: accept string or list
            if isinstance(raw_modes, str):
                raw_modes = [raw_modes]
            mappings.append(
                GestureMapping(
                    name=entry["name"],
                    type=entry["type"],
                    hand=entry.get("hand", "right"),
                    condition=entry.get("condition", {}),
                    action=entry.get("action", ""),
                    cooldown_ms=entry.get("cooldown_ms", 300),
                    continuous=entry.get("continuous", False),
                    enabled=entry.get("enabled", True),
                    active_modes=raw_modes,
                    mode_toggle=entry.get("mode_toggle", False),
                    confirm_frames=entry.get("confirm_frames"),
                    blanking_ms=entry.get("blanking_ms"),
                    confidence_threshold=entry.get("confidence_threshold"),
                )
            )
        except KeyError as exc:
            logger.warning("Skipping malformed gesture entry: %s (%s)", entry, exc)
    return mappings


def load_config(path: Path | None = None) -> SigilConfig:
    """Load YAML config; falls back to defaults if missing."""
    _ensure_dirs()
    _install_default_config()

    cfg_path = path or CONFIG_FILE
    if not cfg_path.exists():
        logger.warning("Config not found at %s – using built-in defaults", cfg_path)
        return SigilConfig()

    with open(cfg_path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    tracking = _parse_tracking(raw.get("tracking", {}))
    recording = _parse_recording(raw.get("recording", {}))
    execution = _parse_execution(raw.get("execution", {}))
    overlay_settings = _parse_overlay(raw.get("overlay_settings", {}))
    gestures = _parse_gestures(raw.get("gestures", []))

    return SigilConfig(
        tracking=tracking,
        recording=recording,
        execution=execution,
        overlay_settings=overlay_settings,
        gestures=gestures,
        log_level=raw.get("log_level", "info"),
        overlay=raw.get("overlay", True),
        daemon=raw.get("daemon", True),
    )


def save_config(cfg: SigilConfig, path: Path | None = None) -> None:
    """Serialize config back to YAML."""
    cfg_path = path or CONFIG_FILE
    _ensure_dirs()

    data: dict[str, Any] = {
        "log_level": cfg.log_level,
        "overlay": cfg.overlay,
        "daemon": cfg.daemon,
        "tracking": {
            "num_hands": cfg.tracking.num_hands,
            "min_detection_confidence": cfg.tracking.min_detection_confidence,
            "min_tracking_confidence": cfg.tracking.min_tracking_confidence,
            "model_asset_path": cfg.tracking.model_asset_path,
            "camera_id": cfg.tracking.camera_id,
            "frame_width": cfg.tracking.frame_width,
            "frame_height": cfg.tracking.frame_height,
            "target_fps": cfg.tracking.target_fps,
            "low_fps_fallback": cfg.tracking.low_fps_fallback,
            "zoom": cfg.tracking.zoom,
        },
        "recording": {
            "hotkey": cfg.recording.hotkey,
            "samples_per_class": cfg.recording.samples_per_class,
            "min_samples": cfg.recording.min_samples,
            "max_samples": cfg.recording.max_samples,
            "output_dir": cfg.recording.output_dir,
        },
        "execution": {
            "confirm_frames": cfg.execution.confirm_frames,
            "blanking_ms": cfg.execution.blanking_ms,
            "confidence_threshold": cfg.execution.confidence_threshold,
        },
        "overlay_settings": {
            "position": cfg.overlay_settings.position,
            "width": cfg.overlay_settings.width,
            "height": cfg.overlay_settings.height,
            "opacity": cfg.overlay_settings.opacity,
            "camera_preview": cfg.overlay_settings.camera_preview,
            "margin": cfg.overlay_settings.margin,
        },
        "gestures": [
            {
                "name": g.name,
                "type": g.type,
                "hand": g.hand,
                "condition": g.condition,
                "action": g.action,
                "cooldown_ms": g.cooldown_ms,
                "continuous": g.continuous,
                "enabled": g.enabled,
                "active_modes": g.active_modes,
                "mode_toggle": g.mode_toggle,
                "confirm_frames": g.confirm_frames,
                "blanking_ms": g.blanking_ms,
                "confidence_threshold": g.confidence_threshold,
            }
            for g in cfg.gestures
        ],
    }

    with open(cfg_path, "w") as fh:
        yaml.dump(data, fh, default_flow_style=False, sort_keys=False)
    logger.info("Config saved → %s", cfg_path)
