# Sigil – Gesture Control Framework for Hyprland

Sigil is a real-time hand gesture recognition system designed for the Hyprland Wayland compositor. It uses Google's MediaPipe for hand tracking and landmark detection, allowing users to control their desktop environment through webcam-based gestures.

## Project Overview

- **Core Goal:** Low-latency (<80ms), local-only gesture control for Hyprland WM.
- **Architecture:** 
    - `Tracker`: Captures webcam frames and extracts 21 3D landmarks per hand using MediaPipe Hand Landmarker.
    - `Classifier`: A hybrid engine supporting three gesture types:
        - **Instant:** Single-frame poses (e.g., Open Palm, Closed Fist) via MediaPipe Gesture Recognizer.
        - **Gradual:** Continuous features (e.g., pinch distance, palm velocity) triggering actions based on thresholds.
        - **Sequential:** Multi-frame state machines (e.g., swipe followed by a circle).
    - `ActionMapper`: Maps classified gestures to specific `hyprctl` dispatcher commands via YAML configuration.
    - `Executor`: Dispatches commands to Hyprland using a UNIX socket (preferred) or the CLI for low-latency execution.
    - `Overlay`: A Wayland-native (GTK4 + layer-shell) or OpenCV fallback visualization showing landmarks and status.
- **Tech Stack:** Python 3.11+, MediaPipe, OpenCV, NumPy, GTK4 (via PyGObject), Cairo, PyYAML.

## Building and Running

### Prerequisites
- **OS:** Linux (Arch/CachyOS recommended) running Hyprland.
- **Dependencies:** `gtk4`, `gtk4-layer-shell`, `python-gobject`, `python-cairo`, `opencv`.

### Installation
```bash
# Clone the repository
git clone https://github.com/prathamesh/sigil.git
cd sigil

# Install in editable mode with all extras
pip install -e ".[train,wayland,dev]"
```

### Key Commands
- **Start Daemon:** `sigil run` (starts tracking and overlay).
- **Record Gestures:** `sigil record <gesture_name> --mode instant` (collects training data).
- **Train Models:** `sigil train --type all` (retrains classifiers on recorded data).
- **Configuration:** `sigil config --edit` (opens `~/.config/sigil/config.yaml` in your `$EDITOR`).

## Development Conventions

- **Linting & Formatting:** Use `ruff` for code style and linting.
- **Type Safety:** `mypy` is used for static type checking; all new functions should have type hints and `disallow_untyped_defs = true` is enforced.
- **Testing:** Use `pytest` for unit and integration tests.
- **Logging:** Use the standard `logging` module. Verbosity can be adjusted via CLI flags (`-v`, `-vv`).
- **Performance:** Maintain frame rates ≥ 30 fps and keep CPU usage low (<25% on modern 8-core CPUs).
- **Modular Design:** Keep Tracker, Classifier, and Executor logic decoupled. Use the `Daemon` class in `sigil/daemon.py` to orchestrate component interaction.
- **Wayland Integration:** Prefer GTK4 Layer Shell for UI components to ensure proper integration with the compositor (blur, alpha, positioning).

## Data Paths
- **Config:** `~/.config/sigil/config.yaml`
- **Models:** `~/.local/share/sigil/models/`
- **Recordings:** `~/.local/share/sigil/recordings/`
- **Logs:** `~/.local/share/sigil/logs/`
