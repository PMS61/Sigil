# Sigil (§5.3–5.5)

**Sigil** is a high-performance, low-latency gesture control framework designed specifically for the **Hyprland** Wayland compositor. It enables users to record, train, and map custom hand gestures (static, continuous, and temporal) to `hyprctl` dispatcher commands and system actions using only a standard webcam.

Replace or supplement your keyboard input with intuitive air gestures for window management, workspace navigation, media control, and application launching.

## ✨ Key Features

- **Dual-Hand Tracking:** Real-time 3D landmark detection for both hands simultaneously using MediaPipe.
- **Gesture Class Diversity:**
  - **Instant (Static):** Poses like "Closed_Fist", "Victory", or custom trained gestures.
  - **Gradual (Continuous):** Dynamic features like "Two-Hand Distance" for volume/zoom control.
  - **Sequential (Temporal):** Multi-frame patterns for complex macro execution (e.g., swipe patterns).
- **Custom Recording & Training:** Built-in pipeline to record your own gestures and retrain TFLite models in minutes.
- **Deep Hyprland Integration:** Direct mapping of gestures to `hyprctl dispatch` commands in a clean YAML configuration.
- **Low-Latency Architecture:** Optimized for CachyOS/Arch with target end-to-end latency < 80ms.
- **Flexible Overlays:** Choice of GTK4 Layer Shell, standard GTK4, or OpenCV for visual feedback.

## 🛠 Tech Stack

- **Language:** Python 3.11+
- **Inference:** MediaPipe Hand Landmarker & Gesture Recognizer (TFLite)
- **Computer Vision:** OpenCV
- **Windowing/UI:** GTK4 with `gtk4-layer-shell` (Wayland native)
- **Configuration:** YAML
- **OS Target:** Linux (optimized for Arch/CachyOS with Hyprland)

## 📋 Prerequisites

- **Hardware:** Webcam (720p+ recommended)
- **Software:**
  - Hyprland WM
  - Python 3.11 or 3.12
  - `v4l-utils` (for camera access)
  - `playerctl` (for default media bindings)
  - `libadwaita` / `gtk4` (for GUI/Overlay)

## 🚀 Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/user/sigil.git
cd sigil
```

### 2. Set Up Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install Dependencies

```bash
# Core dependencies
pip install .

# Optional: For training custom models
pip install ".[train]"

# Optional: For Wayland native overlays (GTK4 Layer Shell)
pip install ".[wayland]"
```

### 4. Configuration

Sigil looks for `config.yaml` in its root or `~/.config/sigil/config.yaml`.

```bash
# Generate/copy the default configuration
cp sigil/default_config.yaml config.yaml
```

### 5. Run the Daemon

```bash
sigil run
```

This starts the gesture tracking in the background and enables the overlay.

## 🎮 Usage & Commands

| Command | Description |
| :--- | :--- |
| `sigil run` | Start the Sigil daemon and begin gesture tracking. |
| `sigil record <class>` | Enter recording mode to collect samples for a new gesture. |
| `sigil train` | (Re)train models from your recorded samples. |
| `sigil config --edit` | Open your configuration file in `$EDITOR`. |
| `sigil gui` | Launch the graphical configuration app. |
| `sigil --version` | Show version information. |

## ⚙️ Configuration Reference

The `config.yaml` file controls tracking sensitivity, overlays, and gesture-to-action mappings.

### Global Settings
| Variable | Description | Default |
| :--- | :--- | :--- |
| `log_level` | `debug`, `info`, `error` | `info` |
| `overlay` | Enable/Disable the visual landmark overlay | `true` |
| `daemon` | Run as a background process | `true` |

### Gesture Mappings
Gestures are defined in the `gestures` list. Each entry requires:
- `name`: Unique identifier.
- `type`: `instant` or `gradual`.
- `hand`: `left`, `right`, or `both`.
- `condition`: Pose name (MediaPipe built-ins: `Closed_Fist`, `Open_Palm`, `Victory`, etc.) or dynamic rule.
- `action`: Shell command or `hyprctl dispatch` string.

**Example Mapping:**
```yaml
- name: close_window
  type: instant
  hand: right
  condition: { pose: "Closed_Fist" }
  action: "hyprctl dispatch killactive"
  cooldown_ms: 800
```

## 📂 Architecture Overview

```text
sigil/
├── classifier.py     # Gesture inference & logic
├── tracker.py        # MediaPipe landmark extraction
├── executor.py       # Action dispatch (hyprctl)
├── daemon.py         # Main background loop
├── recorder.py       # Dataset collection for training
├── trainer.py        # TFLite model retraining
├── ui/               # Overlay & GUI implementations
│   ├── app.py        # Config GUI
│   └── wayland_overlay.py # GTK4 Layer Shell implementation
└── config.py         # YAML configuration handling
```

### Request Lifecycle
1. **Input:** OpenCV captures frames from `/dev/video0`.
2. **Inference:** MediaPipe processes frames → Hand Landmarks (21 points).
3. **Classification:** Landmarks are passed to the TFLite model (Instant) or Rule Engine (Gradual).
4. **Validation:** Gesture is confirmed over `N` frames to prevent jitter.
5. **Action:** `executor.py` triggers the mapped command via `subprocess` or `hyprctl` socket.

## 🧪 Testing

```bash
# Run the test suite
pytest

# Test specific components
pytest tests/test_classifier.py
```

## 🛠 Troubleshooting

- **No Camera Access:** Ensure your user is in the `video` group: `sudo usermod -aG video $USER`.
- **Low FPS:** Check if `TF_ENABLE_ONEDNN_OPTS=0` and `CUDA_VISIBLE_DEVICES=""` are set (Sigil defaults to CPU inference for stability).
- **Overlay Not Visible:** If using Hyprland, ensure you have `gtk4-layer-shell` installed. You can force the OpenCV backend with `sigil run --backend opencv`.

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.
