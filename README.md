# Sigil – Gesture Control Framework for Hyprland

Control your Hyprland desktop with hand gestures via webcam.

Sigil uses **MediaPipe Hand Landmarker** to track dual-hand landmarks in real time, classifies gestures (instant / gradual / sequential), and maps them to `hyprctl` dispatcher commands — all locally, with zero cloud dependency.

---

## Features

| Capability | Details |
|---|---|
| **Dual-hand tracking** | 21 3-D landmarks per hand @ 30–60 fps |
| **Instant gestures** | Single-frame poses (pinch, open palm, thumbs up/down …) |
| **Gradual gestures** | Continuous features — pinch zoom, finger curl, palm velocity |
| **Sequential gestures** | Multi-frame state machines (swipe → action) |
| **Recording mode** | Collect custom samples interactively, per-class |
| **One-click training** | MediaPipe Model Maker (instant) + scikit-learn (dynamic) |
| **YAML config** | Gesture → hyprctl mapping, hot-reloadable |
| **Overlay** | Optional OpenCV window with landmarks + labels + FPS |

---

## Requirements

- **OS:** Arch Linux / CachyOS (or any distro running Hyprland)
- **Python:** 3.11+
- **Webcam:** ≥ 720p USB / integrated
- **Hyprland:** Running Wayland session

---

## Installation

```bash
# Clone
git clone https://github.com/prathamesh/sigil.git
cd sigil

# Install (editable)
pip install -e .

# Or with training extras
pip install -e ".[train]"

# Or with dev tools
pip install -e ".[dev]"
```

### AUR (planned)

```bash
yay -S sigil-gesture
```

---

## Quick Start

```bash
# 1. Start the daemon (opens webcam + overlay)
sigil run

# 2. Record a custom gesture (50+ samples recommended)
sigil record my_gesture --mode instant --hand right -n 100

# 3. Retrain models
sigil train

# 4. Edit config to map the new gesture
sigil config --edit
```

---

## CLI Reference

```
sigil run                       Start the daemon
sigil record <class> [options]  Record gesture samples
  -m, --mode {instant,gradual,sequential}
  --hand {left,right,both}
  -n, --num-samples N           Auto-stop after N samples
sigil train [options]           Retrain models
  --type {instant,dynamic,all}
  --epochs N  --batch N
sigil config                    Print current config
  --path                        Print config file path
  --edit                        Open in $EDITOR
sigil -V                        Print version
sigil -v / -vv                  Increase log verbosity
```

---

## Configuration

Config lives at `~/.config/sigil/config.yaml`. A default is generated on first run.

```yaml
tracking:
  camera_index: 0
  target_fps: 30
  num_hands: 2
  min_detection_confidence: 0.7
  min_tracking_confidence: 0.6
  low_fps_fallback: 15

overlay: true

gestures:
  - name: pinch_close
    type: instant
    hands: [right]
    pose: Closed_Fist
    action: "hyprctl dispatch killactive"

  - name: open_palm_fullscreen
    type: instant
    hands: [right]
    pose: Open_Palm
    action: "hyprctl dispatch fullscreen 1"
```

### Gesture types

| Type | `type` value | How it works |
|---|---|---|
| Instant | `instant` | Matched by MediaPipe Gesture Recognizer pose label |
| Gradual | `gradual` | Fires when a derived feature crosses a threshold with direction |
| Sequential | `sequential` | State machine: ordered sequence of poses within a time window |

---

## Data & Model Paths

| Path | Purpose |
|---|---|
| `~/.config/sigil/config.yaml` | Configuration |
| `~/.local/share/sigil/models/` | Downloaded + trained models |
| `~/.local/share/sigil/recordings/` | Recorded samples per class |
| `~/.local/share/sigil/logs/` | Log files |

---

## Architecture

```
webcam
  │
  ▼
Tracker (MediaPipe Hand Landmarker)
  │
  ├── InstantClassifier  (Gesture Recognizer .task/.tflite)
  ├── GradualClassifier  (feature thresholds)
  └── SequentialClassifier (state machine)
         │
         ▼
    ActionMapper (config.yaml)
         │
         ▼
    Executor (hyprctl socket / CLI)
         │
         ▼
    Hyprland WM actions
```

---

## Development

```bash
pip install -e ".[dev]"
ruff check sigil/
mypy sigil/
```

---

## Performance Targets (§2)

- End-to-end latency: **< 80 ms** (target < 50 ms on Ryzen 7040+)
- Tracking accuracy: **≥ 95%** at 30 fps
- Custom gesture accuracy: **≥ 80%** after 50–150 samples
- CPU usage: **< 25%** on 8-core laptop at 30 fps idle tracking
- Graceful degradation to 15 fps on sustained low FPS

---

## Privacy

All processing is **100% local**. No network calls except the one-time model download from Google's CDN on first run.

---

## License

MIT
