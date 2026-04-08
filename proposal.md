# Sigil Project Proposal

## Title
**Sigil: Gesture Control Framework for Hyprland Window Manager**

A real-time hand gesture recognition system enabling keyboard-free desktop control through webcam-based dual-hand tracking on Linux/Wayland environments.

---

## PS (Problem Statement)

Modern desktop workflows remain heavily dependent on keyboard and mouse input, limiting productivity in scenarios where hands-free or alternative input methods would be beneficial. Existing gesture control solutions suffer from:

- **Hardware Barriers**: Requiring specialized depth cameras or expensive peripherals
- **Platform Limitations**: Primarily targeting mobile/tablet platforms or proprietary desktop environments
- **Performance Issues**: High latency (>150ms) or excessive CPU usage (>50%) during operation
- **Limited Customization**: Pre-defined gesture sets with no user training capabilities
- **Ecosystem Lock-in**: Lack of integration with modern tiling window managers like Hyprland on Wayland

For power users on Linux systems—particularly those running lightweight, efficient window managers like Hyprland on Arch-based distributions—no viable solution exists for accurate, low-latency, customizable gesture control using standard webcam hardware.

**Target User**: Prathamesh, a Linux power user in Mumbai running CachyOS/Hyprland on a modern Ryzen 7040+ laptop, seeks to augment or replace keyboard shortcuts with intuitive air gestures for window management, workspace navigation, and media control during extended desk sessions.

**Core Problem**: How can we enable sub-80ms latency, ≥85% accuracy custom gesture recognition using only a standard webcam, with minimal CPU overhead (<25% on 8-core systems), while providing an intuitive recording/training workflow for personalized gesture vocabularies?

---

## Objective

### Primary Goal
Develop a production-ready gesture control framework that enables users to **record, train, and map custom hand gestures to Hyprland window manager commands** with performance, accuracy, and resource efficiency suitable for daily desktop use.

### Specific Objectives

**1. Performance Targets**
- End-to-end latency: **<80ms** (target <50ms) from gesture detection to Hyprland action execution
- Tracking frame rate: **30-60 fps** with dual-hand landmark detection
- CPU utilization: **<25%** on modern 8-core laptops during idle tracking
- Inference latency: **<30ms** per frame for gesture classification

**2. Accuracy & Reliability**
- Dual-hand landmark detection: **≥95%** accuracy at 30 fps under office lighting
- Custom gesture classification: **≥85%** accuracy with 50-150 samples per gesture class
- Zero false positives in idle state (no unintended command triggers)
- Graceful degradation to 15 fps under low-resource conditions

**3. User Experience**
- **Instant recording mode**: Capture gesture samples via hotkey toggle
- **One-click training**: Retrain models with single command after recording
- **Flexible mapping**: YAML-based configuration linking gestures to `hyprctl` commands
- **Visual feedback**: Optional overlay showing landmarks, mode, and confidence scores

**4. Gesture Type Support**
- **Instant gestures**: Single-frame static poses (e.g., peace sign, thumbs up)
- **Gradual gestures**: Continuous values (e.g., pinch distance 0-1 for zoom)
- **Sequential gestures**: Multi-frame temporal patterns (e.g., swipe sequences)

**5. Privacy & Deployment**
- **100% local processing**: No cloud dependencies or data transmission
- **Lightweight install**: pip-based or AUR package for Arch Linux
- **Minimal dependencies**: MediaPipe, OpenCV, PyYAML—no heavy frameworks

---

## Methodology

### Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Webcam     │───▶│  Tracking    │───▶│  Gesture    │───▶│  Hyprland    │
│  (v4l2)     │    │  Engine      │    │  Classifier │    │  Executor    │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────────┘
                           │                    │
                           ▼                    ▼
                   ┌──────────────┐    ┌─────────────┐
                   │  Recording   │    │   Config    │
                   │     Mode     │    │   Mapping   │
                   └──────────────┘    └─────────────┘
```

### Phase 1: Tracking Engine (M1)
**Technology**: MediaPipe Hand Landmarker (2026 task bundle)

**Implementation**:
- Initialize `HandLandmarker` with configuration:
  - `num_hands=2` for dual-hand tracking
  - `min_hand_detection_confidence=0.7`
  - `min_tracking_confidence=0.6`
  - `running_mode=VIDEO` for continuous frames
- Extract **21 3D landmarks per hand** + handedness + world coordinates
- Fallback to legacy `mp_hands` API if task format changes

**Deliverable**: Real-time dual-hand visualization with OpenCV overlay showing landmark connections

---

### Phase 2: Gesture Classification (M2-M3)

#### 2A. Instant Gesture Recognition
**Approach**: Custom-trained Random Forest on landmark geometry

**Feature Engineering** (101 features per frame):
- Normalized 3D landmark positions (63 features)
- Finger curl states: open/closed per finger (5 features)
- Inter-landmark distances: palm-to-fingertip (5 features)
- Joint angles: MCP/PIP flexion (10 features)
- Fingertip adjacency angles (4 features)
- Palm orientation vector (3 features) ✨
- Finger spread metric (1 feature) ✨
- Hand compactness ratio (1 feature) ✨

**Model Training**:
```python
RandomForestClassifier(
    n_estimators=250,
    max_depth=20,
    min_samples_split=3,
    max_features='sqrt',
    class_weight='balanced',
    bootstrap=True,
    oob_score=True
)
```

**Data Augmentation** (12x per sample):
- Rotation: ±20° (80% probability)
- Scale: ±10% (60% probability)
- Translation: ±2% (40% probability)
- Landmark jitter: ±0.005 (100% probability)

#### 2B. Gradual Gestures
**Approach**: Derived geometric features from landmarks

**Examples**:
- Pinch strength: Normalized thumb-index fingertip distance → [0, 1]
- Hand openness: Average finger curl metric → [0, 1]
- Two-hand distance: Inter-palm distance for zoom gestures

**Output**: Continuous values for real-time control

#### 2C. Sequential Gestures
**Approach**: Temporal buffering + pattern matching

**Buffer Configuration**:
- Window size: 8-30 frames (0.27-1.0 seconds at 30 fps)
- Algorithms: Dynamic Time Warping (DTW) or small LSTM on landmark deltas
- Use case: Swipe patterns, directional gestures

---

### Phase 3: Recording & Training Workflow (M2)

#### Recording Mode
**Trigger**: Hotkey `Super+Alt+G` (configurable in `config.yaml`)

**Sub-modes**:
1. **Instant**: Capture static pose samples
2. **Gradual**: Record continuous value ranges
3. **Sequential**: Capture multi-frame temporal patterns

**Data Collection**:
- Target: **50-200 samples per gesture class**
- Storage format:
  - Instant: Normalized landmark JSON files + optional RGB frames
  - Sequential: Time-series CSV with 101 features × frames
- Directory structure: `~/.local/share/sigil/recordings/{gesture_name}/`

**UI Feedback**:
- Overlay displays: mode, gesture name, sample count, current confidence
- Controls: Discard last sample, finalize class

#### Training Process
**Command**: `sigil train` or programmatic API

**Workflow**:
1. Load samples from recordings directory
2. Apply 12x augmentation pipeline
3. Train Random Forest classifier
4. Validate with OOB score + test split
5. Export model: `~/.local/share/sigil/models/custom_gestures.pkl`
6. Log feature importance and accuracy metrics

**Metrics Logged**:
- Test accuracy, precision, recall, F1-score per class
- OOB score for generalization estimate
- Top 10 feature importance rankings
- Inference time benchmarks

---

### Phase 4: Mapping & Execution (M4)

#### Configuration Format
**File**: `~/.config/sigil/config.yaml`

```yaml
gestures:
  - name: "close_window"
    type: instant
    hands: right
    action: "hyprctl dispatch killactive"
    confidence_threshold: 0.70
  
  - name: "zoom_in"
    type: gradual
    hands: both
    value_range: [0.3, 1.0]
    action: "hyprctl keyword misc:cursor_zoom_factor {value}"
  
  - name: "workspace_next"
    type: sequential
    pattern: "swipe_right"
    action: "hyprctl dispatch workspace +1"
```

#### Execution Engine
**Runtime Loop** (asyncio):
1. Capture frame from webcam (v4l2/pipewire)
2. Detect hands and extract landmarks
3. Classify gesture(s) with confidence scores
4. Apply temporal smoothing (9-frame buffer, 2-frame confirmation)
5. Match against configured mappings (priority order)
6. Execute `hyprctl` command via socket/subprocess
7. Apply blanking period (1000ms) to prevent re-triggers

**Fallback**: Direct evdev input simulation if `hyprctl` unavailable

---

### Phase 5: Optimization & Polish (M5)

#### Performance Optimization
- **Model compression**: Prune Random Forest to <100 trees for <2ms inference
- **Multi-threading**: Separate threads for capture, inference, execution
- **Low-latency mode**: Reduce buffer sizes for <50ms end-to-end
- **Auto-tuning**: Adjust FPS based on CPU usage (30 fps → 15 fps fallback)

#### Temporal Smoothing
**Configuration**:
- `SMOOTHING_BUFFER_SIZE=9` frames
- `CONFIRM_FRAMES=2` consecutive matches required
- `BLANKING_PERIOD_MS=1000` post-trigger cooldown
- `CONFIDENCE_THRESHOLD=0.70` minimum score

**Logic**: Prevents flickering between gestures, reduces false positives

#### Debugging & Visualization
- **Overlay modes**:
  - Landmarks + connections (always)
  - Bounding boxes + handedness labels
  - Current gesture name + confidence bar
  - Recording status + sample count
- **Logging**: Debug/info/error levels to file + systemd journal

---

### Technology Stack

**Core Dependencies**:
- **MediaPipe** ≥0.10.14: Hand landmark detection
- **OpenCV** ≥4.9.0: Camera I/O and visualization
- **scikit-learn** ≥1.4: Random Forest classifier
- **NumPy** ≥1.26, <2.0: Feature computation
- **PyYAML** ≥6.0: Configuration management

**Optional**:
- **MediaPipe Model Maker**: Alternative TFLite training path
- **PyGObject + pycairo**: Wayland-native overlay rendering
- **JAX/JAXlib** <0.5: For Model Maker support

**Development Tools**:
- **Ruff**: Linting and formatting
- **mypy**: Type checking
- **pytest**: Unit and integration tests

**Platform Requirements**:
- **OS**: Linux (CachyOS/Arch), Wayland session
- **WM**: Hyprland ≥2026 latest
- **Hardware**: Webcam ≥720p, CPU with AVX2 (Ryzen 7040+)
- **Python**: 3.11+

---

### Development Timeline

| Milestone | Deliverables | Duration |
|-----------|-------------|----------|
| **M1** | Dual-hand tracking + visualization | 1-2 weeks |
| **M2** | Instant gesture recording + training | 2-3 weeks |
| **M3** | Gradual/sequential gesture support | 2-3 weeks |
| **M4** | Config mapping + Hyprland integration | 1-2 weeks |
| **M5** | Latency optimization + AUR packaging | 1-2 weeks |

**Total**: ~9-12 weeks for MVP

---

### Success Criteria

**Functional**:
- ✅ User can record 3+ custom instant gestures with 50 samples each
- ✅ Training completes in <60 seconds for 5 gesture classes
- ✅ Gestures trigger corresponding Hyprland commands reliably
- ✅ System runs as background daemon with auto-start support

**Performance**:
- ✅ Latency <80ms (measured from gesture start to window action)
- ✅ CPU usage <25% during idle tracking at 30 fps
- ✅ Classification accuracy ≥85% on held-out test set
- ✅ Zero false positives in 5-minute idle observation

**Usability**:
- ✅ Hotkey toggle between modes works consistently
- ✅ Visual overlay provides clear feedback
- ✅ Config file syntax documented and validated
- ✅ Installation via `pip install sigil` or `yay -S sigil`

---

### Risk Mitigation

| Risk | Impact | Mitigation Strategy |
|------|--------|-------------------|
| MediaPipe API breaking changes | High | Version pinning + fallback to legacy API |
| Wayland input simulation latency | Medium | Benchmark vs X11, use evdev fallback |
| Low accuracy on diverse skin tones | High | Test on varied datasets, adjust lighting |
| Sequential classifier complexity | Medium | Start with simple DTW, defer LSTM to v2 |
| Multi-monitor cursor issues | Low | Document limitation, defer to future work |

---

### Future Enhancements (Out of Scope for v1.0)

- Mobile/Android companion app
- Multi-user profile management
- 3D depth camera support (RealSense)
- Non-Hyprland WM adapters (Sway, i3)
- Voice/gaze hybrid multi-modal control
- Cloud-based gesture sharing marketplace

---

## References

- **Architecture**: Landmark-based geometric feature extraction
- **Augmentation**: Rotation (±20°) + scale (±10%) + translation (±2%) + jitter
- **Model**: scikit-learn RandomForestClassifier with class balancing
- **Feature Set**: 101-dimensional hand geometry features
- **Inspiration**: kinivi/hand-gesture-recognition-mediapipe + MediaPipe Gesture Recognizer examples

---

**Document Version**: 1.0  
**Last Updated**: April 2026  
**Author**: Prathamesh  
**Project Repository**: [github.com/prathamesh/sigil](https://github.com/prathamesh/sigil)
