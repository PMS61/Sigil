Sigil Product Requirements Document (PRD) v1.0  
Project: Sigil – Gesture Control Framework for Hyprland  
Target Environment: CachyOS (Arch-based) laptop running Hyprland WM  
Date: March 2026  
Version: 1.0 (Initial)  

**1. Overview / Purpose**  
Sigil enables users to record, store, classify, and map custom hand gestures (instant/static, gradual/continuous, sequential/temporal) detected via webcam to Hyprland keybinds and dispatcher commands.  
Core goal: Replace/supplement keyboard input with accurate, low-latency, dual-hand air gestures in desktop sessions.  
No hardware beyond standard webcam required.  

**2. Objectives & Success Metrics**  
- End-to-end latency < 80 ms from gesture start to Hyprland action trigger (target < 50 ms on Ryzen 7040+ class hardware).  
- Dual-hand tracking accuracy: ≥ 95% landmark detection at 30 fps under typical office lighting + moderate motion.  
- Custom gesture recording: ≥ 80% classification accuracy on user-recorded dataset after 50–150 samples per class.  
- Minimal CPU usage: < 25% on modern 8-core laptop during idle tracking (30 fps).  
- Zero false positives in default idle state (no unintended triggers).  

**3. Scope**  
In scope:  
- Real-time dual-hand landmark tracking (MediaPipe Hand Landmarker or Hands solution, num_hands=2).  
- Gesture types: instant (single-frame pose), gradual (continuous value e.g. pinch openness 0–1), sequential (multi-frame series for complex patterns).  
- Recording mode for custom gestures (image frames for Model Maker + landmark time-series CSV/JSON).  
- Training/customization path: MediaPipe Gesture Recognizer .tflite for instant; custom lightweight models/rules for dynamic.  
- Mapping layer: YAML/JSON config linking gesture outputs to hyprctl dispatch commands.  
- UI: Minimal overlay (landmarks visualization, mode indicators, recording status) via OpenCV window or Wayland-native if feasible.  
- Input: Standard USB/webcam via v4l2/pipewire.  

Out of scope:  
- Mobile/Android support.  
- Multi-user profiles.  
- 3D gestures requiring depth camera.  
- Integration with non-Hyprland WMs/compositors.  
- Voice/gaze hybrid control.  
- Accessibility features beyond basic gesture fallback.  

**4. User Personas & Use Cases**  
Primary persona: Prathamesh – Linux power user on CachyOS/Hyprland laptop in Mumbai, seeks keyboard-free window/workspace/media control during desk sessions.  

Use cases:  
- Instant: Right-hand pinch → close window (killactive).  
- Gradual: Two-hand distance increase → zoom in (simulate mouse wheel or hyprctl keyword).  
- Sequential: Multi-frame gesture patterns → switch workspace + launch app.  
- Recording: Enter mode → perform gesture 50+ times → name class → train/map.  
- Runtime: Background daemon starts with Hyprland → optional hotkey toggle.  

**5. Functional Requirements**  
5.1 Tracking Engine  
- Use MediaPipe Hand Landmarker task (latest 2025–2026 bundle: hand_landmarker.task float16).  
- Config: num_hands=2, min_hand_detection_confidence=0.7, min_tracking_confidence=0.6, running_mode=VIDEO.  
- Output: 21 3D landmarks/hand + handedness + world coords per frame.  
- Fallback to legacy mp_hands if landmarker API changes.  

5.2 Gesture Classification  
- Instant: MediaPipe Gesture Recognizer (.task or custom .tflite via Model Maker).  
- Gradual: Derived features (e.g. normalized thumb-index distance, finger curl angles).  
- Sequential: Time-series buffer (8–30 frames) → DTW, state machine, or small LSTM on flattened diffs.  
- Hybrid: Instant uses recognizer; dynamic uses landmark rules/model.  

5.3 Recording Mode  
- Hotkey toggle (configurable, default Super+Alt+G).  
- Sub-modes: instant / gradual / sequential.  
- Per class: collect 50–200 samples.  
- For instant: save cropped/annotated RGB frames to class folders (Model Maker compatible).  
- For dynamic: save normalized landmark sequences + timestamps + derived features to CSV/JSON.  
- UI feedback: overlay shows current mode, class name, sample count, confidence.  
- Discard/redo last sample.  

5.4 Training & Model Management  
- Instant: Use mediapipe_model_maker → train on recorded images → export custom_gesture.tflite.  
- Dynamic: Simple scikit-learn MLP/KNN or tiny Keras model on features; or rule-based thresholds stored in config.  
- One-click retrain command after new recordings.  

5.5 Mapping & Execution  
- Config file (~/.config/sigil/config.yaml):  
  gestures list with name, type, hand(s), condition (pose/rules/sequence), action (hyprctl dispatch string).  
- Priority ordering: first match fires.  
- Execution: asyncio loop → hyprctl socket/keyword or direct evdev fallback.  

5.6 Visualization & Debugging  
- Optional OpenCV window: show camera feed + drawn landmarks + gesture label/confidence.  
- Log level: debug/info/error (to file + journalctl).  

**6. Non-Functional Requirements**  
- Platform: Linux (CachyOS Arch), Python 3.11+ or Rust.  
- Dependencies: mediapipe, opencv-python, pyyaml, asyncio; avoid heavy frameworks.  
- Performance: 30–60 fps tracking; inference < 30 ms CPU.  
- Privacy: No cloud; all processing local.  
- Reliability: Graceful degradation on low FPS (drop to 15 fps mode).  
- Install: pip-based or AUR package target.  

**7. Dependencies & Constraints**  
- Hardware: Webcam ≥ 720p; CPU with AVX2/AVX-512 preferred.  
- Software: Hyprland ≥ latest 2026; wayland session.  
- External: MediaPipe models downloaded on first run.  

**8. Timeline & Milestones (indicative)**  
- M1: Dual-hand landmark tracking + visualization.  
- M2: Instant gesture recognition (default + custom recording).  
- M3: Gradual + sequential support + basic rules.  
- M4: Config mapping + Hyprland integration.  
- M5: Polish, latency optimization, packaging.  

**9. Open Questions / Risks**  
- MediaPipe task API stability post-2025.  
- Wayland screenshot/input simulation latency vs X11 fallback.  
- Best sequential classifier (DTW vs tiny NN vs HMM) for low-resource.  
- Handling multi-monitor cursor simulation if needed later.  
- User testing dataset variety (skin tones, lighting in Mumbai apartments).  

Implement core pipeline first (tracking → simple mapping), then layer recording/training. Use kinivi/hand-gesture-recognition-mediapipe as scaffold, merge official Gesture Recognizer examples.
