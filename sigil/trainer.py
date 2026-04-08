"""Model Training & Management (§5.4).

Provides one-click retrain commands using the landmark architecture with data augmentation:
  - Raw 21-point landmarks → Augmentation (12x) → 101-dim features → Random Forest
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sigil.config import MODELS_DIR, RECORDINGS_DIR
from sigil.utils import LandmarkAugmenter, extract_96_features

logger = logging.getLogger(__name__)

AUGMENTATION_MULTIPLIER = 12


def _load_raw_landmarks(
    data_dir: Path, class_dirs: list[str] | None = None
) -> tuple[list[np.ndarray], list[int], list[str]]:
    """Load raw 21-point 3D landmarks from per-class sub-folders.

    Supports:
    - JSON files with raw landmarks (new format)
    - JPG images (legacy format - extracts landmarks via MediaPipe)

    Returns (raw_landmarks_list, labels, class_names).
    """
    if class_dirs is None:
        class_dirs = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]

    all_landmarks: list[np.ndarray] = []
    all_labels: list[int] = []
    final_classes: list[str] = []

    for cls_idx, cls in enumerate(class_dirs):
        cls_dir = data_dir / cls
        cls_landmarks: list[np.ndarray] = []

        jsons = list(cls_dir.glob("*.json"))
        if jsons:
            for jf in sorted(jsons):
                with open(jf) as fh:
                    data: dict[str, Any] = json.load(fh)

                if isinstance(data, list):
                    for frame in data:
                        if "landmarks" in frame:
                            lm = np.array(frame["landmarks"], dtype=np.float32)
                            if lm.shape == (21, 3):
                                cls_landmarks.append(lm)
                else:
                    if "landmarks" in data:
                        lm = np.array(data["landmarks"], dtype=np.float32)
                        if lm.shape == (21, 3):
                            cls_landmarks.append(lm)

        jpgs = list(cls_dir.glob("*.jpg"))
        if not cls_landmarks and jpgs:
            try:
                import mediapipe as mp

                logger.info("Extracting landmarks from %d images in '%s'…", len(jpgs), cls)
                for img_path in sorted(jpgs):
                    lm = _extract_landmarks_from_image(img_path)
                    if lm is not None and lm.shape == (21, 3):
                        cls_landmarks.append(lm)
            except ImportError:
                logger.warning(
                    "MediaPipe not available - cannot extract landmarks from legacy images in '%s'. "
                    "Record new samples in GRADUAL mode to save landmarks directly.",
                    cls,
                )

        if cls_landmarks:
            final_classes.append(cls)
            all_landmarks.extend(cls_landmarks)
            all_labels.extend([cls_idx] * len(cls_landmarks))

    return all_landmarks, all_labels, final_classes


def _augment_and_extract(
    raw_landmarks: list[np.ndarray],
    labels: list[int],
    multiplier: int = AUGMENTATION_MULTIPLIER,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply augmentation and extract 101-dim features.

    Args:
        raw_landmarks: List of (21, 3) raw landmark arrays
        labels: Corresponding class labels
        multiplier: Augmentation factor (1 = no augmentation)

    Returns:
        (features, labels) with augmented data
    """
    augmenter = LandmarkAugmenter()
    features: list[np.ndarray] = []
    augmented_labels: list[int] = []

    for lm, label in zip(raw_landmarks, labels, strict=False):
        if multiplier > 1:
            augmented = augmenter.augment(lm, multiplier=multiplier)
        else:
            augmented = [lm]

        for aug_lm in augmented:
            feat = extract_96_features(aug_lm)
            features.append(feat)
            augmented_labels.append(label)

    return np.array(features, dtype=np.float32), np.array(augmented_labels)


def _extract_landmarks_from_image(img_path: Path) -> np.ndarray | None:
    """Extract raw 21-point 3D landmarks from a single JPG image using MediaPipe HandLandmarker."""
    import cv2
    import numpy as np

    from sigil.utils import ensure_model, landmark_to_array

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )
        from mediapipe import Image, ImageFormat

        model_path = ensure_model("hand_landmarker.task")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
        )

        with HandLandmarker.create_from_options(options) as landmarker:
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                return None

            landmarks = landmark_to_array(result.hand_landmarks[0])
            return landmarks

    except (ImportError, Exception) as e:
        logger.warning("Failed to extract landmarks from %s: %s", img_path, e)
        return None


def _extract_features_from_image(img_path: Path) -> dict[str, Any] | None:
    """Extract 96 geometric features from a single JPG image using MediaPipe HandLandmarker."""
    import cv2
    import numpy as np

    from sigil.recorder import Recorder
    from sigil.tracker import FrameResult, HandResult
    from sigil.utils import ensure_model, landmark_to_array

    img = cv2.imread(str(img_path))
    if img is None:
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            HandLandmarker,
            HandLandmarkerOptions,
            RunningMode,
        )
        from mediapipe import Image, ImageFormat

        model_path = ensure_model("hand_landmarker.task")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
        )

        with HandLandmarker.create_from_options(options) as landmarker:
            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)
            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                return None

            landmarks = landmark_to_array(result.hand_landmarks[0])
            handedness = result.handedness[0][0].category_name if result.handedness else "Right"

            fr = FrameResult(timestamp_ms=0)
            hr = HandResult(handedness=handedness, landmarks=landmarks)
            return Recorder._extract_features(fr, hr)

    except (ImportError, Exception):
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import HandLandmarker, RunningMode

        model_path = ensure_model("hand_landmarker.task")

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
            num_hands=1,
        )

        with HandLandmarker.create_from_options(options) as landmarker:
            from mediapipe import ImageFormat, Image

            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                return None

            landmarks = landmark_to_array(result.hand_landmarks[0])
            handedness = result.handedness[0][0].category_name if result.handedness else "Right"

            fr = FrameResult(timestamp_ms=0)
            hr = HandResult(handedness=handedness, landmarks=landmarks)
            return Recorder._extract_features(fr, hr)

    except (ImportError, Exception):
        return None

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import GestureRecognizer, RunningMode

        model_path = ensure_model("gesture_recognizer.task")

        options = GestureRecognizerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=RunningMode.IMAGE,
        )

        with GestureRecognizer.create_from_options(options) as recognizer:
            mp_image = type("obj", (object,), {"image_format": 1})()
            from mediapipe import ImageFormat, Image

            mp_image = Image(image_format=ImageFormat.SRGB, data=rgb)

            recognition = recognizer.recognize(mp_image)

            if not recognition.gestures:
                return None

            handedness = (
                recognition.handedness[0][0].category_name if recognition.handedness else "Right"
            )

            fr = FrameResult(timestamp_ms=0)
            hr = HandResult(
                handedness=handedness, landmarks=np.random.rand(21, 3).astype(np.float32)
            )
            return Recorder._extract_features(fr, hr)

    except (ImportError, Exception):
        return None


def train_instant(
    data_dir: Path | None = None,
    output_path: Path | None = None,
    augmentation: int = AUGMENTATION_MULTIPLIER,
) -> Path:
    """Train a Random Forest classifier with data augmentation.

    Pipeline:
    1. Load raw 21-point landmarks from JSON files
    2. Apply augmentation (12x): rotate ±20°, scale ±10%, translate ±2%, add jitter
    3. Extract 101-dim geometric features from augmented data
    4. Train Random Forest (250 trees with class balancing)

    Args:
        data_dir: Path to recordings directory
        output_path: Path for output model file
        augmentation: Multiplier for data augmentation (default 12x)

    Returns:
        Path to trained model
    """
    try:
        import joblib
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for training. Install with: pip install 'sigil[train]'"
        ) from exc

    data_dir = data_dir or RECORDINGS_DIR
    out = output_path or (MODELS_DIR / "custom_gesture.pkl")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        raise FileNotFoundError(f"Recordings directory not found: {data_dir}")

    class_dirs = [d.name for d in data_dir.iterdir() if d.is_dir()]
    class_names = class_dirs  # Already a list of names

    if "None" not in class_names:
        if "none" in class_names:
            logger.info("Renaming 'none' class to 'None' for compatibility")
            (data_dir / "none").rename(data_dir / "None")
        else:
            raise ValueError(
                "A 'None' class is required for background/noise samples. "
                "Please record some samples for a gesture named 'None'."
            )

    if len(class_names) < 2:
        raise ValueError(
            f"Need at least 2 classes to train (found: {class_names}). "
            "Record at least one gesture PLUS the 'None' background class."
        )

    logger.info("Loading raw landmarks from %s …", data_dir)
    raw_landmarks, raw_labels, final_classes = _load_raw_landmarks(data_dir)

    if len(raw_landmarks) < 10:
        raise ValueError(f"Not enough raw samples ({len(raw_landmarks)}). Need ≥ 10.")

    logger.info(
        "Loaded %d raw samples across %d classes: %s",
        len(raw_landmarks),
        len(final_classes),
        final_classes,
    )

    logger.info("Applying %dx augmentation and extracting features …", augmentation)
    features, labels = _augment_and_extract(raw_landmarks, raw_labels, multiplier=augmentation)

    logger.info("Augmented dataset: %d samples (%dx)", len(features), augmentation)

    feat_train, feat_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test = scaler.transform(feat_test)

    clf = RandomForestClassifier(
        n_estimators=250,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        oob_score=True,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    logger.info("Training Random Forest (250 trees with class balancing) …")
    clf.fit(feat_train, y_train)

    accuracy = clf.score(feat_test, y_test)
    oob_score = clf.oob_score_ if hasattr(clf, "oob_score_") else 0.0
    logger.info("Model accuracy: %.4f (OOB: %.4f)", accuracy, oob_score)
    report = classification_report(y_test, clf.predict(feat_test), target_names=final_classes)
    logger.info("Classification report:\n%s", report)
    
    feature_importance = clf.feature_importances_
    top_features = np.argsort(feature_importance)[-10:][::-1]
    logger.info("Top 10 feature indices: %s", top_features)

    artifact = {
        "model": clf,
        "scaler": scaler,
        "class_names": final_classes,
        "augmentation": augmentation,
        "original_samples": len(raw_landmarks),
        "augmented_samples": len(features),
        "accuracy": accuracy,
        "oob_score": oob_score,
        "feature_importance": feature_importance,
    }
    joblib.dump(artifact, out)
    logger.info("Exported geometric model → %s", out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3.  One-click retrain (§5.4)
def retrain(
    data_dir: Path | None = None,
    augmentation: int = AUGMENTATION_MULTIPLIER,
) -> dict[str, Path]:
    """Retrain geometric model from current recordings.

    Returns dict of ``{model_name: output_path}`` for successfully trained models.
    """
    results: dict[str, Path] = {}
    data_dir = data_dir or RECORDINGS_DIR

    try:
        results["geometric"] = train_instant(data_dir=data_dir, augmentation=augmentation)
    except Exception as exc:
        logger.error("Geometric training failed: %s", exc)

    return results
