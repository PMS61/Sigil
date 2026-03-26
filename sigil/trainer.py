"""Model Training & Management (§5.4).

Provides one-click retrain commands:
  - Instant gestures: MediaPipe Model Maker → custom_gesture.tflite
  - Dynamic gestures: scikit-learn MLP/KNN on extracted features.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from sigil.config import MODELS_DIR, RECORDINGS_DIR

logger = logging.getLogger(__name__)


# ═════════════════════════════════════════════════════════════════════════════
# 1.  Instant gesture training via MediaPipe Model Maker (§5.4)
# ═════════════════════════════════════════════════════════════════════════════
def train_instant(
    data_dir: Path | None = None,
    output_path: Path | None = None,
    epochs: int = 20,
    batch_size: int = 32,
) -> Path:
    """Train a custom gesture recognizer .tflite from recorded images.

    Expects *data_dir* to contain one sub-folder per class, each with .jpg samples
    (the standard Model Maker image-classification layout).

    Raises ImportError if ``mediapipe-model-maker`` is not installed.
    """
    try:
        from mediapipe_model_maker import gesture_recognizer as gr
    except ImportError as exc:
        raise ImportError(
            "mediapipe-model-maker is required for training instant gestures. "
            "Install with: pip install 'sigil[train]'"
        ) from exc

    data_dir = data_dir or RECORDINGS_DIR
    out = output_path or (MODELS_DIR / "custom_gesture.tflite")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset from %s …", data_dir)
    data = gr.Dataset.from_folder(str(data_dir))

    # 80/20 train/test split
    train_data, test_data = data.split(0.8)

    logger.info(
        "Training instant gesture model (%d epochs, batch=%d) …", epochs, batch_size
    )

    hparams = gr.HParams(
        export_dir=str(out.parent),
        epochs=epochs,
        batch_size=batch_size,
    )
    model = gr.GestureRecognizer.create(
        train_data=train_data,
        validation_data=test_data,
        hparams=hparams,
    )

    # Evaluate
    loss, accuracy = model.evaluate(test_data)
    logger.info("Instant model – loss=%.4f  accuracy=%.4f", loss, accuracy)

    # Export
    model.export_model(str(out))
    logger.info("Exported instant model → %s", out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 2.  Dynamic (gradual/sequential) gesture training via scikit-learn (§5.4)
# ═════════════════════════════════════════════════════════════════════════════
def _load_feature_dataset(
    data_dir: Path, class_dirs: list[str] | None = None
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Load features.csv or .json samples from per-class sub-folders.

    Returns (X, y, class_names).
    """
    if class_dirs is None:
        class_dirs = [d.name for d in sorted(data_dir.iterdir()) if d.is_dir()]

    all_x: list[list[float]] = []
    all_y: list[int] = []

    for idx, cls in enumerate(class_dirs):
        cls_dir = data_dir / cls
        # Prefer CSV
        csv_path = cls_dir / "features.csv"
        if csv_path.exists():
            import csv as csv_mod

            with open(csv_path) as fh:
                reader = csv_mod.DictReader(fh)
                for row in reader:
                    # Skip non-numeric columns
                    feats = [
                        float(v)
                        for k, v in sorted(row.items())
                        if k not in ("timestamp_ms", "handedness")
                    ]
                    all_x.append(feats)
                    all_y.append(idx)
        else:
            # Fall back to individual JSONs
            for jf in sorted(cls_dir.glob("*.json")):
                with open(jf) as fh:
                    data: dict[str, Any] = json.load(fh)
                if isinstance(data, list):
                    # Sequential: flatten
                    for frame in data:
                        feats = [
                            float(v)
                            for k, v in sorted(frame.items())
                            if k not in ("timestamp_ms", "handedness", "frame_index")
                        ]
                        all_x.append(feats)
                        all_y.append(idx)
                else:
                    feats = [
                        float(v)
                        for k, v in sorted(data.items())
                        if k not in ("timestamp_ms", "handedness")
                    ]
                    all_x.append(feats)
                    all_y.append(idx)

    return np.array(all_x, dtype=np.float32), np.array(all_y), class_dirs


def train_dynamic(
    data_dir: Path | None = None,
    output_path: Path | None = None,
    model_type: str = "mlp",
) -> Path:
    """Train a lightweight scikit-learn classifier on recorded landmark features.

    Supported *model_type*: ``mlp`` (default), ``knn``.

    Raises ImportError if scikit-learn is not installed.
    """
    try:
        import joblib
        from sklearn.metrics import classification_report
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise ImportError(
            "scikit-learn is required for training dynamic gestures. "
            "Install with: pip install 'sigil[train]'"
        ) from exc

    data_dir = data_dir or RECORDINGS_DIR
    out = output_path or (MODELS_DIR / f"dynamic_{model_type}.pkl")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    features, labels, class_names = _load_feature_dataset(data_dir)
    if len(features) < 10:
        raise ValueError(f"Not enough samples ({len(features)}). Need ≥ 10.")

    logger.info(
        "Loaded %d samples across %d classes: %s",
        len(features),
        len(class_names),
        class_names,
    )

    feat_train, feat_test, y_train, y_test = train_test_split(
        features, labels, test_size=0.2, random_state=42, stratify=labels
    )

    scaler = StandardScaler()
    feat_train = scaler.fit_transform(feat_train)
    feat_test = scaler.transform(feat_test)

    if model_type == "knn":
        clf = KNeighborsClassifier(n_neighbors=5, weights="distance")
    else:
        clf = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            early_stopping=True,
            random_state=42,
        )

    logger.info("Training %s classifier …", model_type.upper())
    clf.fit(feat_train, y_train)

    accuracy = clf.score(feat_test, y_test)
    logger.info("Dynamic model accuracy: %.4f", accuracy)
    report = classification_report(
        y_test, clf.predict(feat_test), target_names=class_names
    )
    logger.info("Classification report:\n%s", report)

    # Save model + scaler + class names
    artifact = {
        "model": clf,
        "scaler": scaler,
        "class_names": class_names,
    }
    joblib.dump(artifact, out)
    logger.info("Exported dynamic model → %s", out)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# 3.  One-click retrain (§5.4)
# ═════════════════════════════════════════════════════════════════════════════
def retrain(
    instant: bool = True,
    dynamic: bool = True,
    data_dir: Path | None = None,
    epochs: int = 20,
    batch_size: int = 32,
) -> dict[str, Path]:
    """Retrain all models from current recordings.

    Returns dict of ``{model_name: output_path}`` for successfully trained models.
    """
    results: dict[str, Path] = {}
    data_dir = data_dir or RECORDINGS_DIR

    if instant:
        try:
            results["instant"] = train_instant(
                data_dir=data_dir, epochs=epochs, batch_size=batch_size
            )
        except Exception as exc:
            logger.error("Instant training failed: %s", exc)

    if dynamic:
        for mtype in ("mlp", "knn"):
            try:
                results[f"dynamic_{mtype}"] = train_dynamic(
                    data_dir=data_dir, model_type=mtype
                )
            except Exception as exc:
                logger.error("Dynamic (%s) training failed: %s", mtype, exc)

    return results
