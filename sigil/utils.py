"""Utility helpers for Sigil."""

from __future__ import annotations

import hashlib
import logging
import time
import urllib.request
from pathlib import Path

import numpy as np

from sigil.config import MODELS_DIR

logger = logging.getLogger(__name__)

# ── Model download URLs (MediaPipe official) ─────────────────────────────────
MODEL_URLS: dict[str, str] = {
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
    ),
    "gesture_recognizer.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task"
    ),
}


def ensure_model(filename: str) -> Path:
    """Download a MediaPipe model to MODELS_DIR on first run (§7)."""
    dest = MODELS_DIR / filename
    if dest.exists():
        return dest

    url = MODEL_URLS.get(filename)
    if url is None:
        raise FileNotFoundError(
            f"No download URL for '{filename}'. Place it manually in {MODELS_DIR}"
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s → %s …", filename, dest)
    urllib.request.urlretrieve(url, dest)  # noqa: S310 – trusted Google domain
    logger.info("Download complete: %s (%d bytes)", filename, dest.stat().st_size)
    return dest


# ── Landmark math ────────────────────────────────────────────────────────────
def landmark_to_array(landmarks: list) -> np.ndarray:  # type: ignore[type-arg]
    """Convert MediaPipe NormalizedLandmarkList to (21, 3) float32 array."""
    return np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two points."""
    return float(np.linalg.norm(a - b))


def thumb_index_distance(landmarks: np.ndarray) -> float:
    """Normalised thumb-tip (4) to index-tip (8) distance."""
    return euclidean(landmarks[4], landmarks[8])


def palm_center(landmarks: np.ndarray) -> np.ndarray:
    """Average of wrist (0) and middle-finger MCP (9) as palm center."""
    return (landmarks[0] + landmarks[9]) / 2.0  # type: ignore[no-any-return]


def finger_curl_angles(landmarks: np.ndarray) -> np.ndarray:
    """Curl angle proxy for each finger (5 values, 0=extended, 1=fully curled)."""
    wrist = landmarks[0]
    finger_tips = [4, 8, 12, 16, 20]
    finger_mcps = [1, 5, 9, 13, 17]
    curls = np.zeros(5, dtype=np.float32)
    for i, (tip, mcp) in enumerate(zip(finger_tips, finger_mcps, strict=False)):
        mcp_to_wrist = euclidean(landmarks[mcp], wrist) + 1e-7
        tip_to_mcp = euclidean(landmarks[tip], landmarks[mcp])
        ratio = tip_to_mcp / mcp_to_wrist
        curls[i] = max(0.0, min(1.0, 1.0 - ratio))
    return curls


def count_extended_fingers(landmarks: np.ndarray, handedness: str = "Right") -> int:
    """Returns number of fingers extended (0-5). Uses balanced joint detection."""
    wrist = landmarks[0]
    finger_tips = [8, 12, 16, 20]
    finger_pips = [6, 10, 14, 18]

    extended_count = 0

    for tip, pip in zip(finger_tips, finger_pips, strict=False):
        tip_dist = euclidean(landmarks[tip], wrist)
        pip_dist = euclidean(landmarks[pip], wrist)
        if tip_dist > pip_dist * 1.15:
            extended_count += 1

    thumb_tip = landmarks[4]
    thumb_ip = landmarks[3]
    pinky_mcp = landmarks[17]

    tip_to_pinky = euclidean(thumb_tip, pinky_mcp)
    ip_to_pinky = euclidean(thumb_ip, pinky_mcp)

    if tip_to_pinky > ip_to_pinky * 1.1:
        extended_count += 1

    return extended_count


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """Phase 2: Normalize hand to standard virtual space.

    - Translate: Subtract wrist from all points (wrist becomes [0,0,0])
    - Scale: Divide by distance from wrist to middle finger MCP (9)

    Returns (21, 3) normalized array.
    """
    wrist = landmarks[0]
    translated = landmarks - wrist

    mcp_distance = euclidean(translated[9], np.array([0.0, 0.0, 0.0])) + 1e-7
    scaled = translated / mcp_distance

    return scaled


def compute_joint_angles(landmarks: np.ndarray) -> np.ndarray:
    """Phase 3.4: Calculate knuckle angles at base and middle joints.

    Returns 10 angles (5 fingers × 2 joints each) in degrees.
    Formula: cos(theta) = (BA · BC) / (|BA| * |BC|)
    """
    angles = np.zeros(10, dtype=np.float32)

    joint_sets = [
        (1, 2, 3),  # thumb: CMC-IP-PIP
        (5, 6, 7),  # index: MCP-PIP-DIP
        (9, 10, 11),  # middle: MCP-PIP-DIP
        (13, 14, 15),  # ring: MCP-PIP-DIP
        (17, 18, 19),  # pinky: MCP-PIP-DIP
    ]

    for finger_idx, (a, b, c) in enumerate(joint_sets):
        for joint_idx, (i, j, k) in enumerate([(a, b, c), (b, c, 20)]):
            vec_ba = landmarks[i] - landmarks[j]
            vec_bc = landmarks[k] - landmarks[j]

            norm_ba = np.linalg.norm(vec_ba) + 1e-7
            norm_bc = np.linalg.norm(vec_bc) + 1e-7

            cos_theta = np.dot(vec_ba, vec_bc) / (norm_ba * norm_bc)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = np.arccos(cos_theta)

            idx = finger_idx * 2 + joint_idx
            angles[idx] = np.degrees(theta)

    return angles


def finger_extension_states(landmarks: np.ndarray) -> np.ndarray:
    """Phase 3.2: Binary finger extension states (1=extended, 0=curled).

    Compare distance from fingertip to wrist vs finger base to wrist.
    """
    states = np.zeros(5, dtype=np.float32)
    wrist = landmarks[0]

    finger_tips = [4, 8, 12, 16, 20]
    finger_bases = [1, 5, 9, 13, 17]

    for i, (tip, base) in enumerate(zip(finger_tips, finger_bases, strict=False)):
        tip_dist = euclidean(landmarks[tip], wrist)
        base_dist = euclidean(landmarks[base], wrist)

        if tip_dist > base_dist:
            states[i] = 1.0

    return states


def tip_to_wrist_distances(landmarks: np.ndarray) -> np.ndarray:
    """Phase 3.3: Distance from each fingertip to wrist."""
    wrist = landmarks[0]
    distances = np.zeros(5, dtype=np.float32)

    finger_tips = [4, 8, 12, 16, 20]
    for i, tip in enumerate(finger_tips):
        distances[i] = euclidean(landmarks[tip], wrist)

    return distances


def fingertip_adjacency(landmarks: np.ndarray) -> np.ndarray:
    """Phase 3.5: Distances between fingertips.

    Returns 4 distances: index-middle, middle-ring, ring-pinky, thumb-index.
    """
    distances = np.zeros(4, dtype=np.float32)

    pairs = [(8, 12), (12, 16), (16, 20), (4, 8)]
    for i, (a, b) in enumerate(pairs):
        distances[i] = euclidean(landmarks[a], landmarks[b])

    return distances


def tip_to_palm_center_distances(landmarks: np.ndarray) -> np.ndarray:
    """Phase 3.6: Distance from each fingertip to palm center.

    Palm center = average of wrist (0) and all 4 finger MCPs.
    """
    wrist = landmarks[0]
    mcps = [1, 5, 9, 13, 17]
    palm_center = (wrist + np.mean(landmarks[mcps], axis=0)) / 2.0

    distances = np.zeros(5, dtype=np.float32)
    finger_tips = [4, 8, 12, 16, 20]

    for i, tip in enumerate(finger_tips):
        distances[i] = euclidean(landmarks[tip], palm_center)

    return distances


def index_direction(landmarks: np.ndarray) -> float:
    """Phase 3.7: Index finger angle relative to vertical.

    Returns angle in degrees of index finger (tip to base) from vertical.
    """
    wrist = landmarks[0]
    index_base = landmarks[5]
    index_tip = landmarks[8]

    vec_finger = index_tip - index_base
    vec_vertical = np.array([0.0, -1.0, 0.0])

    norm_finger = np.linalg.norm(vec_finger) + 1e-7
    cos_angle = np.dot(vec_finger, vec_vertical) / norm_finger
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    return np.degrees(np.arccos(cos_angle))


def thumb_spread(landmarks: np.ndarray) -> np.ndarray:
    """Phase 3.8: Thumb tip offset relative to index finger base.

    Returns (x, y, z) offset from index MCP (5) to thumb tip (4).
    """
    index_base = landmarks[5]
    thumb_tip = landmarks[4]

    return thumb_tip - index_base


# ── Data Augmentation ─────────────────────────────────────────────────────────
class LandmarkAugmenter:
    """Creates realistic variations of raw 3D hand landmarks for data augmentation.

    Augmentation strategies:
    - 3D Rotation: Randomly rotate hand ±20 degrees around X, Y, Z axes
    - Gaussian Noise: Add tracking jitter (±0.005) to simulate MediaPipe inaccuracy
    - Scale Variation: Apply ±10% scaling to simulate depth changes
    - Translation: Apply ±2% position shift to simulate hand movement
    """

    @staticmethod
    def _rotate_points_around_wrist(points: np.ndarray, max_angle: float = 20.0) -> np.ndarray:
        """Rotates the hand around the wrist pivot point."""
        wrist = points[0].copy()
        centered = points - wrist

        angles = np.random.uniform(-max_angle, max_angle, 3)
        angles_rad = np.radians(angles)

        cos_x, sin_x = np.cos(angles_rad[0]), np.sin(angles_rad[0])
        cos_y, sin_y = np.cos(angles_rad[1]), np.sin(angles_rad[1])
        cos_z, sin_z = np.cos(angles_rad[2]), np.sin(angles_rad[2])

        Rx = np.array([[1, 0, 0], [0, cos_x, -sin_x], [0, sin_x, cos_x]])
        Ry = np.array([[cos_y, 0, sin_y], [0, 1, 0], [-sin_y, 0, cos_y]])
        Rz = np.array([[cos_z, -sin_z, 0], [sin_z, cos_z, 0], [0, 0, 1]])

        R = Rz @ Ry @ Rx
        rotated = centered @ R.T

        return rotated + wrist

    @staticmethod
    def _add_jitter(points: np.ndarray, noise_level: float = 0.005) -> np.ndarray:
        """Adds Gaussian noise to simulate MediaPipe tracking jitter."""
        noise = np.random.normal(0, noise_level, points.shape)
        return points + noise

    @staticmethod
    def _apply_scale(points: np.ndarray, scale_range: tuple[float, float] = (0.9, 1.1)) -> np.ndarray:
        """Apply random scaling to simulate depth variation."""
        wrist = points[0].copy()
        centered = points - wrist
        scale = np.random.uniform(*scale_range)
        scaled = centered * scale
        return scaled + wrist

    @staticmethod
    def _apply_translation(points: np.ndarray, translation_range: float = 0.02) -> np.ndarray:
        """Apply random translation to simulate hand movement."""
        translation = np.random.uniform(-translation_range, translation_range, 3)
        return points + translation

    @staticmethod
    def augment(raw_landmarks: np.ndarray, multiplier: int = 12) -> list[np.ndarray]:
        """Generate augmented versions of raw 21-point hand landmarks.

        Args:
            raw_landmarks: (21, 3) array of raw x,y,z coordinates
            multiplier: Number of augmented samples to generate (including original)

        Returns:
            List of (21, 3) arrays, starting with original followed by augmented versions
        """
        augmented = [raw_landmarks.copy()]

        for i in range(multiplier - 1):
            sample = raw_landmarks.copy()

            # Rotate 80% of samples
            if np.random.random() > 0.2:
                sample = LandmarkAugmenter._rotate_points_around_wrist(sample, max_angle=20.0)

            # Scale 60% of samples
            if np.random.random() > 0.4:
                sample = LandmarkAugmenter._apply_scale(sample, scale_range=(0.9, 1.1))

            # Translate 40% of samples
            if np.random.random() > 0.6:
                sample = LandmarkAugmenter._apply_translation(sample, translation_range=0.02)

            # Always add jitter
            sample = LandmarkAugmenter._add_jitter(sample, noise_level=0.005)

            augmented.append(sample)

        return augmented


def palm_orientation(landmarks: np.ndarray) -> np.ndarray:
    """Calculate palm orientation vector (3 values).
    
    Uses normal vector to plane defined by wrist, index MCP, and pinky MCP.
    """
    wrist = landmarks[0]
    index_mcp = landmarks[5]
    pinky_mcp = landmarks[17]
    
    vec1 = index_mcp - wrist
    vec2 = pinky_mcp - wrist
    
    normal = np.cross(vec1, vec2)
    norm = np.linalg.norm(normal) + 1e-7
    normalized_normal = normal / norm
    
    return normalized_normal


def finger_spread(landmarks: np.ndarray) -> float:
    """Calculate overall finger spread/splay metric.
    
    Returns average angle between adjacent fingers.
    """
    wrist = landmarks[0]
    finger_tips = [4, 8, 12, 16, 20]
    
    angles = []
    for i in range(len(finger_tips) - 1):
        vec1 = landmarks[finger_tips[i]] - wrist
        vec2 = landmarks[finger_tips[i + 1]] - wrist
        
        norm1 = np.linalg.norm(vec1) + 1e-7
        norm2 = np.linalg.norm(vec2) + 1e-7
        
        cos_angle = np.dot(vec1, vec2) / (norm1 * norm2)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    return float(np.mean(angles))


def hand_compactness(landmarks: np.ndarray) -> float:
    """Measure how compact/open the hand is.
    
    Returns ratio of bounding box volume to palm size.
    """
    min_coords = np.min(landmarks, axis=0)
    max_coords = np.max(landmarks, axis=0)
    bbox_size = np.prod(max_coords - min_coords)
    
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    palm_size = euclidean(wrist, middle_mcp) ** 3 + 1e-7
    
    return float(bbox_size / palm_size)


def extract_96_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract all 101 geometric features from normalized landmarks.

    Pipeline:
    1. Normalized positions (63): xyz of 21 normalized points
    2. Finger extension states (5): binary curled/extended
    3. Tip-to-wrist distances (5): how far each finger reaches
    4. Joint angles (10): knuckle bend angles
    5. Fingertip adjacency (4): distances between fingertips
    6. Tip-to-palm-center distances (5)
    7. Index direction (1)
    8. Thumb spread (3)
    9. Palm orientation (3): normal vector to palm plane
    10. Finger spread (1): overall splay metric
    11. Hand compactness (1): open/closed hand metric
    
    Total: 101 features (63+5+5+10+4+5+1+3+3+1+1)
    """
    norm = normalize_landmarks(landmarks)

    features = []

    features.extend(norm.flatten())

    features.extend(finger_extension_states(norm))

    features.extend(tip_to_wrist_distances(norm))

    features.extend(compute_joint_angles(norm))

    features.extend(fingertip_adjacency(norm))

    features.extend(tip_to_palm_center_distances(norm))

    features.append(index_direction(norm))

    features.extend(thumb_spread(norm))
    
    features.extend(palm_orientation(norm))
    
    features.append(finger_spread(norm))
    
    features.append(hand_compactness(norm))

    return np.array(features, dtype=np.float32)


# ── Timing helpers ───────────────────────────────────────────────────────────
def monotonic_ms() -> int:
    """Current monotonic time in milliseconds."""
    return int(time.monotonic() * 1000)


def file_hash(path: Path) -> str:
    """SHA-256 hex digest of a file (for cache checks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
