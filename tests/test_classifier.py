import pytest
from sigil.classifier import GestureClassifier, ClassificationResult
from sigil.config import GestureMapping, ExecutionConfig
from sigil.tracker import FrameResult

from unittest.mock import patch
import pytest
from sigil.classifier import GestureClassifier, ClassificationResult
from sigil.config import GestureMapping, ExecutionConfig
from sigil.tracker import FrameResult

def test_continuous_gesture_bypasses_blanking():
    # Setup mappings
    mappings = [
        GestureMapping(name="discrete", type="instant", hand="right", condition={"pose": "Closed_Fist"}),
        GestureMapping(name="continuous", type="instant", hand="right", condition={"pose": "Pointing_Up"}, continuous=True)
    ]
    
    # Execution config with long blanking
    exec_cfg = ExecutionConfig(confirm_frames=1, blanking_ms=1000)
    
    classifier = GestureClassifier(mappings, execution=exec_cfg)
    
    # Trigger discrete action
    with patch("sigil.classifier.monotonic_ms", return_value=1000):
        classifier.instant.classify = lambda _: [
            ClassificationResult(gesture_name="discrete", gesture_type="instant", confidence=1.0, raw_label="Closed_Fist")
        ]
        
        res = classifier.classify(FrameResult(timestamp_ms=1000))
        assert len(res) == 1
        assert res[0].gesture_name == "discrete"
        assert classifier._blanking_until == 1000 + 1000
    
    # Next frame, blanking is active
    with patch("sigil.classifier.monotonic_ms", return_value=1100):
        # Discrete should be suppressed
        classifier.instant.classify = lambda _: [
            ClassificationResult(gesture_name="discrete", gesture_type="instant", confidence=1.0, raw_label="Closed_Fist")
        ]
        res2 = classifier.classify(FrameResult(timestamp_ms=1100))
        assert len(res2) == 0
        
        # Continuous should NOT be suppressed
        classifier.instant.classify = lambda _: [
            ClassificationResult(gesture_name="continuous", gesture_type="instant", confidence=1.0, raw_label="Pointing_Up")
        ]
        res3 = classifier.classify(FrameResult(timestamp_ms=1200))
        assert len(res3) == 1
        assert res3[0].gesture_name == "continuous"

def test_gradual_pointer_position():
    mappings = [
        GestureMapping(name="cursor", type="gradual", hand="right", condition={"feature": "pointer_position"}, continuous=True)
    ]
    classifier = GestureClassifier(mappings)
    
    # Mock HandResult with landmarks
    import numpy as np
    from sigil.tracker import HandResult, FrameResult
    
    # Landmark 8 is index finger tip
    landmarks = np.zeros((21, 3))
    landmarks[8] = [0.1, 0.2, 0.3]
    
    hand = HandResult(handedness="Right", landmarks=landmarks, score=0.9)
    frame = FrameResult(timestamp_ms=1000, hands=[hand])
    
    res = classifier.classify(frame)
    assert len(res) == 1
    assert res[0].gesture_name == "cursor"
    assert res[0].position == (0.1, 0.2)
