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
        GestureMapping(
            name="continuous",
            type="instant",
            hand="right",
            condition={"pose": "Pointing_Up"},
            continuous=True,
        ),
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


def test_per_gesture_blanking_override():
    mappings = [
        GestureMapping(
            name="tap",
            type="instant",
            hand="right",
            condition={"pose": "Thumb_Index_Pinch"},
            confirm_frames=1,
            blanking_ms=0,
        )
    ]
    exec_cfg = ExecutionConfig(confirm_frames=3, blanking_ms=1500, confidence_threshold=0.75)
    classifier = GestureClassifier(mappings, execution=exec_cfg)

    with patch("sigil.classifier.monotonic_ms", return_value=5000):
        classifier.instant.classify = lambda _: [
            ClassificationResult(gesture_name="tap", gesture_type="instant", confidence=1.0, raw_label="Thumb_Index_Pinch")
        ]
        res = classifier.classify(FrameResult(timestamp_ms=5000))
        assert len(res) == 1
        assert res[0].gesture_name == "tap"
        # Per-gesture blanking_ms=0 should win
        assert classifier._blanking_until == 5000

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


def test_pinch_ratio_hysteresis_press_edge():
    import numpy as np

    def mk_hand(pinch_ratio_value: float) -> "HandResult":
        # Build landmarks so pinch_ratio == pinch distance (base is forced to 1.0).
        # pinch_ratio = dist(lm[4], lm[8]) / dist(lm[0], lm[5])
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0] = [0.0, 0.0, 0.0]  # wrist
        landmarks[5] = [1.0, 0.0, 0.0]  # index_mcp (base scale)
        landmarks[8] = [0.0, 0.0, 0.0]  # index_tip
        landmarks[4] = [pinch_ratio_value, 0.0, 0.0]  # thumb_tip
        from sigil.tracker import HandResult

        return HandResult(handedness="Right", landmarks=landmarks, score=0.9)

    mappings = [
        GestureMapping(
            name="pinch_tap",
            type="gradual",
            hand="right",
            condition={
                "feature": "pinch_ratio",
                "enter_below": 0.18,
                "exit_above": 0.25,
                "edge": "press",
            },
            action="noop",
            cooldown_ms=0,
        )
    ]

    classifier = GestureClassifier(mappings)

    # 1) Open pinch (no trigger)
    res1 = classifier.classify(FrameResult(timestamp_ms=1000, hands=[mk_hand(0.30)]))
    assert res1 == []

    # 2) Cross below enter_below => press trigger
    res2 = classifier.classify(FrameResult(timestamp_ms=1033, hands=[mk_hand(0.10)]))
    assert len(res2) == 1
    assert res2[0].gesture_name == "pinch_tap"

    # 3) Stay closed (no re-trigger)
    res3 = classifier.classify(FrameResult(timestamp_ms=1066, hands=[mk_hand(0.12)]))
    assert res3 == []

    # 4) Release above exit_above (edge is press only => no trigger)
    res4 = classifier.classify(FrameResult(timestamp_ms=1099, hands=[mk_hand(0.26)]))
    assert res4 == []

    # 5) Press again (cross below) => trigger again
    res5 = classifier.classify(FrameResult(timestamp_ms=1132, hands=[mk_hand(0.10)]))
    assert len(res5) == 1
    assert res5[0].gesture_name == "pinch_tap"


def test_tuple_delta_gating_min_delta_dy():
    import numpy as np
    from sigil.tracker import HandResult, FrameResult

    # For finger_deltas, palm_center uses wrist(0) and middle_mcp(9).
    def mk_hand(palm_y: float) -> HandResult:
        landmarks = np.zeros((21, 3), dtype=np.float32)
        landmarks[0] = [0.0, palm_y, 0.0]
        landmarks[9] = [0.0, palm_y, 0.0]
        return HandResult(handedness="Right", landmarks=landmarks, score=0.9)

    mappings = [
        GestureMapping(
            name="scroll_gate",
            type="gradual",
            hand="right",
            continuous=True,
            condition={"feature": "finger_deltas", "min_delta": 0.05, "delta_component": "dy"},
            action="noop",
            cooldown_ms=0,
        )
    ]

    classifier = GestureClassifier(mappings)

    # First frame: no prev landmarks => dy=0 => should not trigger
    res1 = classifier.classify(FrameResult(timestamp_ms=1000, hands=[mk_hand(0.0)]))
    assert res1 == []

    # Move palm by +0.10 => dy=0.10 => should trigger
    res2 = classifier.classify(FrameResult(timestamp_ms=1033, hands=[mk_hand(0.10)]))
    assert len(res2) == 1
    assert res2[0].gesture_name == "scroll_gate"

    # No further movement => dy=0 => should not trigger
    res3 = classifier.classify(FrameResult(timestamp_ms=1066, hands=[mk_hand(0.10)]))
    assert res3 == []
