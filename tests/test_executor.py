import pytest
from sigil.executor import Executor
from sigil.classifier import ClassificationResult

def test_executor_interpolation():
    executor = Executor()
    # Mock screen resolution
    executor._screen_res = (1920, 1080)
    executor._sensitivity = 1.0
    
    # Test position interpolation
    res = ClassificationResult(
        gesture_name="test",
        gesture_type="instant",
        position=(0.5, 0.5),
        value=0.0
    )
    
    action = "hyprctl dispatch movecursor {x} {y}"
    interpolated = executor._interpolate_action(action, res)
    assert interpolated == "hyprctl dispatch movecursor 960 540"

def test_executor_interpolation_normalized():
    executor = Executor()
    executor._screen_res = (1920, 1080)
    executor._sensitivity = 1.0 # Use 1.0 for predictable test
    
    res = ClassificationResult(
        gesture_name="test",
        gesture_type="instant",
        position=(0.1, 0.2),
        value=0.0
    )
    
    action = "echo {nx} {ny}"
    interpolated = executor._interpolate_action(action, res)
    assert interpolated == "echo 0.1000 0.2000"

def test_executor_smoothing():
    executor = Executor()
    executor._screen_res = (1000, 1000)
    executor._smoothing_alpha = 0.5
    executor._sensitivity = 1.0
    
    # First frame
    res1 = ClassificationResult(gesture_name="t", gesture_type="i", position=(0.0, 0.0))
    int1 = executor._interpolate_action("{x} {y}", res1)
    assert int1 == "0 0"
    
    # Second frame (should be halfway)
    res2 = ClassificationResult(gesture_name="t", gesture_type="i", position=(1.0, 1.0))
    int2 = executor._interpolate_action("{x} {y}", res2)
    assert int2 == "500 500"
    
    # Third frame
    res3 = ClassificationResult(gesture_name="t", gesture_type="i", position=(1.0, 1.0))
    int3 = executor._interpolate_action("{x} {y}", res3)
    assert int3 == "750 750"
