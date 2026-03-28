import numpy as np
import mediapipe as mp
import cv2

# maybe it's uint8 vs something else, or contiguous vs not
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if ret:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    
    BaseOptions = mp.tasks.BaseOptions
    HandLandmarker = mp.tasks.vision.HandLandmarker
    HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path='/home/prathamesh/.local/share/sigil/models/hand_landmarker.task'),
        running_mode=VisionRunningMode.VIDEO)
    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            landmarker.detect_for_video(img, 100)
            print("Detect success")
    except Exception as e:
        print(f"Error: {e}")
cap.release()
