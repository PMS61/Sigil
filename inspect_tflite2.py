import zipfile
import io

model_path = "/home/prathamesh/.local/share/sigil/models/custom_gesture.tflite"

with zipfile.ZipFile(model_path, 'r') as z:
    inner = z.read("hand_gesture_recognizer.task")

with zipfile.ZipFile(io.BytesIO(inner), 'r') as inner_z:
    for name in inner_z.namelist():
        print("Inner file:", name)
        if name.endswith('.txt'):
            print(f"--- {name} ---")
            print(inner_z.read(name).decode('utf-8'))
            print("------")
