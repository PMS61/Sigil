import zipfile
import json
import os

model_path = "/home/prathamesh/.local/share/sigil/models/custom_gesture.tflite"
if not os.path.exists(model_path):
    print("Model not found")
    exit(1)

# The task file is a zip format
with zipfile.ZipFile(model_path, 'r') as z:
    for name in z.namelist():
        print("File inside model:", name)
        if name.endswith('.txt'):
            print(f"--- {name} ---")
            print(z.read(name).decode('utf-8'))
            print("------")
