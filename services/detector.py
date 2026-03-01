from ultralytics import YOLO
import os
import urllib.request

MODEL_PATH = "best.pt"

MODEL_URL = "https://drive.google.com/uc?export=download&id=1-1ZENMjtZGkQjCcEZ4q3My6VNbRwmz1w"

# download model jika belum ada
if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model from Google Drive...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded")

model = YOLO(MODEL_PATH)


def detect_image(path):
    results = model(path)

    detections = []

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "label": model.names[cls],
                "confidence": round(conf, 3),
                "x": x1,
                "y": y1,
                "w": x2 - x1,
                "h": y2 - y1
            })

    return detections