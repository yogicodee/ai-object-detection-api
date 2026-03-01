from ultralytics import YOLO
import os
import gdown

MODEL_PATH = "best.pt"

FILE_ID = "1-1ZENMjtZGkQjCcEZ4q3My6VNbRwmz1w"
URL = f"https://drive.google.com/uc?id={FILE_ID}"

if not os.path.exists(MODEL_PATH):
    print("Downloading YOLO model from Google Drive...")
    gdown.download(URL, MODEL_PATH, quiet=False)
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