from ultralytics import YOLO
import os

MODEL_PATH = "best.pt"

# pastikan file model ada
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("best.pt not found in project root")

# load model sekali saat server start
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