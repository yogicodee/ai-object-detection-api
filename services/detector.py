from ultralytics import YOLO

model = YOLO("best.pt")

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
