# yolo_detector.py
from typing import List, Dict

from ultralytics import YOLO
import numpy as np

# Load YOLO once at import time
_yolo_model = YOLO("yolov8n.pt")


def run_yolo_on_image(image_bgr: np.ndarray, conf: float = 0.3) -> List[Dict]:
    """
    Run YOLOv8 on a BGR image and return a list of detections:
    [{"label": str, "score": float, "bbox": {"x":..., "y":..., "w":..., "h":...}}, ...]
    """
    results = _yolo_model.predict(source=image_bgr, imgsz=640, conf=conf, verbose=False)
    detections: List[Dict] = []

    for r in results:
        boxes = r.boxes
        names = _yolo_model.names
        for b in boxes:
            cls_id = int(b.cls[0].item())
            label = names.get(cls_id, str(cls_id))
            score = float(b.conf[0].item())
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            detections.append(
                {
                    "label": label,
                    "score": score,
                    "bbox": {
                        "x": x1,
                        "y": y1,
                        "w": x2 - x1,
                        "h": y2 - y1,
                    },
                }
            )

    return detections