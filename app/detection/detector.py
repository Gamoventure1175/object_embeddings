import cv2
from ultralytics import YOLO
from app.config import MODEL_PATH

model = YOLO(MODEL_PATH)

def detect_faces(frame):
    results = model(frame, verbose=False)[0]
    boxes = []
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        if conf >= 0.5:
            boxes.append((x1, y1, x2, y2))
    return boxes
