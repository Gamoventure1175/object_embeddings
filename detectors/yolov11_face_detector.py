# detectors/yolov11_face_detector.py

from ultralytics import YOLO
import cv2
class YOLOv11FaceDetector:
    def __init__(self, model_path="models/yolov11n-face.pt", conf_thresh=0.3):
        """
        model_path: path to YOLOv11 face detection model weights
        conf_thresh: confidence threshold for detections
        """
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect_faces(self, image_path):
        """
        Runs face detection on an image.
        Returns: list of (x1, y1, x2, y2, conf) for each face
        """
        results = self.model(image_path, conf=self.conf_thresh)[0]
        detections = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])
            detections.append((float(x1), float(y1), float(x2), float(y2), conf))
        
        return detections

    def visualize(self, image_path, detections, save_path="output.jpg"):
        """
        Draws bounding boxes on image and saves it.
        """
        img = cv2.imread(image_path)
        for x1, y1, x2, y2, conf in detections:
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(img, f"{conf:.2f}", (int(x1), int(y1)-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.imwrite(save_path, img)
        print(f"Saved output to {save_path}")


if __name__ == "__main__":
    detector = YOLOv11FaceDetector("models/yolov11n-face.pt", conf_thresh=0.3)
    test_img = "media/images/bradpitt.jpg"  # path to any face image
    detections = detector.detect_faces(test_img)
    print("Detections:", detections)
    detector.visualize(test_img, detections, "detected_faces.jpg")
