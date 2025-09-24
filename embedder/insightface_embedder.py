import cv2
import numpy as np
from insightface.model_zoo import ArcFaceONNX
from detectors.yolov11_face_detector import YOLOv11FaceDetector

class ArcFaceEmbedder:
    def __init__(self, model_path=None):
        if model_path is None:
            import os
            home_dir = os.path.expanduser('~')
            model_path = os.path.join(
                home_dir, '.insightface', 'models', 'buffalo_l', 'w600k_r50.onnx'
            )
        self.rec_model = ArcFaceONNX(model_path)
        self.rec_model.prepare(ctx_id=-1)  # CPU only

    def _normalize_embedding(self, emb):
        """Normalize embedding to unit length (L2 norm = 1)."""
        norm = np.linalg.norm(emb)
        if norm == 0:
            return emb
        return emb / norm

    def get_embedding_from_crop(self, face_crop, tta_flip=True):
        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (112, 112))

        emb = self.rec_model.get_feat(face_rgb).flatten()

        if tta_flip:
            face_flip = cv2.flip(face_rgb, 1)
            emb_flip = self.rec_model.get_feat(face_flip).flatten()
            emb = 0.5 * (emb + emb_flip)

        emb = self._normalize_embedding(emb)
        return emb


    def get_embeddings_from_image(self, image_path, detections):
        img = cv2.imread(image_path)
        embeddings = []
        for det in detections:
            x1, y1, x2, y2 = map(int, det[:4])
            face_crop = img[y1:y2, x1:x2] # type:ignore
            emb = self.get_embedding_from_crop(face_crop)
            embeddings.append(emb)
        return embeddings


if __name__ == "__main__":
    import inspect, insightface
    from insightface.model_zoo import ArcFaceONNX
    print(inspect.getsource(ArcFaceONNX))   # if source is available
    print([m for m in dir(ArcFaceONNX) if not m.startswith('_')])
    

    img_path = "media/images/bradpitt.jpg"

    # Step 1: Detect faces with YOLO
    detector = YOLOv11FaceDetector()
    detections = detector.detect_faces(img_path)

    # Step 2: Get embeddings using ArcFace
    embedder = ArcFaceEmbedder()
    embeddings = embedder.get_embeddings_from_image(img_path, detections)

    print("Found", len(embeddings), "faces")
    for i, emb in enumerate(embeddings):
        print(f"Face {i+1} embedding (normalized, norm={np.linalg.norm(emb):.4f}):", emb)
