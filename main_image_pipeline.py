import sys
from detectors.yolov11_face_detector import YOLOv11FaceDetector
from embedder.insightface_embedder import ArcFaceEmbedder
from matcher.annoy_matcher import AnnoyFaceMatcher
import cv2

# ==== CONFIG ====
IMG_PATH = "media/photos/tom_holland3.jpg"
REGISTER_MODE = False        # True = register new faces, False = match faces
PERSON_NAME = "Tom Holland"  # Used only if REGISTER_MODE=True
MATCH_THRESHOLD = 1.5        # Euclidean distance threshold for matching
TOP_K = 3                    # Number of nearest neighbors to retrieve in matching
# =================

def main():
    # Initialize detector, embedder and matcher
    detector = YOLOv11FaceDetector()
    embedder = ArcFaceEmbedder()
    matcher = AnnoyFaceMatcher(threshold=MATCH_THRESHOLD)

    # Step 1: Detect faces in the image
    detections = detector.detect_faces(IMG_PATH)
    if not detections:
        print("❌ No faces found in the image.")
        sys.exit(0)

    # Step 2: Extract embeddings for all detected faces
    embeddings = embedder.get_embeddings_from_image(IMG_PATH, detections)
    if not embeddings:
        print("❌ Could not extract embeddings from detected faces.")
        sys.exit(0)

    # Step 3: Register or match depending on the mode
    if REGISTER_MODE:
        # Register all detected faces with the same PERSON_NAME (could be customized)
        for i, emb in enumerate(embeddings):
            matcher.register(PERSON_NAME, emb)
            print(f"✅ Registered face {i+1} for '{PERSON_NAME}'.")
    else:
        # Load the existing face database for matching
        try:
            matcher.load_for_matching()
        except FileNotFoundError:
            print("❌ No registered faces found. Please run in REGISTER_MODE first.")
            sys.exit(0)

        # Match each detected face embedding against the database
        for i, emb in enumerate(embeddings):
            results = matcher.match(emb, top_k=TOP_K)
            if results:
                print(f"✅ Match results for face {i+1}:")
                for name, dist in results:
                    print(f"  - {name} (distance: {dist:.4f})")
            else:
                print(f"❌ No match found within threshold for face {i+1}.")

if __name__ == "__main__":
    main()
