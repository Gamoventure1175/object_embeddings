import sys
from detectors.yolov11_face_detector import YOLOv11FaceDetector
from embedder.insightface_embedder import ArcFaceEmbedder
from matcher.annoy_matcher import AnnoyFaceMatcher

# ==== CONFIG ====
IMG_PATH = "media/photos/tom_holland4.png"
REGISTER_MODE = False          # True = Register faces, False = Match faces
PERSON_NAME = "Tom Holland"    # Used only in REGISTER_MODE
MATCH_THRESHOLD = 1.5          # Max distance for valid match
TOP_K = 10                    # Number of nearest neighbors to return in matching
# =================

def main():
    detector = YOLOv11FaceDetector()
    embedder = ArcFaceEmbedder()
    matcher = AnnoyFaceMatcher(threshold=MATCH_THRESHOLD)

    # Step 1: Detect faces
    detections = detector.detect_faces(IMG_PATH)
    if not detections:
        print("❌ No faces found in the image.")
        sys.exit(0)

    # Step 2: Get embeddings
    embeddings = embedder.get_embeddings_from_image(IMG_PATH, detections)
    if not embeddings:
        print("❌ Could not extract any embeddings.")
        sys.exit(0)

    if REGISTER_MODE:
        # Load existing data for update (if any)
        try:
            matcher.load_for_matching()
        except FileNotFoundError:
            print("⚠️ No previous registered faces found. Creating new database.")

        # Register all detected faces with the given PERSON_NAME
        for i, emb in enumerate(embeddings):
            # If you want different names for each face, extend logic here
            matcher.register(PERSON_NAME, emb)
            print(f"✅ Registered face {i+1} for '{PERSON_NAME}'")
    else:
        # Matching mode - load index and metadata
        try:
            matcher.load_for_matching()
        except FileNotFoundError:
            print("❌ No registered faces found. Please run in REGISTER_MODE first.")
            sys.exit(0)

        # Match all detected faces
        for i, emb in enumerate(embeddings):
            results = matcher.match(emb, top_k=TOP_K)
            if results:
                print(f"✅ Match results for face {i+1}:")
                for name, dist in results:
                    print(f"  {name} with distance {dist:.4f}")
            else:
                print(f"❌ No match found for face {i+1} within threshold.")

if __name__ == "__main__":
    main()
