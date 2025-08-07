import cv2
from app.config import (
    VIDEO_PATH, IMAGE_PATH, FACE_OUTPUT_DIR, EMBEDDING_OUTPUT_DIR,
    SESSION_LOG_PATH, CURRENT_TIMESTAMP
)
from app.detection.detector import detect_faces
from app.detection.embedder import get_embedding
from app.utils.io_utils import save_face_image, save_embedding
from app.utils.time_utils import read_last_session, save_session_timestamp
from app.comparison.comparer import compare_embeddings
import os
import cv2

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"⚠️ Cannot read image: {image_path}")
        return
    run_detection(frame)

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Cannot open video: {video_path}")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        run_detection(frame)
        cv2.imshow("YOLO Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

def run_detection(frame):
    boxes = detect_faces(frame)
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        face_crop = frame[y1:y2, x1:x2]
        tag = f"{i}"

        face_path = os.path.join(FACE_OUTPUT_DIR, f"face_{tag}.jpg")
        save_face_image(face_path, face_crop)

        emb = get_embedding(face_crop)
        if emb is not None:
            emb_path = os.path.join(EMBEDDING_OUTPUT_DIR, f"embedding_{tag}.npy")
            save_embedding(emb_path, emb)
            print(f"✅ Saved embedding: {emb_path}")
        else:
            print("⚠️ Embedding failed.")

def main():
    if VIDEO_PATH:
        process_video(VIDEO_PATH)
    elif IMAGE_PATH:
        process_image(IMAGE_PATH)
    else:
        print("❌ No image or video input.")

    last_session = read_last_session(SESSION_LOG_PATH)
    if last_session and last_session != CURRENT_TIMESTAMP:
        prev_emb_dir = os.path.join("embeddings", last_session)
        prev_face_dir = os.path.join("detected_faces", last_session)
        if os.path.exists(prev_emb_dir) and os.path.exists(prev_face_dir):
            compare_embeddings(prev_emb_dir, prev_face_dir, EMBEDDING_OUTPUT_DIR, FACE_OUTPUT_DIR)
        else:
            print("⚠️ Missing previous session data.")
    else:
        print("ℹ️ First run or no previous session.")

    save_session_timestamp(SESSION_LOG_PATH, CURRENT_TIMESTAMP)

if __name__ == "__main__":
    main()
