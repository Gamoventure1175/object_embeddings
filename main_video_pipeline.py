# main_video_pipeline.py
import cv2
import numpy as np
from config import CONFIG
from detectors.yolov11_face_detector import YOLOv11FaceDetector
from embedder.insightface_embedder import ArcFaceEmbedder
from matcher.annoy_matcher import AnnoyFaceMatcher
from trackers.bytetrack_tracker import ByteTrackWrapper
from trackers.track_manager import TrackManager
from utils.rec_logger import RecognitionLogger

if not hasattr(np, "float"):  # Compatibility for numpy 2.0+
    np.float = float

def main():
    video_path = CONFIG["VIDEO_PATH"]
    display = CONFIG["DISPLAY"]
    output_path = CONFIG["OUTPUT_PATH"]

    # --- Initialize modules ---
    detector = YOLOv11FaceDetector()
    embedder = ArcFaceEmbedder()
    matcher = AnnoyFaceMatcher(
        vector_size=CONFIG["VECTOR_SIZE"],
        index_path=CONFIG["INDEX_PATH"],
        meta_path=CONFIG["META_PATH"],
        emb_path=CONFIG["EMBEDDINGS_PATH"],
        threshold=CONFIG["MATCH_THRESHOLD"],
        n_trees=CONFIG["ANNOY_TREES"]
    )
    tracker = ByteTrackWrapper()
    track_manager = TrackManager()
    logger = RecognitionLogger(CONFIG["CSV_LOG_PATH"]) if CONFIG["LOG_TO_CSV"] else None

    # Load face DB if exists
    try:
        matcher.load_for_matching()
        print("[main] Loaded face DB for matching.")
    except FileNotFoundError:
        print("[main] No face DB found — starting empty.")
    except Exception as e:
        print(f"[main] Warning loading DB: {e}")

    # --- Video IO setup ---
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[main] Failed to open video: {video_path}")
        return

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_no = 0
    print("[main] Starting processing... press ESC to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_no += 1
        frame_h, frame_w = frame.shape[:2]

        # Step 1: Detect faces
        detections = detector.detect_faces(frame)
        if detections:
            scale = min(CONFIG["MODEL_INPUT_SIZE"][0] / float(frame_h),
                        CONFIG["MODEL_INPUT_SIZE"][1] / float(frame_w))
            detections_model = [
                (x1 * scale, y1 * scale, x2 * scale, y2 * scale, conf)
                for x1, y1, x2, y2, conf in detections
            ]
        else:
            detections_model = []

        # Step 2: Track faces
        tracks = tracker.update(
            detections_model,
            img_info=(frame_h, frame_w),
            img_size=CONFIG["MODEL_INPUT_SIZE"]
        )

        # Step 3: Recognition per track
        for t in tracks:
            x1, y1, x2, y2 = map(int, t["bbox"])
            track_id = int(t["track_id"])

            # Crop and embed
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame_w - 1, x2), min(frame_h - 1, y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            face_crop = frame[y1c:y2c, x1c:x2c]
            if face_crop.size == 0:
                continue

            try:
                emb = embedder.get_embedding_from_crop(face_crop)
            except Exception as e:
                print(f"[main] Embedder failure: {e}")
                continue
            if emb is None:
                continue

            # Match with DB
            results = matcher.match(emb, top_k=CONFIG["TOP_K"]) or []
            if results:
                name, dist = results[0]
                registered = False
            else:
                if CONFIG["AUTO_REGISTER_UNKNOWN"]:
                    name = f"person_{matcher.next_id}"
                    matcher.register(name, emb)
                    registered = True
                    dist = None
                else:
                    name = "Unknown"
                    dist = None
                    registered = False

            # Stabilize recognition
            stable_name = track_manager.update(track_id, name, dist, frame_no)
            label = stable_name if stable_name else name

            # Draw annotation
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{label} ID:{track_id}",
                        (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Log event
            if logger:
                logger.log(frame_no, track_id, label, dist, registered)

        # Remove inactive tracks
        track_manager.remove_inactive_tracks(frame_no, max_age_frames=50)

        # Display or save
        if display:
            cv2.imshow("Face ReID Video", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if writer:
            writer.write(frame)

    # Cleanup
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
