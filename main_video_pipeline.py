# main_video_pipeline.py
import time
import csv
import numpy as np
if not hasattr(np, 'float'):
    np.float = float

import cv2
from detectors.yolov11_face_detector import YOLOv11FaceDetector
from embedder.insightface_embedder import ArcFaceEmbedder
from matcher.annoy_matcher import AnnoyFaceMatcher
from trackers.bytetrack_tracker import ByteTrackWrapper

from config import CONFIG

# ---------------- Video Settings ----------------
VIDEO_PATH = CONFIG['VIDEO_PATH']
DISPLAY = CONFIG['DISPLAY']
OUTPUT_PATH = CONFIG['OUTPUT_PATH']
MODEL_INPUT_SIZE = CONFIG['MODEL_INPUT_SIZE']
TOP_K = CONFIG['TOP_K']
# -------------------------------------------------


def main(video_path=VIDEO_PATH, display=DISPLAY, output_path=OUTPUT_PATH):
    detector = YOLOv11FaceDetector()
    embedder = ArcFaceEmbedder()
    matcher = AnnoyFaceMatcher(threshold=CONFIG["MATCH_THRESHOLD"])

    # Try to load existing embeddings
    try:
        matcher.load_for_matching()
        print("[main] Loaded face DB for matching.")
    except FileNotFoundError:
        print("[main] No face DB found — starting empty.")
    except Exception as e:
        print(f"[main] Warning loading DB: {e}")
        print("[main] Continuing with empty DB.")

    tracker = ByteTrackWrapper()

    # CSV logging setup
    if CONFIG.get("LOG_TO_CSV"):
        csv_file = open(CONFIG["CSV_LOG_PATH"], mode="a", newline="")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "track_id", "name", "distance"])
    else:
        csv_writer = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # State tracking
    trackid_to_name = {}
    trackid_frame_counts = {}  # How many consecutive frames this track has been seen
    last_match_frame = {}      # Last frame index when we matched
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_h, frame_w = frame.shape[:2]

        # Step 1: Detection
        detections = detector.detect_faces(frame)
        if detections:
            scale = min(MODEL_INPUT_SIZE[0] / float(frame_h),
                        MODEL_INPUT_SIZE[1] / float(frame_w))
            detections_model = [
                (float(x1) * scale, float(y1) * scale,
                 float(x2) * scale, float(y2) * scale, float(conf))
                for (x1, y1, x2, y2, conf) in detections
            ]
        else:
            detections_model = []

        # Step 2: Tracking
        img_info = (frame_h, frame_w)
        tracks = tracker.update(detections_model, img_info=img_info, img_size=MODEL_INPUT_SIZE)

        # Step 3: Recognition
        for t in tracks:
            bbox = list(map(int, t["bbox"]))
            x1, y1, x2, y2 = bbox
            track_id = int(t["track_id"])

            # Track appearance count
            trackid_frame_counts[track_id] = trackid_frame_counts.get(track_id, 0) + 1

            # Skip until stable
            if trackid_frame_counts[track_id] < CONFIG["STABLE_FRAMES"]:
                continue

            # Skip matching if too soon
            if frame_idx - last_match_frame.get(track_id, -9999) < CONFIG["MATCH_EVERY_N"]:
                assigned_name = trackid_to_name.get(track_id, None)
                if assigned_name:
                    label = f"{assigned_name} ID:{track_id} (cached)"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                continue

            last_match_frame[track_id] = frame_idx

            # Crop face
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame_w - 1, x2), min(frame_h - 1, y2)
            if x2c <= x1c or y2c <= y1c:
                continue
            face_crop = frame[y1c:y2c, x1c:x2c]
            if face_crop.size == 0:
                continue

            # Embed
            try:
                embedding = embedder.get_embedding_from_crop(face_crop)
            except Exception as e:
                print(f"[main] Embedder failure for track {track_id}: {e}")
                continue
            if embedding is None:
                continue

            # Match
            results = []
            try:
                results = matcher.match(embedding, top_k=TOP_K)
            except Exception:
                results = []

            assigned_name = None
            if results:
                assigned_name, dist = results[0]
                trackid_to_name[track_id] = assigned_name
                label = f"{assigned_name} ID:{track_id} Dist:{dist:.2f}"

                if csv_writer:
                    csv_writer.writerow([time.time(), track_id, assigned_name, f"{dist:.4f}"])

            else:
                if track_id in trackid_to_name:
                    assigned_name = trackid_to_name[track_id]
                    label = f"{assigned_name} ID:{track_id} (cached)"
                else:
                    label = f"Unknown ID:{track_id}"
                    if CONFIG["AUTO_REGISTER_UNKNOWN"]:
                        new_name = f"person_{matcher.next_id}"
                        print(f"[main] Registering new unknown as '{new_name}' for track {track_id}")
                        try:
                            matcher.register(new_name, embedding)
                            trackid_to_name[track_id] = new_name
                            label = f"{new_name} ID:{track_id} (registered)"
                        except Exception as e:
                            print(f"[main] Failed to register unknown face: {e}")

            # Draw
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, max(0, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Output
        if display:
            cv2.imshow("Face ReID Video", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
        if writer:
            writer.write(frame)

        frame_idx += 1

    cap.release()
    if writer:
        writer.release()
    if csv_writer:
        csv_file.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(VIDEO_PATH, DISPLAY, OUTPUT_PATH)
