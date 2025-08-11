import numpy as np
if not hasattr(np, 'float'):
    np.float = float


import cv2
from detectors.yolov11_face_detector import YOLOv11FaceDetector
from embedder.insightface_embedder import ArcFaceEmbedder
from matcher.annoy_matcher import AnnoyFaceMatcher
from trackers.bytetrack_tracker import ByteTrackWrapper

def main(video_path, display=True, output_path=None):
    # Initialize components
    detector = YOLOv11FaceDetector()
    embedder = ArcFaceEmbedder()
    matcher = AnnoyFaceMatcher(threshold=1.2)
    
    try:
        matcher.load_for_matching()
    except FileNotFoundError:
        print("❌ No registered faces found. Please register faces first.")
        return

    tracker = ByteTrackWrapper()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return

    # Optional: output video writer setup
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Model input size your detector expects (adjust if different)
    model_input_size = (640, 640)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Detect faces
        detections = detector.detect_faces(frame)  # Assuming detect_faces returns list of [x1,y1,x2,y2,score]

        # Step 2: Track faces using ByteTrack
        frame_height, frame_width = frame.shape[:2]
        img_info = (frame_height, frame_width)
        tracks = tracker.update(detections, img_info=img_info, img_size=model_input_size)

        # Step 3: For each tracked face get embedding + match
        for track in tracks:
            # track is a dict with keys: 'track_id', 'bbox', 'score'
            x1, y1, x2, y2 = map(int, track['bbox'])
            track_id = track['track_id']

            # Crop face region safely within image bounds
            x1c, y1c = max(0, x1), max(0, y1)
            x2c, y2c = min(frame_width, x2), min(frame_height, y2)
            face_crop = frame[y1c:y2c, x1c:x2c]

            if face_crop.size == 0:
                continue  # skip invalid crops

            embedding = embedder.get_embedding_from_crop(face_crop)
            results = matcher.match(embedding, top_k=1)

            # Prepare label text
            if results:
                name, dist = results[0]
                label = f"{name} ID:{track_id} Dist:{dist:.2f}"
            else:
                label = f"Unknown ID:{track_id}"

            # Draw bounding box and label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Show frame if display=True
        if display:
            cv2.imshow("Face ReID Video", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
                break

        # Write frame to output video if needed
        if writer:
            writer.write(frame)

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = "media/videos/tom_holland.mp4"  # change to your video path
    main(video_path)
