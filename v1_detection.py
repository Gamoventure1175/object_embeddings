import cv2
import os
import time
import numpy as np
import face_recognition
from datetime import datetime
from ultralytics import YOLO

# ----- CONFIGURATION -----
video_path = None   # Set to None if using image
image_path = "sample_images/miles2.jpg"   # Set to None if using video
model_path = "yolov8n.pt"    # You can switch to yolov5/6/v4 etc. if needed

# ----- LOAD YOLO MODEL -----
model = YOLO(model_path)

# ----- FILE STEM -----
input_file = video_path if video_path else image_path
file_stem = os.path.splitext(os.path.basename(input_file))[0] if input_file else "input"

# Output folders based on file name (not timestamp anymore)
face_output_dir = os.path.join("detected_faces", file_stem)
embedding_output_dir = os.path.join("embeddings", file_stem)
os.makedirs(face_output_dir, exist_ok=True)
os.makedirs(embedding_output_dir, exist_ok=True)

# ----- HELPER: Generate face embedding -----
def get_embedding(face_image):
    rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb_image)
    return encodings[0] if encodings else None

# ----- PROCESS IMAGE -----
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"⚠️ Could not load image: {image_path}")
        return

    results = model(frame, verbose=False)[0]
    face_id = 0

    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = box.conf[0]
        if conf < 0.5:
            continue

        face_crop = frame[y1:y2, x1:x2]
        file_tag = f"{face_id}"

        face_path = os.path.join(face_output_dir, f"face_{file_tag}.jpg")
        cv2.imwrite(face_path, face_crop)

        embedding = get_embedding(face_crop)
        if embedding is not None:
            emb_path = os.path.join(embedding_output_dir, f"embedding_{file_tag}.npy")
            np.save(emb_path, embedding)
            print(f"✅ Saved embedding: {emb_path}")
        else:
            print("⚠️ Failed to generate embedding.")

        face_id += 1

    print("✅ Finished processing image.")

# ----- PROCESS VIDEO -----
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"⚠️ Failed to open video: {video_path}")
        return

    frame_id = 0
    face_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_id += 1
        results = model(frame, verbose=False)[0]

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0]
            if conf < 0.5:
                continue

            face_crop = frame[y1:y2, x1:x2]
            file_tag = f"{frame_id}_{face_id}"

            face_path = os.path.join(face_output_dir, f"face_{file_tag}.jpg")
            cv2.imwrite(face_path, face_crop)

            embedding = get_embedding(face_crop)
            if embedding is not None:
                emb_path = os.path.join(embedding_output_dir, f"embedding_{file_tag}.npy")
                np.save(emb_path, embedding)
                print(f"✅ Saved embedding: {emb_path}")
            else:
                print("⚠️ Failed to generate embedding.")

            face_id += 1

            # Optional: Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow("YOLO Face Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("✅ Finished processing video.")

# ----- MAIN EXECUTION -----
if video_path:
    process_video(video_path)
elif image_path:
    process_image(image_path)
else:
    print("❌ Please provide either a video or image path to process.")
