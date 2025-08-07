import os
from datetime import datetime

VIDEO_PATH = None
IMAGE_PATH = "sample_images/miles2.jpg"
MODEL_PATH = "models/yolov8n.pt"
SESSION_LOG_PATH = "latest_session.txt"

CURRENT_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
INPUT_FILE = VIDEO_PATH if VIDEO_PATH else IMAGE_PATH
FILE_STEM = os.path.splitext(os.path.basename(INPUT_FILE))[0] if INPUT_FILE else "input"

FACE_OUTPUT_DIR = os.path.join("detected_faces", CURRENT_TIMESTAMP)
EMBEDDING_OUTPUT_DIR = os.path.join("embeddings", CURRENT_TIMESTAMP)

os.makedirs(FACE_OUTPUT_DIR, exist_ok=True)
os.makedirs(EMBEDDING_OUTPUT_DIR, exist_ok=True)