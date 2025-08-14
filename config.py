from typing import Dict, Any

CONFIG: Dict[str, Any] = {
    # ---------------- Face Recognition Settings ----------------
    # Auto-register new unknown faces
    "AUTO_REGISTER_UNKNOWN": True,

    # Require the same ID to be detected in N consecutive frames before confirming
    "STABLE_FRAMES": 3,

    # Only re-check embeddings every N frames per track
    "MATCH_EVERY_N": 5,

    # Recognition distance threshold
    "MATCH_THRESHOLD": 1.1,

    # Path to your embeddings file and the meta data file for the embeddings
    "EMBEDDINGS_PATH": "face_embeddings.npy",
    "META_PATH": "faces_meta.json",


    # ---------------- Logging ----------------
    "LOG_TO_CSV": True,
    "CSV_LOG_PATH": "recognitions_log.csv",

    # ---------------- Video Settings ----------------
    "VIDEO_PATH": "media/videos/switzerland.mp4",  # Input video file path
    "DISPLAY": True,                           # Show video output
    "OUTPUT_PATH": None,                       # Save output video to file (you have to provide a path or None to disable)
    "MODEL_INPUT_SIZE": (640, 640),            # YOLO/Tracker model input size
    "TOP_K": 1,                                # How many top matches to return from matcher

    # ---------------- Image Settings ----------------
    "IMG_PATH": 'media/images/bradpitt2.jpg',
    "PERSON_NAME": 'New Person',
    "SAVE_ANNOTATED_IMAGE": None,
    
    # ---------------- Image Paths ----------------
    "INDEX_PATH": 'faces.ann',
    "META_PATH": 'faces_meta.json',
    'EMBEDDINGS_PATH': 'face_embeddings.npy',
    'VECTOR_SIZE': 512,
    'ANNOY_TREES': 10 
}
