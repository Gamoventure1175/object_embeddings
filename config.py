from typing import Dict, Any

CONFIG: Dict[str, Any] = {
    # ---------------- Face Detection ----------------
    "MODEL_INPUT_SIZE": (640, 640),            # YOLO input size
    "AUTO_REGISTER_UNKNOWN": False,             # Register unknowns automatically
    "STABLE_FRAMES": 5,                        # Require stable ID across frames

    # ---------------- Embedding & Matching ----------------
    "VECTOR_SIZE": 512,                        # ArcFace embedding size
    "ANNOY_TREES": 10,                         # Annoy index trees
    "MATCH_THRESHOLD": 0.6,                   # Stricter distance threshold
    "MATCH_RATIO": 0.85,                       # Ratio test (best/second-best)
    "MARGIN_DIFF": 0.05,                       # Minimum margin between best & 2nd
    "MATCH_EVERY_N": 5,                        # Check embeddings every N frames
    "TOP_K": 2,                                # 1 -> only the best match, 2 -> best match out of 2 and so on and so forth
    "REGISTER_MODE": False,

    # ---------------- Persistence Paths ----------------
    "INDEX_PATH": "faces.ann",                 # Annoy index file
    "META_PATH": "faces_meta.json",            # Face metadata
    "EMBEDDINGS_PATH": "face_embeddings.npy",  # Stored embeddings
    "REGISTERED_IMG_DIR": "registered_faces",  # Saved crops of faces

    # ---------------- Image Mode ----------------
    "IMAGE_PATH": "media/images/testface1.jpg",        # Default input image
    "PERSON_NAME": "New Person",               # Label when registering
    "SAVE_ANNOTATED_IMAGE": "registered/",              # Path to save annotated output
    "DISPLAY": True,                           # Show image output

    # ---------------- Video Mode ----------------
    "VIDEO_PATH": "media/videos/switzerland.mp4",
    "OUTPUT_PATH": None,                       # Save output video (None=disable)

    # ---------------- Logging ----------------
    "LOG_TO_CSV": True,
    "CSV_LOG_PATH": "recognitions_log.csv",
}
