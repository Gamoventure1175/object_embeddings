import sys
import os
import cv2
import numpy as np
import glob
import shutil

from detectors.yolov11_face_detector import YOLOv11FaceDetector
from embedder.insightface_embedder import ArcFaceEmbedder
from matcher.annoy_matcher import AnnoyFaceMatcher
from config import CONFIG

# ---------- Config ----------
IMG_PATH        = CONFIG.get("IMAGE_PATH")
REGISTER_MODE   = CONFIG.get("REGISTER_MODE", False)
PERSON_NAME     = CONFIG.get("PERSON_NAME", "New Person")
MATCH_THRESHOLD = CONFIG.get("MATCH_THRESHOLD", 1.1)
TOP_K           = CONFIG.get("TOP_K", 3)
DISPLAY         = CONFIG.get("DISPLAY", True)
SAVE_ANNOTATED  = CONFIG.get("SAVE_ANNOTATED_IMAGE", None)

INDEX_PATH      = CONFIG.get("INDEX_PATH", "faces.ann")
META_PATH       = CONFIG.get("META_PATH", "faces_meta.json")
EMB_PATH        = CONFIG.get("EMBEDDINGS_PATH", "face_embeddings.npy")
VECTOR_SIZE     = CONFIG.get("VECTOR_SIZE", 512)
N_TREES         = CONFIG.get("ANNOY_TREES", 10)

REGISTERED_IMG_DIR = "registered_faces"  # folder to save registered cropped faces
# --------------------------------------------------


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def clamp_bbox(x1, y1, x2, y2, w, h):
    x1 = max(0, int(x1)); y1 = max(0, int(y1))
    x2 = min(w - 1, int(x2)); y2 = min(h - 1, int(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def save_registered_face_image(name, crop_img):
    ensure_dir(REGISTERED_IMG_DIR)
    save_path = os.path.join(REGISTERED_IMG_DIR, name)
    cv2.imwrite(save_path, crop_img)


def register_from_folder(folder_path):
    detector = YOLOv11FaceDetector()
    embedder = ArcFaceEmbedder()
    matcher = AnnoyFaceMatcher(
        vector_size=VECTOR_SIZE,
        index_path=INDEX_PATH,
        meta_path=META_PATH,
        emb_path=EMB_PATH,
        threshold=MATCH_THRESHOLD,
        n_trees=N_TREES,
    )

    img_files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
    if not img_files:
        print("❌ No images found in", folder_path)
        return

    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is None:
            print("⚠️ Failed to read", img_path)
            continue

        detections = detector.detect_faces(img)
        if not detections:
            print(f" - No face in {os.path.basename(img_path)}")
            continue

        for det in detections:
            x1, y1, x2, y2, conf = det
            bbox = clamp_bbox(x1, y1, x2, y2, img.shape[1], img.shape[0])
            if not bbox:
                continue
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            emb = embedder.get_embedding_from_crop(crop)
            name = os.path.basename(img_path)
            matcher.register(name, emb)
            save_registered_face_image(name, crop)
            print(f"✅ Registered {name}")


def match_from_folder(folder_path):
    detector = YOLOv11FaceDetector()
    embedder = ArcFaceEmbedder()
    matcher = AnnoyFaceMatcher(
        vector_size=VECTOR_SIZE,
        index_path=INDEX_PATH,
        meta_path=META_PATH,
        emb_path=EMB_PATH,
        threshold=MATCH_THRESHOLD,
        n_trees=N_TREES,
    )

    try:
        matcher.load_for_matching()
    except Exception as e:
        print("❌ Cannot load matcher DB:", e)
        return

    registered_images = {}
    for _, name in matcher.meta.items():
        reg_img_path = os.path.join(REGISTERED_IMG_DIR, name)
        if os.path.exists(reg_img_path):
            registered_images[name] = cv2.imread(reg_img_path)

    img_files = sorted(glob.glob(os.path.join(folder_path, "*.*")))
    for img_path in img_files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        detections = detector.detect_faces(img)
        for det in detections:
            x1, y1, x2, y2, conf = det
            bbox = clamp_bbox(x1, y1, x2, y2, img.shape[1], img.shape[0])
            if not bbox:
                continue
            crop = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            emb = embedder.get_embedding_from_crop(crop)
            results = matcher.match(emb, top_k=TOP_K)

            if results:
                name, dist = results[0]
                print(f"{os.path.basename(img_path)} → {name} ({dist:.3f})")
                if name in registered_images:
                    query_resized = cv2.resize(crop, (200, 200))
                    reg_resized = cv2.resize(registered_images[name], (200, 200))
                    combined = np.hstack((query_resized, reg_resized))
                    cv2.putText(combined, f"{dist:.3f}", (5, 195),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.imshow("Query | Match", combined)
                    cv2.waitKey(0)
            else:
                # No match found — auto-register
                new_name = f"auto_{len(matcher.meta)}.jpg"
                matcher.register(new_name, emb)
                save_registered_face_image(new_name, crop)
                print(f"🆕 No match found. Auto-registered as {new_name}")
    cv2.destroyAllWindows()



def main():
    if not os.path.exists(IMG_PATH):
        print(f"❌ Image not found: {IMG_PATH}")
        sys.exit(1)

    img = cv2.imread(IMG_PATH)
    if img is None:
        print(f"❌ Failed to read image: {IMG_PATH}")
        sys.exit(1)
    H, W = img.shape[:2]

    detector = YOLOv11FaceDetector()
    embedder = ArcFaceEmbedder()
    matcher = AnnoyFaceMatcher(
        vector_size=VECTOR_SIZE,
        index_path=INDEX_PATH,
        meta_path=META_PATH,
        emb_path=EMB_PATH,
        threshold=MATCH_THRESHOLD,
        n_trees=N_TREES,
    )

    detections = detector.detect_faces(img)
    if not detections:
        print("❌ No faces detected.")
        if SAVE_ANNOTATED:
            cv2.imwrite(SAVE_ANNOTATED, img)
        sys.exit(0)

    print(f"✅ Detected {len(detections)} face(s).")
    draw_items = []

    if not REGISTER_MODE:
        try:
            matcher.load_for_matching()
            print("[image] Loaded face DB for matching.")
        except FileNotFoundError:
            print("❌ No face DB found. Switch to REGISTER_MODE to add entries first.")
            sys.exit(0)
        except Exception as e:
            print(f"⚠️ Failed to load DB: {e}")
            sys.exit(1)

    for i, det in enumerate(detections, start=1):
        x1, y1, x2, y2, conf = det
        bbox = clamp_bbox(x1, y1, x2, y2, W, H)
        if bbox is None:
            print(f" - Face {i}: invalid bbox after clamping, skipping.")
            continue
        x1c, y1c, x2c, y2c = bbox
        crop = img[y1c:y2c, x1c:x2c]
        if crop.size == 0:
            print(f" - Face {i}: empty crop, skipping.")
            continue

        try:
            emb = embedder.get_embedding_from_crop(crop)
        except Exception as e:
            print(f" - Face {i}: embedder error: {e}")
            continue
        if emb is None:
            print(f" - Face {i}: failed to compute embedding.")
            continue

        if REGISTER_MODE:
            filename = f"{PERSON_NAME.replace(' ', '_')}.jpg"
            matcher.register(filename, emb)
            save_registered_face_image(filename, crop)
            label = f"Registered: {PERSON_NAME}"
            print(f"✅ Registered face {i} as '{PERSON_NAME}'.")
        else:
            results = matcher.match(emb, top_k=TOP_K) or []
            if results:
                name, dist = results[0]
                label = f"{name} ({dist:.3f})"
                print(f"✅ Face {i}: {name} (distance: {dist:.4f})")
            else:
                label = "Unknown"
                print(f"❌ Face {i}: no match within threshold.")

        draw_items.append((x1c, y1c, x2c, y2c, label))

    for (x1c, y1c, x2c, y2c, label) in draw_items:
        cv2.rectangle(img, (x1c, y1c), (x2c, y2c), (0, 255, 0), 2)
        cv2.putText(img, label, (x1c, max(0, y1c - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if DISPLAY:
        cv2.imshow("Image Face Pipeline", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if SAVE_ANNOTATED:
        ok = cv2.imwrite(SAVE_ANNOTATED, img)
        print("💾 Saved:", SAVE_ANNOTATED if ok else "⚠️ Failed to save annotated image")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--register-folder", type=str)
    parser.add_argument("--match-folder", type=str)
    parser.add_argument("--reset-db", action="store_true", help="Delete all stored faces, embeddings, and registered images")
    args = parser.parse_args()

    if args.reset_db:
        # Delete Annoy DB, meta, embeddings
        for path in [META_PATH, INDEX_PATH, EMB_PATH]:
            if os.path.exists(path):
                os.remove(path)
                print(f"🗑️ Deleted {path}")
        # Delete registered images folder
        if os.path.exists(REGISTERED_IMG_DIR):
            shutil.rmtree(REGISTERED_IMG_DIR)
            print(f"🗑️ Deleted folder {REGISTERED_IMG_DIR}")
        print("✅ Database reset complete.")
        sys.exit(0)

    if args.register_folder:
        register_from_folder(args.register_folder)
    elif args.match_folder:
        match_from_folder(args.match_folder)
    else:
        main()

