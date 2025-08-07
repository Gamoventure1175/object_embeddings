import os
import numpy as np
from app.utils.io_utils import load_embedding, list_files
from app.comparison.visualizer import show_comparison

def compare_embeddings(prev_emb_dir, prev_face_dir, curr_emb_dir, curr_face_dir):
    prev_files = list_files(prev_emb_dir)
    curr_files = list_files(curr_emb_dir)

    for new_file in curr_files:
        new_emb = load_embedding(os.path.join(curr_emb_dir, new_file))
        new_face_path = os.path.join(curr_face_dir, new_file.replace("embedding_", "face_").replace(".npy", ".jpg"))

        best_distance = float("inf")
        best_cosine = None
        best_match = None

        for old_file in prev_files:
            old_emb = load_embedding(os.path.join(prev_emb_dir, old_file))
            euclidean_distance = np.linalg.norm(new_emb - old_emb)
            cosine_similarity = np.dot(new_emb, old_emb) / (np.linalg.norm(new_emb) * np.linalg.norm(old_emb))

            if euclidean_distance < best_distance:
                best_distance = euclidean_distance
                best_cosine = cosine_similarity
                best_match = old_file

        if best_match:
            old_face_path = os.path.join(prev_face_dir, best_match.replace("embedding_", "face_").replace(".npy", ".jpg"))
            print(f"✅ Match found: {new_file} <-> {best_match} | Euclidean: {best_distance:.2f} | Cosine: {best_cosine:.4f}")
            show_comparison(new_face_path, old_face_path, best_distance, best_cosine)
        else:
            print(f"❌ No match found for {new_file}")