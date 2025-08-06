import numpy as np
import cv2
import os
from sklearn.metrics.pairwise import cosine_similarity
from numpy.linalg import norm
import matplotlib.pyplot as plt

# ----- CONFIG -----
embedding1_path = "embeddings/miles1/embedding_0.npy"
embedding2_path = "embeddings/miles2/embedding_0.npy"

# Optional (for visualization)
image1_path = "detected_faces/miles1/face_0.jpg"
image2_path = "detected_faces/miles2/face_0.jpg"

# ----- Load embeddings -----
embedding1 = np.load(embedding1_path)
embedding2 = np.load(embedding2_path)

# ----- Calculate cosine similarity -----
cos_sim = cosine_similarity([embedding1], [embedding2])[0][0]

# ----- Calculate Euclidean distance -----
euclidean_dist = norm(embedding1 - embedding2)

# ----- Print results -----
print(f"🔍 Cosine Similarity: {cos_sim:.4f} (1 = identical)")
print(f"📏 Euclidean Distance: {euclidean_dist:.4f} (0 = identical)")

# ----- Visualization -----
def show_faces_side_by_side(img1_path, img2_path):
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print("❌ One or both image paths do not exist. Skipping visualization.")
        return

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img1)
    ax[0].axis("off")
    ax[0].set_title("Face 1")

    ax[1].imshow(img2)
    ax[1].axis("off")
    ax[1].set_title("Face 2")

    plt.suptitle(f"Cosine: {cos_sim:.2f} | Euclidean: {euclidean_dist:.2f}", fontsize=14)
    plt.tight_layout()
    plt.show()

show_faces_side_by_side(image1_path, image2_path)
