import cv2
import matplotlib.pyplot as plt

def show_comparison(img1_path, img2_path, euclidean_distance, cosine_similarity):
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None or img2 is None:
        print("⚠️ Failed to load images for comparison.")
        return

    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(img1)
    axs[0].set_title("Current Face")
    axs[1].imshow(img2)
    axs[1].set_title(
        f"Best Match\nEuclidean: {euclidean_distance:.2f}\nCosine: {cosine_similarity:.4f}"
    )
    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()