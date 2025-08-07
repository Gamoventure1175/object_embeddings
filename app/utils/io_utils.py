import os
import cv2
import numpy as np

def save_face_image(path, image):
    cv2.imwrite(path, image)

def save_embedding(path, embedding):
    np.save(path, embedding)

def load_embedding(path):
    return np.load(path)

def list_files(dir_path, ext=".npy"):
    return sorted([f for f in os.listdir(dir_path) if f.endswith(ext)])