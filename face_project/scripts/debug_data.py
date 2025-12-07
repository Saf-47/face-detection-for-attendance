import numpy as np
import os

try:
    embeddings = np.load("face_project/embeddings.npy")
    labels = np.load("face_project/labels.npy", allow_pickle=True)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Unique labels: {np.unique(labels)}")
except Exception as e:
    print(f"Error loading data: {e}")
