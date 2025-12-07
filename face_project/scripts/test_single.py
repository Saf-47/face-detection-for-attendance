import cv2
import numpy as np
import joblib
import os
import sys
import insightface
from insightface.app import FaceAnalysis

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/test_single.py <path_to_image>")
        # Fallback for testing without args if user just runs it
        # Try to find a random image in dataset
        dataset_dir = os.path.join("face_project", "dataset")
        if os.path.exists(dataset_dir):
            for root, dirs, files in os.walk(dataset_dir):
                for f in files:
                    if f.endswith(".jpg"):
                        print(f"No argument provided. Using random sample: {f}")
                        test_image(os.path.join(root, f))
                        return
        return

    img_path = sys.argv[1]
    test_image(img_path)

def test_image(img_path):
    MODEL_PATH = os.path.join("face_project", "face_clf.pkl")
    
    if not os.path.exists(MODEL_PATH):
        print("Error: Model file not found.")
        return
        
    if not os.path.exists(img_path):
        print(f"Error: Image {img_path} not found.")
        return

    print(f"Testing image: {img_path}")
    
    # Load model
    model_data = joblib.load(MODEL_PATH)
    clf = model_data['clf']
    le = model_data['le']
    scaler = model_data['scaler']
    
    # Init InsightFace
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    
    img = cv2.imread(img_path)
    if img is None:
        print("Error: Could not read image.")
        return
        
    faces = app.get(img)
    
    if len(faces) == 0:
        print("No faces detected.")
        return
        
    print(f"Detected {len(faces)} faces.")
    
    for i, face in enumerate(faces):
        if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
            embedding = face.normed_embedding
        else:
            embedding = face.embedding
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
                
        embedding_scaled = scaler.transform([embedding])
        probs = clf.predict_proba(embedding_scaled)[0]
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)
        label = le.inverse_transform([pred_idx])[0]
        
        print(f"Face {i+1}: Predicted {label} with confidence {max_prob:.4f}")

if __name__ == "__main__":
    main()
