import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix

def main():
    EMBEDDINGS_PATH = os.path.join("face_project", "embeddings.npy")
    LABELS_PATH = os.path.join("face_project", "labels.npy")
    MODEL_PATH = os.path.join("face_project", "face_clf.pkl")

    if not os.path.exists(MODEL_PATH):
        print("Model not found.")
        return

    print("Loading data...")
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)
    
    print("Loading model...")
    model_data = joblib.load(MODEL_PATH)
    clf = model_data['clf']
    le = model_data['le']
    scaler = model_data['scaler']

    print("Evaluating...")
    X = embeddings
    y = le.transform(labels)
    X_scaled = scaler.transform(X)
    
    y_pred = clf.predict(X_scaled)
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=le.classes_))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

if __name__ == "__main__":
    main()
