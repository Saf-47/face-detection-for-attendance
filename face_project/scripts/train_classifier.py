import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

def main():
    EMBEDDINGS_PATH = os.path.join("face_project", "embeddings.npy")
    LABELS_PATH = os.path.join("face_project", "labels.npy")
    MODEL_PATH = os.path.join("face_project", "face_clf.pkl")

    if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(LABELS_PATH):
        print("Error: Embeddings or labels not found. Run make_embeddings.py first.")
        return

    print("Loading embeddings and labels...")
    embeddings = np.load(EMBEDDINGS_PATH)
    labels = np.load(LABELS_PATH, allow_pickle=True)

    print(f"Loaded {len(embeddings)} samples.")

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    X = embeddings

    # Split data
    print("Splitting data (80/20)...")
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
    except ValueError as e:
        print(f"Warning: Stratified split failed (maybe too few samples per class?). Using random split. Error: {e}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    # Scale embeddings
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train SVM
    print("Training SVM classifier...")
    clf = SVC(kernel='linear', probability=True, class_weight='balanced')
    clf.fit(X_train_scaled, y_train)

    # Evaluate
    print("\nEvaluating on test set:")
    y_pred = clf.predict(X_test_scaled)
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model
    model_data = {
        'clf': clf,
        'le': le,
        'scaler': scaler
    }
    joblib.dump(model_data, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
