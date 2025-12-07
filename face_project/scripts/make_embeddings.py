import os
import cv2
import numpy as np
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis

def main():
    # Dynamic path resolution
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "embeddings.npy")
    LABELS_PATH = os.path.join(PROJECT_ROOT, "labels.npy")
    IMAGE_PATHS_PATH = os.path.join(PROJECT_ROOT, "image_paths.npy")
    
    if not os.path.exists(DATASET_DIR):
        print(f"Error: Dataset directory '{DATASET_DIR}' not found. Run extract_frames.py first.")
        return

    # Initialize InsightFace (detection + recognition) with smaller detection size
    # since faces are already cropped to 112x112
    # Initialize InsightFace (detection + recognition)
    print("Initializing Face Recognition (Trying GPU)...")
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    try:
        app.prepare(ctx_id=0, det_size=(160, 160))
        print("Face Recognition initialized on GPU.")
    except Exception:
        print("GPU failed. Using CPU.")
        app.prepare(ctx_id=-1, det_size=(160, 160))



    embeddings = []
    labels = []
    image_paths_list = []

    # Walk through dataset
    person_folders = [d for d in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, d))]
    
    print(f"Found {len(person_folders)} classes. Generating embeddings...")
    
    for person_name in tqdm(person_folders, desc="Classes"):
        person_dir = os.path.join(DATASET_DIR, person_name)
        image_files = [f for f in os.listdir(person_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(image_files, desc=f"  {person_name}", leave=False):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path)
            
            if img is None:
                continue
                
            faces = app.get(img)
            
            if len(faces) == 0:
                continue
                
            # If multiple faces, pick largest
            if len(faces) > 1:
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
            
            face = faces[0]
            
            # Get embedding
            if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                embedding = face.normed_embedding
            else:
                embedding = face.embedding
                norm = np.linalg.norm(embedding)
                if norm > 0:
                    embedding = embedding / norm
            
            embeddings.append(embedding)
            labels.append(person_name)
            image_paths_list.append(img_path)


    if not embeddings:
        print("No embeddings generated. Check your dataset.")
        return

    # Save to files
    print(f"\nSaving {len(embeddings)} embeddings...")
    np.save(EMBEDDINGS_PATH, np.array(embeddings, dtype=np.float32))
    np.save(LABELS_PATH, np.array(labels, dtype=object), allow_pickle=True)
    np.save(IMAGE_PATHS_PATH, np.array(image_paths_list, dtype=object), allow_pickle=True)
    
    print("Done. Saved embeddings.npy and labels.npy.")

if __name__ == "__main__":
    main()
