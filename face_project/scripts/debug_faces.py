import os
import cv2
import insightface
from insightface.app import FaceAnalysis

# Test on a few images
dataset_dir = "face_project/dataset/shafiq_4SN23AD001"
images = os.listdir(dataset_dir)[:5]

print("Initializing Face Analysis...")
app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
app.prepare(ctx_id=-1, det_size=(640, 640))

for img_file in images:
    img_path = os.path.join(dataset_dir, img_file)
    img = cv2.imread(img_path)
    print(f"\nImage: {img_file}, Shape: {img.shape if img is not None else 'None'}")
    
    if img is None:
        print("  Could not read image")
        continue
        
    faces = app.get(img)
    print(f"  Faces detected: {len(faces)}")
    
    if len(faces) > 0:
        face = faces[0]
        if hasattr(face, 'normed_embedding'):
            print(f"  Has normed_embedding: {face.normed_embedding is not None}")
        if hasattr(face, 'embedding'):
            print(f"  Has embedding: {face.embedding is not None}")
