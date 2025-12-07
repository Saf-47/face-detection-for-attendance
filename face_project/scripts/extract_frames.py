import os
import cv2
import glob
import sys
import numpy as np
from tqdm import tqdm
import insightface
from insightface.app import FaceAnalysis

def main():
    # Paths
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    VIDEO_DIR = os.path.join(PROJECT_ROOT, "videos")
    DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
    
    # Check if video directory exists
    if not os.path.exists(VIDEO_DIR):
        print(f"Error: Video directory '{VIDEO_DIR}' not found.")
        return

    # Get list of mp4 files
    video_files = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    if not video_files:
        print(f"No .mp4 files found in '{VIDEO_DIR}'. Please place videos there.")
        return

    # Initialize InsightFace detector (detection only)
    # ctx_id=-1 means CPU
    # ctx_id=0 means GPU, -1 means CPU
    print("Initializing Face Detector (Trying GPU)...")
    app = FaceAnalysis(allowed_modules=['detection'])
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("Face Detector initialized on GPU.")
    except Exception:
        print("GPU failed. Using CPU.")
        app.prepare(ctx_id=-1, det_size=(640, 640))

    print(f"Found {len(video_files)} videos. Starting processing...")

    for video_path in video_files:
        filename = os.path.basename(video_path)
        name_usn, ext = os.path.splitext(filename)
        
        # Create dataset folder for this person
        person_dir = os.path.join(DATASET_DIR, name_usn)
        os.makedirs(person_dir, exist_ok=True)
        
        print(f"\nProcessing: {filename} -> {person_dir}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            continue
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample every 5th frame
        sample_rate = 5
        saved_count = 0
        
        pbar = tqdm(total=total_frames, desc="Frames", unit="frame")
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                # Detect faces
                faces = app.get(frame)
                
                if len(faces) > 0:
                    # If multiple faces, pick the largest one by area
                    faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0]) * (x.bbox[3]-x.bbox[1]), reverse=True)
                    face = faces[0]
                    
                    # Get aligned face crop
                    # InsightFace 'kps' are keypoints. We can use face_align.norm_crop if we had the module imported,
                    # but app.get() returns faces with bounding boxes. 
                    # The requirement says "align/crop". 
                    # FaceAnalysis automatically aligns if we use the recognition module, but here we only enabled detection.
                    # Wait, the requirement says "FaceAnalysis with detection module only".
                    # But usually alignment happens during recognition or we need to do it manually using keypoints.
                    # However, the user said "align/crop and save".
                    # Let's use insightface.utils.face_align.norm_crop using the keypoints (kps).
                    
                    from insightface.utils import face_align
                    aimg = face_align.norm_crop(frame, landmark=face.kps)
                    
                    save_path = os.path.join(person_dir, f"{saved_count:03d}.jpg")
                    cv2.imwrite(save_path, aimg)
                    saved_count += 1
            
            frame_idx += 1
            pbar.update(1)
            
        pbar.close()
        cap.release()
        print(f"  -> Saved {saved_count} face images.")

    print("\nAll videos processed.")

if __name__ == "__main__":
    main()
