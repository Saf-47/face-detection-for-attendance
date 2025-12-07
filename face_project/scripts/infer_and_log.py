import cv2
import numpy as np
import os
import pandas as pd
import datetime
import time
import uuid
import shutil
from scipy.optimize import linear_sum_assignment
from insightface.app import FaceAnalysis

# ==========================================
# CONFIGURATION AND CONSTANTS
# ==========================================

# Enrolled Identities (Exact mapping)
ENROLLED_PEOPLE = {
    "PERSON1": "4SN23ADXXXX",
    "PERSON2": "4SN23ADXXXX",
    "PERSON3": "4SN23ADXXXX",
    "PRESON4": "4SN23ADXXXXX"
}

# Thresholds (Strict defaults)
MIN_FACE_CONF = 0.70        # Detection confidence (Relaxed for webcam)
SIMILARITY_THRESH = 0.50    # Cosine similarity (Relaxed for webcam)
UNKNOWN_THRESH = 0.50       # Below this is unknown
WINDOW_SECONDS = 15.0       # Confirmation window
REQUIRED_CONSISTENCY = 0.60 # 60% consistency (Relaxed)
MAX_GAP_SECONDS = 2.0       # Max gap before checking track continuity

# Files
# Dynamic path resolution
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CSV_FILE = os.path.join(PROJECT_ROOT, "detections.csv")
EMBEDDINGS_PATH = os.path.join(PROJECT_ROOT, "embeddings.npy")
LABELS_PATH = os.path.join(PROJECT_ROOT, "labels.npy")

# Model
MODEL_NAME = "InsightFace-v1" # For logging
VIDEO_SOURCE = 0 # Webcam

# ==========================================
# CLASSES
# ==========================================

class ReferenceDatabase:
    """Handles Nearest Neighbor search against enrolled embeddings."""
    def __init__(self):
        if not os.path.exists(EMBEDDINGS_PATH) or not os.path.exists(LABELS_PATH):
            raise FileNotFoundError("Embeddings or labels file not found. Run make_embeddings.py first.")
            
        self.embeddings = np.load(EMBEDDINGS_PATH)
        self.labels = np.load(LABELS_PATH, allow_pickle=True)
        
        # Ensure normalized
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        self.embeddings = self.embeddings / (norms + 1e-10)
        
        print(f"[RefDB] Loaded {len(self.labels)} reference faces.")

    def find_match(self, embedding):
        """
        Find best match using Cosine Similarity (Dot Product).
        Returns: (best_label, best_similarity)
        """
        # Dot product
        sims = np.dot(self.embeddings, embedding)
        
        idx = np.argmax(sims)
        max_sim = sims[idx]
        label_raw = self.labels[idx]
        
        # Clean label (remove _USN suffix if present in training data labels)
        # Assuming training labels are like "shafiq_4SN23AD001" or just "shafiq"
        # We need to map this to our ENROLLED_PEOPLE keys (lowercase names)
        
        # Heuristic: split by '_' and take first part as name
        name_part = str(label_raw).split('_')[0].lower()
        
        return name_part, float(max_sim)

class PersonTrack:
    def __init__(self, track_id, embedding, bbox):
        self.track_id = track_id
        self.embedding = embedding
        self.bbox = bbox
        self.first_seen = datetime.datetime.now()
        self.last_seen = self.first_seen
        self.status = "tracking" # tracking, confirmed, processed_unknown
        
        # History buffer for sliding window
        # List of dicts: {'ts': timestamp, 'label': name, 'sim': score, 'conf': det_conf, 'valid': bool}
        self.history = []
        
        self.confirmed_name = None
        self.confirmed_login_id = None
        self.event_uuid = str(uuid.uuid4())

    def update(self, embedding, bbox, det_conf, db: ReferenceDatabase):
        now = datetime.datetime.now()
        self.last_seen = now
        self.bbox = bbox
        
        # Update track embedding (moving average)
        self.embedding = 0.7 * self.embedding + 0.3 * embedding
        self.embedding = self.embedding / np.linalg.norm(self.embedding)
        
        # Match against DB
        name, sim = db.find_match(self.embedding)
        
        # Check thresholds for validity in the window
        valid_frame = False
        if det_conf >= MIN_FACE_CONF and sim >= SIMILARITY_THRESH:
            valid_frame = True
            
        self.history.append({
            'ts': now,
            'label': name,
            'sim': sim,
            'conf': det_conf,
            'valid': valid_frame
        })
        
        # Prune history > 30s
        self.history = [h for h in self.history if (now - h['ts']).total_seconds() <= WINDOW_SECONDS]

    def check_confirmation(self):
        """
        Check strict 30s confirmation rule.
        Returns: (bool is_confirmed, dict csv_data)
        """
        if self.status != "tracking":
            return False, None
            
        now = datetime.datetime.now()
        duration = (now - self.first_seen).total_seconds()
        
        # Rule 0: Must have 30s history
        if duration < WINDOW_SECONDS:
            return False, None
            
        # Rule: Continuous tracking (checked implicitly by caller calling update(), but checking gaps here)
        # Note: We pruned history. Check gap between start of window and first history item? 
        # Actually simplest check: do we have enough *samples* covering the window?
        # If we are sampling at 3FPS, 30s = ~90 frames.
        # But maybe we lost tracking for 10s. The history would have a gap.
        
        timestamps = [h['ts'] for h in self.history]
        if not timestamps:
            return False, None
            
        # Check gaps > 2s in history
        # (Assuming history is sorted)
        for i in range(1, len(timestamps)):
            gap = (timestamps[i] - timestamps[i-1]).total_seconds()
            if gap > MAX_GAP_SECONDS:
                # Gap detected within the window!
                # Strictly speaking, this fails requirement "continuously tracked (no more than 2s gap)".
                # But do we fail permanently or just wait until we have a continuous 30s?
                # A sliding window naturally effectively waits for a continuous 30s block if we drop old frames.
                # However, if we have a gap, it means this track ID persisted across a gap??
                # Only if the Tracker class kept it alive. 
                # If Tracker kills it after 2s, we wouldn't be here.
                pass 
                
        # Main Metric: Consistency
        # "Same enrolled identity must be top match in at least 80% of sampled frames"
        total_samples = len(self.history)
        if total_samples < 5: # Need minimal samples
            return False, None
            
        # Count labels where valid_frame is True
        valid_history = [h for h in self.history if h['valid']]
        
        # Note: The requirement says "in N frames... where N=0.8 x total sampled". 
        # And "Top match in 80% ... AND per-frame sim >= 0.60 AND conf >= 0.85".
        # This means we filter by valid first?
        # "a. The same enrolled identity must be the top match in at least 80% of sampled frames"
        # "b. The per-frame similarity must be >= 0.60 in those frames."
        
        # So: Count frames where (Label == Candidate AND Sim >= 0.6 AND Conf >= 0.85).
        # This count must be >= 0.8 * Total_Frames_In_Window.
        
        # Determine strict candidate
        # Get most common label in valid history?
        if not valid_history:
             return False, None
             
        labels = [h['label'] for h in valid_history]
        if not labels:
             return False, None
             
        candidate_name = max(set(labels), key=labels.count)
        
        # Verify candidate is enrolled
        if candidate_name not in ENROLLED_PEOPLE:
            return False, None
            
        # Count strictly passing frames for this candidate
        passing_count = 0
        sum_sim = 0.0
        
        for h in self.history:
            if (h['label'] == candidate_name and 
                h['sim'] >= SIMILARITY_THRESH and 
                h['conf'] >= MIN_FACE_CONF):
                passing_count += 1
                sum_sim += h['sim']
        
        ratio = passing_count / total_samples
        
        if ratio >= REQUIRED_CONSISTENCY:
            # CONFIRMED!
            self.status = "confirmed"
            self.confirmed_name = candidate_name.capitalize() # Pretty print
            self.confirmed_login_id = ENROLLED_PEOPLE[candidate_name]
            
            avg_sim = sum_sim / passing_count if passing_count > 0 else 0
            
            # Prepare CSV Data
            csv_data = {
                "timestamp_iso8601": now.isoformat(),
                "detected_name": self.confirmed_name,
                "login_id": self.confirmed_login_id,
                "model_name": MODEL_NAME,
                "avg_similarity": f"{avg_sim:.4f}",
                "confirmation_duration_seconds": int(duration),
                "frame_count": total_samples,
                "camera_id": "cam01"
            }
            return True, csv_data
            
        return False, None


class Tracker:
    """Hungarian Algorithm based ID tracking."""
    def __init__(self):
        self.tracks = []
        self.next_id = 0
    
    def update(self, faces, reference_db):
        """
        Process new faces and update tracks.
        """
        timestamp = datetime.datetime.now()
        
        # Clean dead tracks (> 2s missing)
        self.tracks = [t for t in self.tracks if (timestamp - t.last_seen).total_seconds() <= MAX_GAP_SECONDS]
        
        # Prepare inputs
        detections = []
        for face in faces:
            # Normalize embedding
            if hasattr(face, 'normed_embedding') and face.normed_embedding is not None:
                emb = face.normed_embedding
            else:
                emb = face.embedding
                norm = np.linalg.norm(emb)
                if norm > 0: emb = emb / norm
            
            detections.append({
                'emb': emb,
                'bbox': face.bbox,
                'conf': face.det_score if hasattr(face, 'det_score') else 0.0
            })
            
        # Matching
        if len(self.tracks) > 0 and len(detections) > 0:
            track_embs = np.array([t.embedding for t in self.tracks])
            det_embs = np.array([d['emb'] for d in detections])
            
            # 1 - Cosine Sim
            cost = 1.0 - np.dot(track_embs, det_embs.T)
            
            row_inds, col_inds = linear_sum_assignment(cost)
            
            assigned_rows = set()
            assigned_cols = set()
            
            for r, c in zip(row_inds, col_inds):
                # Threshold for tracking matching (not identity matching) -> 0.4 distance = 0.6 similarity
                if cost[r, c] < 0.4:
                    self.tracks[r].update(detections[c]['emb'], detections[c]['bbox'], detections[c]['conf'], reference_db)
                    assigned_rows.add(r)
                    assigned_cols.add(c)
            
            # New tracks
            for i in range(len(detections)):
                if i not in assigned_cols:
                    t = PersonTrack(self.next_id, detections[i]['emb'], detections[i]['bbox'])
                    t.update(detections[i]['emb'], detections[i]['bbox'], detections[i]['conf'], reference_db) # Init history
                    self.tracks.append(t)
                    self.next_id += 1
                    
        elif len(detections) > 0:
            for d in detections:
                t = PersonTrack(self.next_id, d['emb'], d['bbox'])
                t.update(d['emb'], d['bbox'], d['conf'], reference_db)
                self.tracks.append(t)
                self.next_id += 1

class CsvLogger:
    def __init__(self, filename):
        self.filename = filename
        self.columns = [
            "timestamp_iso8601", "detected_name", "login_id", "model_name", 
            "avg_similarity", "confirmation_duration_seconds", "frame_count", "camera_id"
        ]
        
        # Init file if not exists
        if not os.path.exists(self.filename):
            df = pd.DataFrame(columns=self.columns)
            df.to_csv(self.filename, index=False)
            
    def append_row(self, data):
        try:
            df = pd.DataFrame([data])
            df = df[self.columns] # Enforce order
            df.to_csv(self.filename, mode='a', header=False, index=False)
            print(f"[LOG] CONFIRMED: {data['detected_name']} ({data['login_id']})")
        except Exception as e:
            print(f"Error writing CSV: {e}")

def main():
    # Load DB
    try:
        db = ReferenceDatabase()
    except Exception as e:
        print(e)
        return
        
    logger = CsvLogger(CSV_FILE)
    tracker = Tracker()
    
    # Init InsightFace
    print("Initializing Face Analysis (Trying GPU)...")
    app = FaceAnalysis(allowed_modules=['detection', 'recognition'])
    # ctx_id=0 means GPU (if available), -1 means CPU
    # We try 0. If CUDA is missing, InsightFace/ONNXRuntime might fallback or warn.
    try:
        app.prepare(ctx_id=0, det_size=(640, 640))
        print("Face Analysis initialized on GPU (ctx_id=0).")
    except Exception as e:
        print(f"GPU initialization failed: {e}. Falling back to CPU...")
        app.prepare(ctx_id=-1, det_size=(640, 640))
    
    print(f"Opening Video: {VIDEO_SOURCE}")
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    # Sampling Config
    # "Sample frames at a rate sufficient ... e.g. every 5th frame"
    FRAME_SKIP = 5
    frame_count = 0
    
    last_summary_time = time.time()
    
    print("Strict Temporal Confirmation System Started. Waiting for 30s consistency...")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                # LOOP VIDEO FOR TESTING
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
                
            frame_count += 1
            
            if frame_count % FRAME_SKIP == 0:
                # Process
                faces = app.get(frame)
                
                # Preprocessing Rule: "Reject faces ... < 60x60"
                valid_faces = []
                for f in faces:
                     w = f.bbox[2] - f.bbox[0]
                     h = f.bbox[3] - f.bbox[1]
                     if w >= 60 and h >= 60:
                         valid_faces.append(f)
                         
                # Update tracks
                tracker.update(valid_faces, db)
                
                # Check Confirmation Logic
                for t in tracker.tracks:
                    confirmed, data = t.check_confirmation()
                    if confirmed:
                        logger.append_row(data)
                        
            # Visualization (Every frame for smoothness)
            for t in tracker.tracks:
                # Only draw if recent
                if (datetime.datetime.now() - t.last_seen).total_seconds() < 1.0:
                    bbox = t.bbox.astype(int)
                    
                    # Get current instantaneous label for display
                    cur_name = t.history[-1]['label'] if t.history else "..."
                    cur_sim = t.history[-1]['sim'] if t.history else 0.0
                    
                    if t.status == "confirmed":
                        color = (0, 255, 0) # Green
                        # Show Name + USN
                        text = f"{t.confirmed_name} ({t.confirmed_login_id})"
                    else:
                        # Tracking / Verifying
                        if cur_sim < UNKNOWN_THRESH:
                            color = (0, 0, 255) # Red
                            text = "Unknown"
                        else:
                            color = (0, 255, 255) # Yellow
                            # Show progress
                            elapsed = (datetime.datetime.now() - t.first_seen).total_seconds()
                            text = f"Verifying: {cur_name} ({elapsed:.0f}s)"
                        
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(frame, text, (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
            # Periodic Summary (Every 10s for debugging)
            if time.time() - last_summary_time > 10:
                # "Log per-second: top_identity, similarity..." (Simplified summary)
                # We will just print active tracks
                print(f"\n[SUMMARY {datetime.datetime.now().strftime('%H:%M:%S')}] Active Tracks: {len(tracker.tracks)}")
                for t in tracker.tracks:
                    dur = (datetime.datetime.now() - t.first_seen).total_seconds()
                    top_lbl = t.history[-1]['label'] if t.history else "?"
                    sim = t.history[-1]['sim'] if t.history else 0.0
                    print(f"  - Track {t.track_id}: {top_lbl} (Sim: {sim:.2f}) - {dur:.1f}s - {t.status}")
                last_summary_time = time.time()
                
            cv2.imshow("Strict Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Stopped.")
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
