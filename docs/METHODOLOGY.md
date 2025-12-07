# Detailed Technical Methodology: Face Recognition Attendance System

## 1. Introduction
This document presents the technical methodology for the Face Recognition Attendance System. The system is designed to operate in an offline, CPU-constrained environment while maintaining high accuracy and robustness against transient detection errors. It leverages Deep Metric Learning for feature extraction and a custom temporal tracking algorithm for identity verification.

## 2. Mathematical Foundation

### 2.1 Deep Metric Learning
Unlike traditional classification (which outputs a probability distribution over fixed classes), this system uses **Metric Learning**. The goal is to map face images into a high-dimensional Euclidean space $\mathbb{R}^d$ (where $d=512$) such that:
-   Faces of the **same** person are close together.
-   Faces of **different** people are far apart.

### 2.2 Cosine Similarity
To measure the "closeness" of two face vectors, we use Cosine Similarity. Given two embedding vectors $\mathbf{u}$ (Reference) and $\mathbf{v}$ (Live Capture), the similarity $S$ is defined as the cosine of the angle $\theta$ between them:

$$
S(\mathbf{u}, \mathbf{v}) = \cos(\theta) = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}
$$

Since our embedding network normalizes all output vectors to have a unit length ($||\mathbf{x}|| = 1$), the denominator becomes 1, simplifying the computation to a simple dot product:

$$
S(\mathbf{u}, \mathbf{v}) = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^{512} u_i v_i
$$

**Decision Boundary**:
$$
\text{Match} = \begin{cases} 
      \text{True} & \text{if } S(\mathbf{u}, \mathbf{v}) \geq \tau \\
      \text{False} & \text{if } S(\mathbf{u}, \mathbf{v}) < \tau 
   \end{cases}
$$
Where $\tau$ is the threshold (set to 0.50 in our system).

### 2.3 ArcFace Loss (Conceptual)
The underlying model (InsightFace) is trained using **ArcFace Loss** (Additive Angular Margin Loss).
$$
L = -\frac{1}{N} \sum_{i=1}^{N} \log \frac{e^{s (\cos(\theta_{y_i} + m))}}{e^{s (\cos(\theta_{y_i} + m))} + \sum_{j \neq y_i} e^{s \cos(\theta_j)}}
$$
*   $s$: Scale factor.
*   $m$: Angular margin penalty.
*   **Effect**: This forces the model to compress intra-class variance (tight clusters for each person) and maximize inter-class variance (large gaps between different people), significantly improving accuracy over standard Softmax loss.

## 3. Algorithm Pipeline

### 3.1 Preprocessing & Detection
1.  **Input**: RGB Frame $I \in \mathbb{R}^{H \times W \times 3}$.
2.  **Detection**: We use a single-stage detector (RetinaFace-based) to predict bounding boxes $B = [x_1, y_1, x_2, y_2]$ and 5 facial landmarks $L = \{(x_k, y_k)\}_{k=1}^5$ (eyes, nose, mouth corners).

### 3.2 Geometric Alignment (Affine Transformation)
To ensure pose invariance, we align the face to a standard template. We compute an affine transformation matrix $M$ that maps the detected landmarks $L$ to a set of standard reference points $L_{ref}$.
$$
I_{aligned} = \text{WarpAffine}(I, M, (112, 112))
$$
This ensures the eyes are always horizontal and centered.

### 3.3 Feature Extraction
The aligned image $I_{aligned}$ is passed through a ResNet-based Convolutional Neural Network (CNN).
$$
\mathbf{v} = f_{CNN}(I_{aligned}) \in \mathbb{R}^{512}
$$

## 4. Implementation Logic (Pseudo-Code)

The core logic in `infer_and_log.py` follows this structure:

```python
# Constants
THRESHOLD = 0.50
CONFIRMATION_FRAMES = 0.60 * WINDOW_SIZE

# 1. Load Reference Database
database = LoadEmbeddings("embeddings.npy") # Matrix of shape (N, 512)

# 2. Main Loop
while True:
    frame = Camera.read()
    
    # Detection & Recognition
    faces = InsightFace.get_faces(frame) # Returns list of (bbox, embedding)
    
    for face in faces:
        # Nearest Neighbor Search
        # Compute dot product against all N reference vectors
        similarities = dot_product(database.vectors, face.embedding)
        best_match_idx = argmax(similarities)
        best_score = similarities[best_match_idx]
        
        predicted_id = database.labels[best_match_idx]
        
        # Tracking (Hungarian Algorithm)
        # Assigns this detection to an existing Track ID (e.g., Track #42)
        track = Tracker.update(face.bbox, face.embedding)
        
        # Temporal Consistency Check
        track.history.append(predicted_id, best_score)
        
        if track.is_stable(duration=30s):
            # Check if same ID appears in >60% of frames
            consistency = track.count(predicted_id) / track.total_frames()
            
            if consistency > CONFIRMATION_FRAMES and best_score > THRESHOLD:
                LogAttendance(predicted_id)
                track.mark_confirmed()
```

## 5. System Specifications

### Hardware Requirements
-   **CPU**: Intel Core i3 (8th Gen) or equivalent (AVX2 support recommended for ONNX).
-   **RAM**: 4GB minimum (8GB recommended).
-   **Camera**: Standard USB Webcam (720p recommended).
-   **GPU**: Not required (Optional: NVIDIA GTX 1050+ for faster inference).

### Software Stack
-   **Language**: Python 3.8+
-   **Core Libraries**:
    -   `insightface`: Model inference.
    -   `onnxruntime`: High-performance CPU execution provider.
    -   `opencv-python`: Image processing.
    -   `numpy`: Matrix operations.
    -   `scikit-learn`: (Used in training script only).

## 6. Performance Analysis

| Metric | CPU (Intel i5) | GPU (RTX 3060) |
| :--- | :--- | :--- |
| **Detection Time** | ~40ms | ~10ms |
| **Recognition Time** | ~60ms | ~5ms |
| **Total Latency** | **~100ms (10 FPS)** | **~15ms (60+ FPS)** |

*Note: The system is capped at the webcam's native FPS (usually 30), but CPU processing might drop effective FPS to 10-15, which is sufficient for attendance.*

## 7. Conclusion
This methodology combines the mathematical rigor of ArcFace-based metric learning with practical engineering solutions (temporal tracking) to create a reliable, offline attendance system suitable for deployment on commodity hardware.
