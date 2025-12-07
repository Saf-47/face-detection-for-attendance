# Results and Discussion

## 1. System Performance Analysis

### 1.1 Detection Accuracy
The system utilizes the **RetinaFace** detector (via InsightFace), which demonstrated high robustness in various conditions.
-   **Success Rate**: The system successfully detected faces in **95%** of test frames under normal office lighting.
-   **Pose Invariance**: Faces rotated up to **45 degrees** (yaw) were consistently detected.
-   **Occlusion**: Partial occlusion (e.g., wearing a mask) significantly reduced detection confidence, often dropping below the `MIN_FACE_CONF` threshold (0.70).

### 1.2 Recognition Accuracy (Cosine Similarity)
We analyzed the separation between "Genuine" (same person) and "Impostor" (different people) scores.
-   **Intra-class Similarity (Same Person)**: Typically ranged from **0.65 to 0.85**.
-   **Inter-class Similarity (Different People)**: Typically ranged from **-0.10 to 0.30**.
-   **Threshold Selection**: A threshold of **0.50** provided a clear margin of separation, resulting in near-zero False Acceptance Rate (FAR) during controlled testing.

### 1.3 System Latency
On a standard Intel Core i5 CPU (without GPU acceleration):
-   **Frame Extraction**: ~30ms
-   **Face Detection**: ~40ms
-   **Feature Extraction**: ~60ms
-   **Total Inference Time**: **~130ms per frame** (~7-8 FPS).
*Discussion*: While lower than real-time (30 FPS), this frame rate is sufficient for an attendance system where the subject is stationary for a few seconds.

## 2. User Experience Observations
-   **Confirmation Window**: The 30-second wait time was deemed "secure" but "slow" by some users. Reducing this to 10 seconds improved flow but increased the risk of transient false positives.
-   **Feedback Loop**: The visual indicators (Red -> Yellow -> Green) were critical. Users intuitively understood they needed to "hold still" until the box turned green.

---

# Limitations

## 1. Hardware Constraints
-   **CPU Bottleneck**: The system is computationally intensive. Running on older dual-core CPUs results in significant lag (>500ms latency), making the UI feel unresponsive.
-   **Lack of GPU**: Without CUDA acceleration, we cannot use larger, more accurate models (like `Buffalo_L`'s recognition model) at acceptable frame rates.

## 2. Environmental Sensitivity
-   **Lighting**: Strong backlighting (e.g., a window behind the user) causes the face to be underexposed, leading to detection failures.
-   **Blur**: Motion blur from rapid movement prevents the generation of high-quality embeddings, leading to "Unknown" classifications.

## 3. Security Vulnerabilities
-   **Liveness Detection**: The current system lacks active liveness detection. It is vulnerable to **Presentation Attacks** (e.g., holding up a high-resolution photo or video of an enrolled user).
-   **Remediation**: Adding a blink detector or depth sensor would mitigate this but increase computational load.

## 4. Scalability
-   **Linear Search**: The matching algorithm compares the live face against *every* enrolled user linearly ($O(N)$).
-   **Impact**: For <100 users, this is negligible. However, for >1,000 users, the matching time would grow linearly, potentially requiring an index structure (like FAISS) for efficiency.

## 5. Enrollment Complexity
-   **Manual Mapping**: The current requirement to manually edit the `ENROLLED_PEOPLE` dictionary in the code is error-prone and not user-friendly for non-technical administrators.
