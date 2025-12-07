# Face Recognition Attendance System

A complete, CPU-based face recognition attendance project using InsightFace and SVM.

> [!TIP]
> For detailed architecture, configuration, and logic explanation, see [PROJECT_DOCUMENTATION.md](docs/PROJECT_DOCUMENTATION.md).
> For methodology, see [METHODOLOGY.md](docs/METHODOLOGY.md).
> For results and limitations, see [RESULTS_AND_LIMITATIONS.md](docs/RESULTS_AND_LIMITATIONS.md).
> For diagrams and charts, see [GRAPHICAL_REPRESENTATIONS.md](docs/GRAPHICAL_REPRESENTATIONS.md).

## Project Structure
```
face_project/
  videos/                  # Place your training videos here (Name_USN.mp4)
  dataset/                 # Auto-generated face crops
  scripts/                 # Python scripts
  models/                  # Saved models
  detections.xlsx          # Attendance log
  requirements.txt         # Dependencies
```

## Setup

1.  **Install Python 3.8+**
2.  **Install Dependencies**:
    Open PowerShell in the `face_project` folder (or root) and run:
    ```powershell
    pip install -r face_project/requirements.txt
    ```
    *Note: If you encounter errors with `insightface`, ensure you have "Visual Studio Build Tools" installed with "Desktop development with C++".*

3.  **Prepare Videos**:
    Place your short training videos (approx 10s) in the `face_project/videos/` folder.
    **Filename Format**: `Name_USN.mp4` (e.g., `Shafiq_4SN23AD001.mp4`).
    *The system uses the filename (excluding extension) as the person's label.*

## Usage (Run Order)

Run these commands in order from the project root.

### 1. Extract Frames
Extracts face images from your videos.
```powershell
python face_project/scripts/extract_frames.py
```

### 2. Generate Embeddings
Converts face images into numerical vectors.
```powershell
python face_project/scripts/make_embeddings.py
```

### 3. Train Classifier
Trains the SVM model to recognize faces.
```powershell
python face_project/scripts/train_classifier.py
```

### 4. Run Attendance (Inference)
Starts the webcam, detects faces, and logs attendance.
```powershell
python face_project/scripts/infer_and_log.py
```
- Press `q` to quit.
- Attendance is saved to `detections.csv` upon exit.


### 5. Test Single Image (Optional)
Test the model on a specific image file.
```powershell
python face_project/scripts/test_single.py path/to/image.jpg
```

## Configuration

You can modify `face_project/scripts/infer_and_log.py` to change settings:

-   **`CONF_THRESH` (Default: 0.60)**: Confidence threshold. Increase (e.g., 0.75) to reduce false positives; decrease (e.g., 0.50) if faces aren't being recognized.
-   **`DEDUP_SECONDS` (Default: 10)**: Time in seconds to wait before logging the same person again.
-   **`VIDEO_SOURCE`**: Change `0` to a URL string (e.g., `"http://192.168.1.5:8080/video"`) to use an IP camera.

## Troubleshooting

-   **`insightface` install fails**: Try installing `onnxruntime` first. Ensure C++ build tools are installed.
-   **No faces detected**: Check lighting in videos. Ensure videos are in `.mp4` format.
-   **Camera not opening**: Check if another app is using the camera. Check privacy settings in Windows.
-   **PowerShell Security Error**: If you can't run scripts, try `Set-ExecutionPolicy Unrestricted -Scope Process`.

## Output Format (CSV)
The `detections.csv` file contains:
`timestamp | date | time | frame_no | usn | name | confidence | top | right | bottom | left`
