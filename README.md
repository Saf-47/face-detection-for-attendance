# Face Recognition Attendance System

A professional, privacy-focused, and efficient face recognition attendance system built with Python, InsightFace, and SVM. 

This project is designed to run locally on a CPU, ensuring data privacy while delivering high-accuracy recognition for verifying student attendance.

---

## 🚀 Key Features

*   **High Accuracy**: Uses `InsightFace` (ArcFace) for state-of-the-art face embeddings.
*   **Privacy-First**: No data is sent to the cloud. All face data and models are stored locally.
*   **Real-Time Processing**: optimized for CPU inference to work on standard laptops.
*   **Automated Logging**: Automatically logs verified attendance into a CSV file with timestamps.
*   **Professional Architecture**: Modular code structure separating data extraction, training, and inference.

## 📚 Documentation

For deep dives into the system, check out our detailed documentation in the `docs/` folder:
*   [**Project Documentation**](docs/PROJECT_DOCUMENTATION.md): Complete architecture and logic explanation.
*   [**Methodology**](docs/METHODOLOGY.md): How the recognition pipeline works.
*   [**Results & Limitations**](docs/RESULTS_AND_LIMITATIONS.md): Performance analysis.
*   [**Graphical Representations**](docs/GRAPHICAL_REPRESENTATIONS.md): Flowcharts and diagrams.

---

## 🛠️ Quick Start Guide

Follow these steps to set up and run the project on your machine.

### Prerequisites
-   Python 3.8 or higher.
-   Visual Studio Build Tools (for C++ compilation required by some libraries).

### 1. Installation
Clone the repository and install the required dependencies.

```powershell
# Clone the repo
git clone https://github.com/Saf-47/face-detection-for-attendance.git
cd face-detection-for-attendance

# Install dependencies
pip install -r face_project/requirements.txt
```

### 2. Prepare Your Data (Videos)
The system learns from short video clips of each person.
1.  Record a **10-second video** of the person's face.
2.  Name the file: `Name_USN.mp4` (e.g., `Shafiq_4SN23AD001.mp4`).
3.  Place these videos in the `face_project/videos/` folder.

### 3. Build the Model (3 Steps)
Run these commands in order to process the videos and train the recognition model.

**Step A: Extract Faces**
Crops face images from your videos for the dataset.
```powershell
python face_project/scripts/extract_frames.py
```

**Step B: Create Embeddings**
Converts face images into mathematical vectors (embeddings).
```powershell
python face_project/scripts/make_embeddings.py
```

**Step C: Train Classifier**
Trains a lightweight Support Vector Machine (SVM) to recognize these embeddings.
```powershell
python face_project/scripts/train_classifier.py
```

### 4. Run Attendance
Start the live webcam system to take attendance.
```powershell
python face_project/scripts/infer_and_log.py
```
*   **Quit**: Press `q` to exit.
*   **Logs**: Check `detections.csv` for the attendance record.

---

## ⚙️ Configuration

You can customize the system by editing `face_project/scripts/infer_and_log.py`:

| Setting | Default | Description |
| :--- | :--- | :--- |
| `CONF_THRESH` | `0.60` | Confidence required to verify a face. Increase to reduce false positives. |
| `DEDUP_SECONDS` | `10` | Seconds to wait before logging the same person again to avoid duplicate entries. |
| `VIDEO_SOURCE` | `0` | Camera input. Use `0` for webcam or an RTSP URL for IP cameras. |

---

## ❓ Troubleshooting

*   **`insightface` Installation Error**: This often happens if C++ build tools are missing. Try installing `onnxruntime` separately or ensure you have "Desktop development with C++" installed via Visual Studio Installer.
*   **No Faces Detected**: Ensure your training videos have good lighting and the face is clearly visible.
*   **Permission Errors**: If PowerShell blocks script execution, run `Set-ExecutionPolicy Unrestricted -Scope Process`.

---

## 🛡️ License & Privacy
This project is open-source. For privacy reasons, the public repository **does not contain any training data or trained models**. You must generate these locally using your own videos.
