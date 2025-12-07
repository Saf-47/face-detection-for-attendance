# Graphical Representations

This document contains visual diagrams of the Face Recognition Attendance System's architecture and logic. You can use these in your project report.

## 1. System Architecture (Data Flow)

This flowchart illustrates how data moves through the system, from video input to the final CSV log.

```mermaid
flowchart TD
    subgraph Enrollment Phase
        A[Training Videos] -->|extract_frames.py| B[Face Dataset]
        B -->|make_embeddings.py| C[InsightFace Model]
        C --> D[Embeddings Database]
    end

    subgraph Inference Phase
        E[Webcam Input] -->|Capture Frame| F[Face Detection]
        F -->|Crop & Align| G[Face Recognition]
        G -->|Generate Vector| H[Live Embedding]
        
        H -->|Cosine Similarity| I{Match Found?}
        D --> I
        
        I -- Yes (>0.50) --> J[Tracker Update]
        I -- No --> K[Mark Unknown]
        
        J --> L{Confirmed?}
        L -- Yes (>30s) --> M[Log to CSV]
        L -- No --> N[Continue Tracking]
    end
```

## 2. Sequence Diagram (Runtime Logic)

This diagram shows the step-by-step interaction between the User, the Camera, and the System components during a single frame processing loop.

```mermaid
sequenceDiagram
    participant User
    participant Camera
    participant Detector
    participant Recognizer
    participant Tracker
    participant CSV

    User->>Camera: Stands in front
    Camera->>Detector: Send Frame
    Detector->>Recognizer: Detected Face (BBox + Landmarks)
    Recognizer->>Tracker: Face Embedding (512-d)
    
    activate Tracker
    Tracker->>Tracker: Match with Reference DB
    Tracker->>Tracker: Update Track History
    Tracker->>Tracker: Check Consistency (30s window)
    
    alt Confirmed
        Tracker->>CSV: Log Attendance (Name, Time)
        Tracker-->>User: Show Green Box (Confirmed)
    else Verifying
        Tracker-->>User: Show Yellow Box (Verifying...)
    end
    deactivate Tracker
```

## 3. State Machine (Person Tracking)

This diagram represents the lifecycle of a tracked person ("PersonTrack" object).

```mermaid
stateDiagram-v2
    [*] --> Tracking: New Face Detected
    
    state Tracking {
        [*] --> Accumulating: Add Frame to History
        Accumulating --> CheckThresholds: Is Similarity > 0.5?
        CheckThresholds --> Accumulating: No
        CheckThresholds --> ValidFrame: Yes
    }

    Tracking --> Confirmed: Consistency > 60% (after 30s)
    Tracking --> Lost: Face Lost (> 2s gap)
    
    Confirmed --> [*]: Logged & Finished
    Lost --> [*]: Delete Track
```

## 4. Class Diagram

This shows the structure of the Python classes used in `infer_and_log.py`.

```mermaid
classDiagram
    class ReferenceDatabase {
        +embeddings: numpy.array
        +labels: list
        +find_match(embedding) (name, score)
    }

    class PersonTrack {
        +track_id: int
        +history: list
        +status: str
        +update(embedding, bbox)
        +check_confirmation() bool
    }

    class Tracker {
        +tracks: list[PersonTrack]
        +update(faces, db)
    }

    class CsvLogger {
        +filename: str
        +append_row(data)
    }

    Tracker "1" *-- "*" PersonTrack : manages
    Tracker ..> ReferenceDatabase : uses
    Tracker ..> CsvLogger : uses
```
