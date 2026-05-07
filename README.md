# DeepScan — Deepfake Detection Web Application

A professional Flask web application for real-time deepfake detection using **EfficientNetB0 + Bidirectional LSTM** with **InsightFace SCRFD** face detection and **106-point landmark** biometric analysis.

---

## Architecture

```
Video Upload → Frame Sampling → InsightFace SCRFD Detection
     → 106-pt Landmark Extraction → 14-dim Biometric Features
     → EfficientNetB0 (CNN) + BiLSTM (Temporal)
     → Fake/Real Prediction with Confidence Score
```

## Project Structure

```
deepfake-app/
├── app.py                          # Flask backend (full pipeline)
├── models_best_model.keras         # ← YOUR TRAINED MODEL FILE
├── requirements.txt                # Python dependencies
├── templates/
│   ├── index.html                  # Upload page (drag & drop)
│   └── result.html                 # Analysis results page
└── static/
    ├── uploads/                    # Uploaded videos (auto-created)
    ├── extracted_frames/           # Sampled frames (auto-created)
    └── extracted_faces/            # Cropped faces (auto-created)
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

**Note:** If you don't have a GPU, replace `onnxruntime-gpu` with `onnxruntime` in requirements.txt. The app will automatically fall back to CPU for InsightFace.

### 2. Place Your Model

Copy your trained model file to the project root:

```bash
cp /path/to/models_best_model.keras ./deepfake-app/
```

The model must be the exact architecture from your training notebook:
- **Input 1:** Images — `(batch, 32, 224, 224, 3)` — face crops (0–255 range)
- **Input 2:** Metadata — `(batch, 32, 14)` — biometric feature vectors
- **Input 3:** Valid mask — `(batch, 32)` — float mask (1.0 = valid, 0.0 = padded)
- **Output:** Single sigmoid — probability of being fake

### 3. Run

```bash
cd deepfake-app
python app.py
```

Open **http://localhost:5000** in your browser.

## Features

| Feature | Details |
|---|---|
| **Drag & Drop Upload** | Supports MP4, AVI, MOV, MKV, WebM up to 200 MB |
| **Adaptive Frame Sampling** | Short (<5s), medium (5–32s), long (>32s) strategies |
| **InsightFace SCRFD** | GPU-accelerated face detection with 106-pt landmarks |
| **14 Biometric Features** | EAR, MAR, symmetry, brightness, contrast, sharpness, blink rate, IOD variance |
| **Visual Evidence** | Gallery of extracted frames alongside cropped faces |
| **Per-Frame Biometrics** | Detailed table showing all computed features per face |
| **Confidence Score** | Animated progress bar with Real/Fake probability |
| **JSON API** | POST to `/api/analyze` for programmatic access |
| **Responsive UI** | Works on mobile and desktop — dark cybersecurity theme |

## API Usage

```bash
curl -X POST -F "video=@test_video.mp4" http://localhost:5000/api/analyze
```

Response:
```json
{
  "verdict": "FAKE",
  "fake_confidence": 87.34,
  "real_confidence": 12.66,
  "faces_detected": 28,
  "processing_time": 14.5,
  ...
}
```

## The 14 Biometric Features (Matching Training)

1. `ear_l` — Left Eye Aspect Ratio
2. `ear_r` — Right Eye Aspect Ratio
3. `mar` — Mouth Aspect Ratio
4. `eye_dist/W` — Inter-eye distance normalized by frame width
5. `mouth_w/W` — Mouth width normalized by frame width
6. `face_h/H` — Face height normalized by frame height
7. `nose_chin/face_h` — Nose-to-chin distance normalized by face height
8. `symmetry` — Facial symmetry score (0–1)
9. `brightness` — Mean luminance (GPU-computed)
10. `contrast` — Standard deviation of luminance (GPU-computed)
11. `lap_var` — Laplacian variance / sharpness (GPU-computed)
12. `blink_rate` — Blink frequency normalized over video duration
13. `iod_var` — Inter-ocular distance variance across frames
14. `frame_idx_norm` — Normalized frame position in sequence (0–1)
