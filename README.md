# Blurify AI 🛡️

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-18.0+-61DAFB.svg?logo=react)](https://reactjs.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.103+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com/)

A robust, production-quality pipeline for detecting and dynamically blurring human faces in short video clips and static images. **Blurify AI** is designed to handle difficult real-world conditions like partial faces, side-profiles, tilted faces, occlusions, and fast movement.

## ✨ Features
- **Modern Web Interface**: A sleek, dark-mode React UI with drag-and-drop support and live settings previews.
- **Multi-Detector Architecture**: Supports SCRFD (InsightFace), RetinaFace, YuNet (OpenCV), and MediaPipe.
- **Robust Tracking**: Uses ByteTrack for maintaining face identities across video frames, eliminating "blur flickering".
- **Dynamic Blurring**: Choose between smooth Gaussian, privacy-compliant Pixelation, or Confidence-Adaptive blurs applied via natural elliptical masks.
- **Optimized Video Processing**: Smart frame-skipping and temporal smoothing for rapid video rendering while preserving original audio tracks.

---

## 📋 System Requirements

To run Blurify AI, your system must meet the following requirements:

### 1. General Software
- **Python 3.9 or higher**: Required for the backend AI engine.
- **Node.js (v16+)**: Required for running the Web UI frontend.

### 2. External Dependencies
- **FFmpeg**: Required for preserving audio in processed videos.
  - **Windows**: Install via `winget install ffmpeg` or download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to your system PATH.
  - **macOS**: `brew install ffmpeg`
  - **Linux (Ubuntu/Debian)**: `sudo apt install ffmpeg`

### 3. Hardware (Optional but Recommended)
- **CPU**: Works out-of-the-box on any modern CPU (utilizing the OpenCV YuNet detector fallback).
- **GPU (NVIDIA)**: For maximum performance and accuracy (SCRFD/RetinaFace), an NVIDIA GPU with CUDA installed is highly recommended. *(Requires replacing `onnxruntime` with `onnxruntime-gpu` in your environment).*

---

## 🚀 Installation

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/blurify-ai.git
cd blurify-ai
```

### Step 2: Install Backend Dependencies (Python)
It is highly recommended to use a virtual environment:
```bash
# Create and activate virtual environment (Windows)
python -m venv .venv
.venv\Scripts\activate

# Install requirements
pip install -r requirements.txt
```

### Step 3: Install Frontend Dependencies (Node.js)
```bash
cd ui
npm install
cd ..
```

---

## 💻 Running the Application

Blurify AI can be operated via the modern Web UI (recommended) or the highly configurable Command Line Interface (CLI).

### Option A: Modern Web UI (Recommended)

**For Windows Users:**
The easiest way to start the application is to simply double-click the `run_windows.bat` file located in the project folder. This will automatically set up your environment, start both the API and UI servers, and open your web browser!

**For macOS / Linux (or Manual Start):**
You need two terminal windows open to run the full stack.

**Terminal 1: Start the AI Backend**
```bash
python -m src.api
```
*(The API will start at `http://127.0.0.1:8000`)*

**Terminal 2: Start the React Frontend**
```bash
cd ui
npm run dev
```
*(The UI will start at `http://localhost:5173`)*

**Next**: Open your web browser and navigate to `http://localhost:5173`. You can drag and drop media, adjust blur intensity, and download the anonymized results directly to your machine.

---

### Option B: Command Line Interface (CLI)

For batch processing or headless servers, the CLI is extremely powerful.

**Process a Single Image:**
```bash
python main.py -i input/photo.jpg -o output/photo_blurred.jpg
```

**Process a Video:**
```bash
python main.py -i input/video.mp4 -o output/video_blurred.mp4
```

**Process an Entire Directory:**
```bash
python main.py -i input/ -o output/
```

#### CLI Presets
You can use presets to quickly apply optimized settings without manually tuning parameters:
- `fast`: Sacrifices accuracy for speed. Uses YuNet, frame skipping, no smoothing.
- `accurate`: Maximum accuracy. Uses SCRFD with low thresholds.
- `privacy`: GDPR-compliant preset. Uses pixelation blur and aggressive padding.

```bash
python main.py -i input/video.mp4 --preset privacy
```

#### Advanced CLI Configuration
Adjust blur styles and detection thresholds manually:
```bash
python main.py -i input/video.mp4 --blur pixelate --blur-strength 151 --confidence 0.6
```

---

## 🏗️ Architecture Overview
1. **API Layer (`src/api.py`)**: FastAPI wrapper handling asynchronous uploads/downloads.
2. **Detection (`src/detector.py`)**: ONNX-based models extract `FaceDetection` objects.
3. **Tracking (`src/tracker.py`)**: A simplified ByteTrack implementation recovers missed detections.
4. **Blur (`src/blur.py`)**: Applies masking logic dynamically.
5. **Pipelines (`src/video_pipeline.py`)**: Coordinates frame extraction, tracking, and FFmpeg audio muxing.
