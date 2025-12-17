# AI Training Application

A full-stack application for speech-to-text transcription and object detection with fine-tuning capabilities.

## Features

### Frontend (Angular 21)
- **Listen Section**: Speech-to-text using Hugging Face Whisper Tiny model
  - Microphone recording with start/stop controls
  - Language selection (English/Italian)
  - Real-time transcription display

- **See Section**: Object detection using YOLO model
  - Camera access and live video feed
  - Real-time object detection with bounding boxes
  - Manual corrections interface for fine-tuning
  - Send corrected data to backend for model training

### Backend (Python FastAPI)
- YOLO model serving for detection
- Fine-tuning capabilities with uploaded training data
- Training logs storage and retrieval
- SQLite database for data persistence

## Technology Stack

### Frontend
- Angular 21
- Hugging Face Transformers.js
- TypeScript
- HTML5 Camera API
- MediaRecorder API

### Backend
- Python 3.x
- FastAPI
- Ultralytics YOLO
- SQLite
- PIL/Pillow

## Installation

### Prerequisites
- **Docker and Docker Compose** (recommended for easiest setup)
- **Local Development**: Node.js 20+ and Python 3.11+

### Docker Setup (Recommended)
Clone the repository and run:
```bash
./start.sh
```

### Alternative: Using Makefile
```bash
make help          # Show available commands
make dev           # Start development environment
make prod          # Start production environment
make logs          # View all logs
make stop          # Stop all services
make clean         # Clean up containers and volumes
```

### Local Development Setup

**Frontend Setup:**
```bash
cd frontend/training
npm install
```

**Backend Setup:**
```bash
cd backend
pip install -r requirements.txt
```

## Running the Application

### Docker Compose (Recommended)

**Production Build:**
```bash
./start.sh
```
Frontend runs on http://localhost:4223, Backend on http://localhost:8000

**Development Build (with hot reload):**
```bash
./start.sh --dev
```
Frontend runs on http://localhost:4223, Backend on http://localhost:8000

### Manual Development Setup

**Backend:**
```bash
cd backend
python main.py
```
Backend runs on http://localhost:8000

**Frontend:**
```bash
cd frontend/training
npm start
```
Frontend runs on http://localhost:4223

## Usage

1. **Speech-to-Text** (`/listen`):
   - Select language (English/Italian)
   - Click "Start Recording" and speak
   - Click "Stop Recording" to transcribe
   - View transcription in the transcript area

2. **Object Detection** (`/see`):
   - Click "Start Camera" to enable camera access
   - Click "Detect Objects" to run YOLO detection
   - View bounding boxes with labels and confidence scores
   - Click "Edit Boxes" to enter correction mode
   - Use "Send to Backend" to upload corrected data for fine-tuning

3. **Training Logs** (`/logs`):
   - View training history and metrics
   - Refresh to see latest training progress

## API Endpoints

### Backend API
- `GET /health`: Health check
- `POST /detect`: Object detection on uploaded images
- `POST /upload-training-data`: Upload training data
- `POST /train`: Start fine-tuning
- `GET /training-logs`: Get training logs
- `GET /whisper-status`: Check Whisper status

## Architecture

- **Frontend**: Browser-based ML inference using Transformers.js
- **Backend**: Python API server for model training and serving
- **Data Flow**: Frontend captures media → processes locally → sends corrections to backend for training
- **Training**: YOLO models can be fine-tuned with user-corrected bounding boxes

## Security Notes

- CORS enabled for development (restrict in production)
- Camera and microphone permissions granted by browser
- All data processed locally when possible (privacy-first approach)

## Browser Requirements

- Modern browser with WebRTC support (camera/microphone)
- WebAssembly support for ML models
- HTTPS required for camera/microphone access (or localhost)

## Development

### Adding New Languages
Update the language selection in `listen.component.html` and the processing logic in `listen.component.ts`.

### Extending Object Detection
Modify the YOLO model loading in `see.component.ts` to support additional models.

### Training Pipeline
Update `backend/main.py` to implement full training cycles with proper data annotation pipelines.
