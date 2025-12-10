from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn
import os
import json
import datetime
import io
from pathlib import Path
import shutil
# Set headless mode and environment variables
os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ['XDG_RUNTIME_DIR'] = '/tmp/xdg-runtime'
os.environ['DISPLAY'] = ':0'

# Creating a mock backend for testing without GUI dependencies
# We'll import actual ML libraries only when needed
import sqlite3
from contextlib import contextmanager

# Create data directories following the new structure
DATA_DIR = Path("data")
YOLO_DATA_DIR = DATA_DIR / "vision"
WHISPER_DATA_DIR = DATA_DIR / "speech"

# Task directories for YOLO
DIGITS_DIR = YOLO_DATA_DIR / "digits"
DIGITS_WAITING_DIR = DIGITS_DIR / "waiting"
DIGITS_TRAINED_DIR = DIGITS_DIR / "trained"

COLORS_DIR = YOLO_DATA_DIR / "colors"
COLORS_WAITING_DIR = COLORS_DIR / "waiting"
COLORS_TRAINED_DIR = COLORS_DIR / "trained"

# Task directories for Whisper
WHISPER_WAITING_DIR = WHISPER_DATA_DIR / "waiting"
WHISPER_TRAINED_DIR = WHISPER_DATA_DIR / "trained"

# Create all directories
for dir_path in [
    DATA_DIR, YOLO_DATA_DIR, WHISPER_DATA_DIR,
    DIGITS_DIR, DIGITS_WAITING_DIR, DIGITS_TRAINED_DIR,
    COLORS_DIR, COLORS_WAITING_DIR, COLORS_TRAINED_DIR,
    WHISPER_WAITING_DIR, WHISPER_TRAINED_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Models directory structure
MODELS_DIR = Path("models")
YOLO_MODELS_DIR = MODELS_DIR / "yolo" / "yolov8n"
LORAS_DIR = YOLO_MODELS_DIR / "loras"
TRAIN_LOG_DIR = Path("train_logs")
TRAINED_DIR = Path("trained")

for dir_path in [MODELS_DIR, YOLO_MODELS_DIR, LORAS_DIR, TRAIN_LOG_DIR, TRAINED_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Initialize database (service-agnostic)
def init_db():
    conn = sqlite3.connect('db/vision.db')
    cursor = conn.cursor()

    # Create training logs table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            epoch INTEGER,
            accuracy REAL,
            loss REAL,
            val_accuracy REAL,
            val_loss REAL,
            metadata TEXT
        )
    ''')

    # Create annotations table (can be used by multiple services)
    # First, check if table exists and create if not
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS annotations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            service TEXT NOT NULL DEFAULT 'vision',
            filename TEXT NOT NULL,
            data TEXT NOT NULL,
            labels TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    ''')

    # Add service column if it doesn't exist (migration for existing databases)
    try:
        cursor.execute("ALTER TABLE annotations ADD COLUMN service TEXT NOT NULL DEFAULT 'vision'")
    except sqlite3.OperationalError as e:
        if "duplicate column name" not in str(e):
            print(f"Error adding service column: {e}")

    conn.commit()
    conn.close()

init_db()

app = FastAPI(title="AI Training Backend", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== SERVICE INITIALIZATION =====
# Initialize Whisper and YOLO services with separate directories

print("🔊 Initializing Whisper service...")
from speech_service import SpeechService
whisper_service = SpeechService(
    models_dir="models/speech",
    data_dir="data/speech"
)

print("👁️ Initializing Vision service...")
from vision_service import VisionService
yolo_service = VisionService(
    models_dir="models/vision",
    data_dir="data/vision"
)

class TrainingData(BaseModel):
    imageData: str  # Base64 encoded image
    detections: List[Dict[str, Any]]  # Bounding boxes and labels

class TrainingRequest(BaseModel):
    epochs: int = 10
    batch_size: int = 16
    val_split: float = 0.2
    use_lora: bool = False
    learning_rate: float = 0.001

class SpecializedTrainingRequest(BaseModel):
    training_type: str  # "digits" or "colors"
    epochs: int = 60
    lora_rank: int = 8
    learning_rate: float = 0.001

class SpecializedTrainingData(BaseModel):
    training_type: str  # "digits" or "colors"
    imageData: str  # Base64 encoded image
    detections: List[Dict[str, Any]]  # Bounding boxes and labels

@contextmanager
def get_db_connection():
    conn = sqlite3.connect('db/vision.db')
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

@app.on_event("startup")
async def startup_event():
    # YOLO model is loaded automatically when yolo_service is initialized
    pass

@app.get("/health")
async def health_check():
    return {"status": "healthy", "models": ["YOLO"]}

@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """Detect objects in uploaded image using YOLO service"""
    try:
        contents = await file.read()
        return await yolo_service.detect_objects(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train")
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start YOLO model training using the YOLO service"""
    background_tasks.add_task(run_yolo_training,
                            epochs=request.epochs,
                            batch_size=request.batch_size,
                            val_split=request.val_split,
                            use_lora=request.use_lora,
                            learning_rate=request.learning_rate)

    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO training_logs (timestamp, metadata) VALUES (?, ?)",
            (datetime.datetime.now().isoformat(), json.dumps({
                "service": "yolo",
                "type": "training_started",
                "epochs": request.epochs,
                "batch_size": request.batch_size,
                "val_split": request.val_split,
                "use_lora": request.use_lora,
                "learning_rate": request.learning_rate
            }))
        )

    return {"message": "YOLO Training started in background"}

def run_yolo_training(epochs: int, batch_size: int, val_split: float, use_lora: bool = False, learning_rate: float = 0.001):
    """Train YOLO model using the YOLO service"""
    try:
        print(f"Starting YOLO training: epochs={epochs}, use_lora={use_lora}, lr={learning_rate}")

        # Use YOLO service to handle training
        result = yolo_service.train_model(
            epochs=epochs,
            batch_size=batch_size,
            val_split=val_split,
            use_lora=use_lora,
            learning_rate=learning_rate
        )

        if result.get('error'):
            print(f"YOLO training failed: {result['error']}")
            log_training_error(result['error'])
        else:
            print("YOLO training completed successfully!")
            log_training_completion()

    except Exception as e:
        print(f"YOLO training failed: {e}")
        log_training_error(str(e))

def run_specialized_lora_training(training_type: str, epochs: int, lora_rank: int, learning_rate: float):
    """Train specialized LoRA adapter for digits or colors"""
    try:
        print(f"Starting specialized LoRA training: type={training_type}, epochs={epochs}, lora_rank={lora_rank}, lr={learning_rate}")

        # Use YOLO service to handle specialized LoRA training with frozen backbone
        result = yolo_service.train_specialized_lora(
            training_type=training_type,
            epochs=epochs,
            lora_rank=lora_rank,
            learning_rate=learning_rate
        )

        if result.get('error'):
            print(f"Specialized LoRA training failed: {result['error']}")
            log_specialized_training_error(training_type, result['error'])
        else:
            print(f"Specialized LoRA training completed successfully for {training_type}!")
            save_lora_adapter(training_type)
            log_specialized_training_completion(training_type)
            yolo_service.move_specialized_training_images(training_type)

    except Exception as e:
        print(f"Specialized LoRA training failed: {e}")
        log_specialized_training_error(training_type, str(e))

def save_lora_adapter(training_type: str):
    """Save the trained LoRA adapter"""
    try:
        # Get the trained adapter from YOLO service and save to loras/ directory
        adapter_data = yolo_service.export_lora_adapter()

        if adapter_data:
            adapter_path = LORAS_DIR / f"{training_type}.safetensors"
            with open(adapter_path, 'wb') as f:
                f.write(adapter_data)
            print(f"Saved LoRA adapter for {training_type}: {adapter_path}")
        else:
            print(f"Warning: No adapter data received for {training_type}")

    except Exception as e:
        print(f"Error saving LoRA adapter for {training_type}: {e}")

def log_specialized_training_completion(training_type: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO training_logs (timestamp, metadata) VALUES (?, ?)",
            (
                datetime.datetime.now().isoformat(),
                json.dumps({"service": "yolo_lora", "type": "training_completed", "specialization": training_type})
            )
        )

def log_specialized_training_error(training_type: str, error_message: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO training_logs (timestamp, metadata) VALUES (?, ?)",
            (
                datetime.datetime.now().isoformat(),
                json.dumps({"service": "yolo_lora", "type": "training_error", "specialization": training_type, "error": error_message})
            )
        )

def create_synthetic_training_data():
    """Prepare uploaded training images for YOLO training"""
    try:
        # Get all annotations from database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM annotations")
            rows = cursor.fetchall()

        if not rows:
            print("No training data found in database")
            return

        # Create train directory structure (images only, labels stored in database)
        train_images_dir = DATA_DIR / "train" / "images"
        train_images_dir.mkdir(parents=True, exist_ok=True)

        print(f"Preparing {len(rows)} training images (labels stored in database)...")

        for row in rows:
            image_filename = row[1]  # image_filename
            bbox_data_str = row[2]   # bbox_data (JSON)
            labels_str = row[3]      # labels (JSON)

            # Source image path (now from images/ subdirectory)
            source_path = DATA_DIR / "images" / image_filename
            if not source_path.exists():
                print(f"Warning: Source image {source_path} not found, skipping")
                continue

            # Destination image path (labels remain in database only)
            dest_image_path = train_images_dir / image_filename
            shutil.copy2(str(source_path), str(dest_image_path))

            # Labels are already stored in database - no .txt files needed
            try:
                bbox_data = json.loads(bbox_data_str)
                labels_data = json.loads(labels_str)
                print(f"Prepared image: {image_filename} with {len(bbox_data)} annotations (stored in DB)")

            except json.JSONDecodeError as e:
                print(f"Error parsing annotation data for {image_filename}: {e}")
                continue

        print(f"Successfully prepared {len(rows)} training images")

    except Exception as e:
        print(f"Error preparing training data: {e}")
        raise

def move_processed_images():
    """Move all training images from data/ to data/processed/ to avoid reuse"""
    try:
        moved_count = 0

        # Find all image files in data directory (uploaded training images)
        for file_path in DATA_DIR.glob("*.jpg"):
            if file_path.is_file():
                # Move to processed directory
                processed_path = PROCESSED_DATA_DIR / file_path.name
                shutil.move(str(file_path), str(processed_path))
                moved_count += 1
                print(f"Moved processed image: {file_path.name} -> processed/")

        # Find all image files in data/train directory as well
        train_dir = DATA_DIR / "train"
        if train_dir.exists():
            for file_path in train_dir.rglob("*.jpg"):
                if file_path.is_file():
                    processed_path = PROCESSED_DATA_DIR / file_path.name
                    shutil.move(str(file_path), str(processed_path))
                    moved_count += 1
                    print(f"Moved processed image: {file_path.name} -> processed/")

        print(f"Successfully moved {moved_count} images to processed directory")

    except Exception as e:
        print(f"Warning: Failed to move processed images: {e}")

def cleanup_old_models(current_model: Path):
    """Keep only yolov8n.pt and the most recent trained_*.pt model"""
    try:
        # Find all trained_*.pt files
        trained_models = list(MODELS_DIR.glob("trained_*.pt"))
        if not trained_models:
            return

        # Sort by modification time (newest first)
        trained_models.sort(key=lambda p: p.stat().st_mtime, reverse=True)

        # Keep only the first one (most recent) and remove others
        kept_model = trained_models[0]
        deleted_count = 0

        for model_file in trained_models[1:]:  # Skip the first (most recent)
            if model_file != current_model:  # Make sure we don't delete the current one
                model_file.unlink()
                deleted_count += 1
                print(f"Cleaned up old model: {model_file.name}")

        if deleted_count > 0:
            print(f"Model cleanup: kept {kept_model.name}, deleted {deleted_count} old models")

    except Exception as e:
        print(f"Warning: Model cleanup failed: {e}")

def log_training_metrics(epoch: int, metrics: Dict[str, float]):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO training_logs (timestamp, epoch, accuracy, loss, val_accuracy, val_loss, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.datetime.now().isoformat(),
                epoch + 1,
                metrics.get("accuracy"),
                metrics.get("loss"),
                metrics.get("val_accuracy"),
                metrics.get("val_loss"),
                json.dumps({"service": "yolo", "type": "epoch_metrics"})
            )
        )

def log_training_completion():
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO training_logs (timestamp, metadata) VALUES (?, ?)",
            (
                datetime.datetime.now().isoformat(),
                json.dumps({"service": "yolo", "type": "training_completed"})
            )
        )

def log_training_error(error_message: str):
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO training_logs (timestamp, metadata) VALUES (?, ?)",
            (
                datetime.datetime.now().isoformat(),
                json.dumps({"service": "yolo", "type": "training_error", "error": error_message})
            )
        )

@app.post("/upload-training-data")
async def upload_training_data(training_data: TrainingData):
    try:
        # Save image and annotations
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save image to images subdirectory
        import base64
        images_dir = DATA_DIR / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        image_data = base64.b64decode(training_data.imageData.split(',')[1])
        image_path = images_dir / f"image_{timestamp}.jpg"

        with open(image_path, "wb") as f:
            f.write(image_data)

        # Store in service-aware database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO annotations (service, filename, data, labels, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    "vision",
                    f"image_{timestamp}.jpg",
                    json.dumps({"bboxes": [d["bbox"] for d in training_data.detections]}),
                    json.dumps([d.get("label", "unknown") for d in training_data.detections]),
                    timestamp
                )
            )
            print(f"Inserted vision annotation into database: image_{timestamp}.jpg with {len(training_data.detections)} detections")

        return {"message": "Training data uploaded successfully", "file_id": timestamp}

    except Exception as e:
        print(f"Error uploading training data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-speech-training-data")
async def upload_speech_training_data(
    audio_file: UploadFile = File(...),
    language: str = "en",
    transcript: str = ""
):
    """Upload audio and transcript for speech recognition training"""
    try:
        # Read the audio file
        audio_bytes = await audio_file.read()

        # Process and store the speech training data
        result = await whisper_service.process_speech_training_data(
            audio_bytes, language, transcript
        )

        return {
            "message": "Speech training data uploaded successfully",
            "audio_path": result["audio_path"],
            "transcript_path": result["transcript_path"],
            "language": result["language"]
        }

    except Exception as e:
        print(f"Speech training data upload failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/training-logs")
async def get_training_logs():
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            # Only return YOLO training logs, filter out whisper logs that may have been accidentally stored
            cursor.execute("""
                SELECT * FROM training_logs
                WHERE json_extract(metadata, '$.service') IN ('yolo', NULL)
                ORDER BY timestamp DESC
            """)
            logs = cursor.fetchall()

            result = []
            for log in logs:
                result.append({
                    "id": log[0],
                    "timestamp": log[1],
                    "epoch": log[2],
                    "accuracy": log[3],
                    "loss": log[4],
                    "val_accuracy": log[5],
                    "val_loss": log[6],
                    "metadata": json.loads(log[7]) if log[7] else None
                })

            return {"logs": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe-audio")
async def transcribe_audio(audio_file: UploadFile = File(...)):
    """Transcribe audio using Whisper service"""
    try:
        result = await whisper_service.transcribe_audio(audio_file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/whisper-status")
async def get_whisper_status():
    """Get Whisper service status"""
    return whisper_service.get_status()

@app.get("/speech-training-count")
async def get_speech_training_count():
    """Get count of uploaded speech training samples"""
    try:
        # Count files in whisper data directory
        speech_files = list(whisper_service.data_dir.glob("*.wav"))
        return {"count": len(speech_files), "message": "Speech training data count"}
    except Exception as e:
        return {"error": str(e), "count": 0}

@app.get("/whisper-training-status")
async def get_whisper_training_status():
    """Get status of Whisper training data and settings"""
    try:
        status = whisper_service.get_training_status()
        return status
    except Exception as e:
        return {"error": str(e)}

@app.get("/whisper-training-status-details")
async def get_whisper_training_status_details():
    """Get detailed Whisper training status for frontend polling"""
    return whisper_service.get_training_status_details()

@app.post("/whisper-fine-tune-lora")
async def start_whisper_lora_fine_tuning(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start LoRA fine-tuning for Whisper model"""
    try:
        # Check if we have training data
        status = whisper_service.get_training_status()
        if status.get("training_samples", 0) == 0:
            raise HTTPException(
                status_code=400,
                detail="No speech training data found. Upload audio samples first."
            )

        # Determine language (use first available language from data)
        language = status.get("available_languages", ["en"])[0] if status.get("available_languages") else "en"

        # Start LoRA fine-tuning in background
        background_tasks.add_task(
            whisper_service.fine_tune_lora,
            language=language,
            epochs=request.epochs,
            output_dir=f"whisper_models/whisper-lora-{language}"
        )

        return {
            "message": f"Whisper LoRA fine-tuning started for language: {language}",
            "language": language,
            "epochs": request.epochs,
            "status": "background_processing"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-whisper-lora-adapter")
async def load_whisper_lora_adapter(adapter_path: str):
    """Load a previously fine-tuned LoRA adapter for Whisper inference"""
    try:
        success = whisper_service.load_lora_adapter(adapter_path, "unknown")
        if success:
            return {"message": "LoRA adapter loaded successfully", "adapter_path": adapter_path}
        else:
            raise HTTPException(status_code=500, detail="Failed to load LoRA adapter")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading LoRA adapter: {str(e)}")

@app.get("/models/{model_name}/{filename}")
async def get_model_file(model_name: str, filename: str):
    """Serve YOLO model files locally to avoid HuggingFace CORS issues"""
    # Create models directory if it doesn't exist
    models_dir = MODELS_DIR / model_name
    models_dir.mkdir(exist_ok=True)

    file_path = models_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Model file {filename} not found locally. Please download it first.")

    return FileResponse(
        path=file_path,
        media_type="application/octet-stream",
        filename=filename
    )

@app.get("/available-tags")
async def get_available_tags(mode: str = "general"):
    """Return the list of available object detection class tags based on mode"""
    if mode == "digits":
        tags = [str(i) for i in range(10)]
        description = "Digits 0-9 for digit recognition"
    elif mode == "colors":
        tags = [
            "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
            "black", "white", "gray", "cyan", "magenta", "lime", "teal"
        ]
        description = "Color classes for color detection"
    else:
        # COCO dataset class names (80 classes)
        tags = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
            "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        ]
        description = "COCO dataset object classes"

    return {"tags": tags, "description": description, "mode": mode}

@app.get("/tags/digits")
async def get_digits_tags():
    """Return digit tags for digit recognition"""
    tags = [str(i) for i in range(10)]
    return {"tags": tags, "description": "Digits 0-9 for digit recognition", "mode": "digits"}

@app.get("/tags/colors")
async def get_colors_tags():
    """Return color tags for color detection"""
    tags = [
        "red", "blue", "green", "yellow", "orange", "purple", "pink", "brown",
        "black", "white", "gray", "cyan", "magenta", "lime", "teal"
    ]
    return {"tags": tags, "description": "Color classes for color detection", "mode": "colors"}

@app.post("/reset-model")
async def reset_model():
    """Reset the YOLO model back to the original base YOLOv8n model using service method"""
    try:
        success = yolo_service.reset_to_base()
        if success:
            return {
                "message": "Model reset to base YOLOv8n successfully",
                "status": "base_model_loaded"
            }
        else:
            return {
                "error": "Failed to reset YOLO model",
                "status": "reset_failed"
            }
    except Exception as e:
        return {
            "error": f"Failed to reset model: {str(e)}",
            "status": "reset_failed"
        }

@app.get("/model-status")
async def get_model_status():
    """Check if YOLO model is loaded and ready using service status"""
    try:
        status = yolo_service.get_status()
        return {
            "loaded": status.get("loaded", False),
            "model_type": status.get("model_type", "unknown"),
            "source": status.get("source", "unknown"),
            "message": "Service status retrieved" if status.get("loaded") else "Model not loaded"
        }
    except Exception as e:
        return {
            "loaded": False,
            "model_type": "error",
            "source": "unknown",
            "message": f"Error getting model status: {str(e)}"
        }

@app.post("/upload-specialized-training-data")
async def upload_specialized_training_data(data: SpecializedTrainingData):
    """Upload training data for specialized LoRA training (digits or colors)"""
    try:
        training_type = data.training_type
        if training_type not in ["digits", "colors"]:
            raise HTTPException(status_code=400, detail="Invalid training type. Must be 'digits' or 'colors'")

        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

        # Create task-specific waiting subdirectory
        if training_type == "digits":
            task_dir = DIGITS_WAITING_DIR
        else:  # colors
            task_dir = COLORS_WAITING_DIR

        task_dir.mkdir(parents=True, exist_ok=True)

        # Save image to task-specific waiting subdirectory
        import base64
        image_data = base64.b64decode(data.imageData.split(',')[1])
        image_path = task_dir / f"specialized_{training_type}_{timestamp}.jpg"

        with open(image_path, "wb") as f:
            f.write(image_data)

        # Store in service-aware database
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO annotations (service, filename, data, labels, timestamp) VALUES (?, ?, ?, ?, ?)",
                (
                    "vision",  # service
                    image_path.name,  # filename
                    json.dumps({**{"training_type": training_type}, **{"bboxes": [d["bbox"] for d in data.detections]}}),
                    json.dumps([d.get("label", "unknown") for d in data.detections]),
                    timestamp
                )
            )

        print(f"Inserted specialized {training_type} annotation into database: {image_path.name} with {len(data.detections)} detections")
        print(f"Image saved to waiting pool: {image_path}")

        return {"message": f"{training_type.title()} training data uploaded successfully", "file_id": timestamp}

    except Exception as e:
        print(f"Error uploading specialized training data: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-lora-specialized")
async def train_lora_specialized(request: SpecializedTrainingRequest, background_tasks: BackgroundTasks):
    """Train specialized LoRA adapter for digits or colors"""
    try:
        training_type = request.training_type
        if training_type not in ["digits", "colors"]:
            raise HTTPException(status_code=400, detail="Invalid training type. Must be 'digits' or 'colors'")

        # Check if we have training data for this type
        data_count = get_specialized_training_count(training_type)
        if data_count == 0:
            raise HTTPException(status_code=400, detail=f"No {training_type} training data found. Upload training data first.")

        # Start specialized LoRA training
        background_tasks.add_task(
            run_specialized_lora_training,
            training_type=training_type,
            epochs=request.epochs,
            lora_rank=request.lora_rank,
            learning_rate=request.learning_rate
        )

        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO training_logs (timestamp, metadata) VALUES (?, ?)",
                (datetime.datetime.now().isoformat(), json.dumps({
                    "service": "yolo_lora",
                    "type": "training_started",
                    "training_type": training_type,
                    "epochs": request.epochs,
                    "lora_rank": request.lora_rank,
                    "learning_rate": request.learning_rate
                }))
            )

        return {"message": f"LoRA training started for {training_type} specialization"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/load-lora-adapter")
async def load_lora_adapter(training_type: str):
    """Load a specialized LoRA adapter for inference"""
    try:
        if training_type not in ["digits", "colors"]:
            raise HTTPException(status_code=400, detail="Invalid training type. Must be 'digits' or 'colors'")

        adapter_path = LORAS_DIR / f"{training_type}.safetensors"

        if not adapter_path.exists():
            raise HTTPException(status_code=404, detail=f"LoRA adapter for {training_type} not found. Train it first.")

        # Load the adapter in YOLO service
        success = yolo_service.load_lora_adapter(str(adapter_path))
        if success:
            global using_base_model
            using_base_model = False
            return {"message": f"'{training_type}' LoRA adapter loaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to load LoRA adapter")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/specialized-training-count/{training_type}")
async def get_specialized_training_count_endpoint(training_type: str):
    """Get count of specialized training data for a specific type"""
    if training_type not in ["digits", "colors"]:
        raise HTTPException(status_code=400, detail="Invalid training type. Must be 'digits' or 'colors'")

    count = get_specialized_training_count(training_type)
    return {"count": count, "training_type": training_type}

def get_specialized_training_count(training_type: str) -> int:
    """Get count of training data for a specific type"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT COUNT(*) FROM annotations
                WHERE service = 'vision' AND json_extract(data, '$.training_type') = ?
            """, (training_type,))
            count = cursor.fetchone()[0]
            return count
    except Exception:
        return 0

@app.get("/debug-annotations")
async def debug_annotations():
    """Debug endpoint to see what's in the annotations table"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM annotations")
            rows = cursor.fetchall()
            return {"count": len(rows), "rows": rows}
    except Exception as e:
        return {"error": str(e)}

@app.get("/training-queue-status")
async def get_training_queue_status():
    """Get status of images waiting for training"""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM annotations")
            image_count = cursor.fetchone()[0]

            # Also get total training attempts
            cursor.execute("SELECT COUNT(*) FROM training_logs WHERE json_extract(metadata, '$.type') = 'training_started'")
            training_sessions = cursor.fetchone()[0]

            # Get last training completion info
            cursor.execute("""
                SELECT timestamp, metadata FROM training_logs
                WHERE json_extract(metadata, '$.type') = 'training_completed'
                ORDER BY timestamp DESC LIMIT 1
            """)
            last_training_row = cursor.fetchone()

            last_training_time = None
            if last_training_row:
                last_training_time = last_training_row[0]

            # Get current model info
            if using_base_model:
                model_info = "YOLOv8n (Base)"
            else:
                model_info = "Custom (Fine-tuned)"

            # Calculate processed images across all service trained subdirectories
            processed_count = 0
            for trained_dir in [DIGITS_TRAINED_DIR, COLORS_TRAINED_DIR, WHISPER_TRAINED_DIR]:
                if trained_dir.exists():
                    processed_count += len(list(trained_dir.glob("*.jpg")))

            return {
                "images_waiting": image_count,
                "training_sessions": training_sessions,
                "ready_for_training": image_count > 0,
                "last_training_time": last_training_time,
                "current_model": model_info,
                "processed_images": processed_count,
                "total_images_annotated": image_count + processed_count
            }
    except Exception as e:
        return {
            "error": str(e),
            "images_waiting": 0,
            "training_sessions": 0,
            "ready_for_training": False,
            "last_training_time": None,
            "current_model": "YOLOv8n (Base)",
            "processed_images": 0,
            "total_images_annotated": 0
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
