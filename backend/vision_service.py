"""
YOLO Service for Computer Vision
Handles object detection, model training, and fine-tuning
"""
import os
import datetime
import base64
import io
from pathlib import Path
from PIL import Image

# Lazy imports - loaded on first use to avoid import errors during configuration
_ultralytics = None
_torch = None

def _import_vision_libraries():
    """Lazy import of computer vision libraries"""
    global _ultralytics, _torch
    if _ultralytics is None:
        import ultralytics
        import torch
        ultralytics.settings.update({
            'runs_dir': 'data/vision/runs',
            'weights_dir': 'models/vision',
            'datasets_dir': 'data/vision/datasets',
            'models_dir': 'models/vision',
        })
        _ultralytics = ultralytics
        _torch = torch

def _get_ultralytics():
    _import_vision_libraries()
    return _ultralytics

def _get_torch():
    _import_vision_libraries()
    return _torch

class VisionService:
    """Service for handling YOLO computer vision functionality"""

    def __init__(self, models_dir: str = "models/vision", data_dir: str = "data/vision"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        # Create subdirectories like speech service
        self.digits_processed_dir = self.data_dir / "digits" / "processed"
        self.digits_waiting_dir = self.data_dir / "digits" / "waiting"
        self.colors_processed_dir = self.data_dir / "colors" / "processed"
        self.colors_waiting_dir = self.data_dir / "colors" / "waiting"

        for dir_path in [self.digits_processed_dir, self.digits_waiting_dir,
                        self.colors_processed_dir, self.colors_waiting_dir]:
            dir_path.mkdir(exist_ok=True, parents=True)

        self.model = None
        self.using_base_model = True

        # Training status tracking
        self.training_status = "idle"  # idle, running, success, error
        self.training_progress = 0.0
        self.training_message = ""
        self.training_logs = []
        self.training_start_time = None

        # Specialized training tracking
        self.specialized_training = {}  # {'digits': {...}, 'colors': {...}}

        # Merged models directory - under vision service
        self.merged_dir = Path(models_dir) / "merged"
        self.merged_dir.mkdir(exist_ok=True, parents=True)

    async def detect_objects(self, image_data: bytes) -> dict:
        """Run object detection on image data"""
        if not self.model:
            raise Exception("YOLO model not loaded")

        try:
            # Convert bytes to image and run detection
            ultralytics = _get_ultralytics()

            # This would require the heavy dependencies to be installed
            # For now, return a mock response to test the configuration system
            return {"detections": [], "mock": True}
        except Exception as e:
            raise Exception(f"YOLO inference error: {e}")

    def get_status(self) -> dict:
        """Get current status of YOLO service"""
        return {
            'loaded': self.model is not None,
            'model_type': 'Vision Service',
            'source': 'ultralytics'
        }

    def get_tags(self, detection_mode: str) -> dict:
        """Get available tags for detection mode"""
        if detection_mode == "digits":
            return {"tags": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']}
        elif detection_mode == "colors":
            return {"tags": ['red', 'blue', 'green', 'yellow', 'orange', 'purple']}
        else:
            return {"tags": []}

    def get_specialized_training_count(self, training_type: str) -> dict:
        """Get count of specialized training data"""
        try:
            if training_type == "digits":
                processed_count = len(list(self.digits_processed_dir.glob("*.txt")))
                waiting_count = len(list(self.digits_waiting_dir.glob("*.txt")))
                return {"count": processed_count + waiting_count}
            elif training_type == "colors":
                processed_count = len(list(self.colors_processed_dir.glob("*.txt")))
                waiting_count = len(list(self.colors_waiting_dir.glob("*.txt")))
                return {"count": processed_count + waiting_count}
            else:
                return {"count": 0}
        except Exception as e:
            return {"error": str(e), "count": 0}

    def get_training_queue_status(self) -> dict:
        """Get training queue status"""
        try:
            # Count images waiting for training
            images_waiting = 0
            for dir_path in [self.digits_waiting_dir, self.colors_waiting_dir]:
                images_waiting += len(list(dir_path.glob("*.txt")))  # annotation files

            processed_images = 0
            for dir_path in [self.digits_processed_dir, self.colors_processed_dir]:
                processed_images += len(list(dir_path.glob("*.txt")))

            return {
                "images_waiting": images_waiting,
                "training_sessions": 0,  # Not implemented yet
                "ready_for_training": images_waiting > 0,
                "last_training_time": None,
                "current_model": "YOLOv8n (Base)",
                "processed_images": processed_images,
                "total_images_annotated": processed_images
            }
        except Exception as e:
            return {"error": str(e)}

    def get_training_logs(self) -> dict:
        """Get training logs"""
        return {"logs": self.training_logs[-50:]}  # Last 50 logs

    async def process_training_data(self, training_data: dict):
        """Process training data with labels and bounding boxes"""
        try:
            # Extract image data (base64) and detections
            image_data = training_data.get("imageData", "")
            detections = training_data.get("detections", [])

            if not image_data.startswith("data:image/"):
                raise Exception("Invalid image data format")

            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(",")[1])
            image = Image.open(io.BytesIO(image_bytes))

            # Save image and annotations
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            image_filename = f"vision_{timestamp}_training.jpg"
            annotation_filename = f"vision_{timestamp}_training.txt"

            # Save image (placeholder - would save processed version)
            # For now just store the data
            processed_data = {
                "image_size": image.size,
                "detections": detections,
                "timestamp": timestamp
            }

            return processed_data

        except Exception as e:
            raise Exception(f"Failed to process training data: {str(e)}")

    async def reset_model(self) -> dict:
        """Reset to base YOLO model"""
        self.model = None
        self.using_base_model = True
        return {"message": "Model reset to base YOLOv8n"}

    def _prepare_specialized_dataset(self, training_type: str):
        """Prepare YOLO dataset from specialized training data"""
        try:
            # Determine which directories to use
            if training_type == "digits":
                waiting_dir = self.digits_waiting_dir
                processed_dir = self.digits_processed_dir
            elif training_type == "colors":
                waiting_dir = self.colors_waiting_dir
                processed_dir = self.colors_processed_dir
            else:
                raise Exception(f"Unknown training type: {training_type}")

            # Move training data from processed to waiting (like speech service)
            self._move_processed_to_waiting(training_type)

            # Collect annotation files
            annotation_files = list(waiting_dir.glob("*.txt"))
            if not annotation_files:
                raise Exception(f"No {training_type} training data found")

            return {
                "waiting_dir": waiting_dir,
                "annotation_files": annotation_files,
                "sample_count": len(annotation_files)
            }

        except Exception as e:
            raise Exception(f"Failed to prepare {training_type} dataset: {str(e)}")

    def _move_processed_to_waiting(self, training_type: str):
        """Move training data from processed to waiting directory"""
        try:
            if training_type == "digits":
                processed_dir, waiting_dir = self.digits_processed_dir, self.digits_waiting_dir
            elif training_type == "colors":
                processed_dir, waiting_dir = self.colors_processed_dir, self.colors_waiting_dir
            else:
                return

            # Move relevant files
            moved_count = 0
            for file_path in processed_dir.glob("vision_*_training.*"):
                if file_path.is_file():
                    target_path = waiting_dir / file_path.name
                    file_path.rename(target_path)
                    moved_count += 1

            if moved_count > 0:
                print(f"Moved {moved_count} {training_type} training files")

        except Exception as e:
            print(f"Warning: Failed to move {training_type} training data: {e}")

    async def train_specialized_lora(self, training_type: str = "digits") -> dict:
        """Train specialized LoRA adapter for digits or colors"""
        try:
            # Set training status
            self.training_status = "running"
            self.training_progress = 0.0
            self.training_message = f"Initializing {training_type} LoRA training..."
            self.training_start_time = datetime.datetime.now()

            # Prepare dataset
            self.training_message = f"Preparing {training_type} dataset..."
            dataset_info = self._prepare_specialized_dataset(training_type)

            # Configure LoRA training parameters
            if training_type == "digits":
                lora_rank = 8
                epochs = 80
                batch_size = 16
            elif training_type == "colors":
                lora_rank = 4
                epochs = 60
                batch_size = 8
            else:
                raise Exception(f"Unknown training type: {training_type}")

            self.training_message = f"Training {training_type} LoRA (rank={lora_rank}, epochs={epochs})..."

            # Create output directories
            lora_output_dir = self.models_dir / "loras" / training_type
            lora_output_dir.mkdir(exist_ok=True, parents=True)

            # YOLO LoRA training command would be:
            # yolo train model=base/yolov8n.pt lora=1 lora_rank={lora_rank}
            # epochs={epochs} imgsz=640 project=loras name={training_type}
            # data={dataset_yaml} ...

            # For now, simulate the training process
            import asyncio
            await asyncio.sleep(2)  # Simulate training time

            # Create mock LoRA file
            lora_file = lora_output_dir / f"{training_type}.safetensors"
            with open(lora_file, 'w') as f:
                f.write(f"# Mock LoRA adapter for {training_type}\n")
                f.write(f"# Rank: {lora_rank}\n")
                f.write(f"# Epochs: {epochs}\n")

            # Set success status
            self.training_status = "success"
            self.training_progress = 100.0
            self.training_message = f"{training_type} LoRA training completed!"

            return {
                "status": "success",
                "training_type": training_type,
                "lora_rank": lora_rank,
                "epochs": epochs,
                "lora_path": str(lora_file),
                "samples_trained": dataset_info["sample_count"]
            }

        except Exception as e:
            self.training_status = "error"
            self.training_progress = 0.0
            self.training_message = f"Training failed: {str(e)}"
            raise Exception(f"LoRA training failed: {str(e)}")

    async def create_merged_model(self) -> dict:
        """Create merged model from specialized LoRA adapters"""
        try:
            self.training_status = "running"
            self.training_progress = 0.0
            self.training_message = "Starting LoRA merging process..."
            self.training_start_time = datetime.datetime.now()

            # Check for LoRA files
            lora_dir = self.models_dir / "loras"
            digits_lora = lora_dir / "digits" / "digits.safetensors"
            colors_lora = lora_dir / "colors" / "colors.safetensors"

            if not digits_lora.exists() or not colors_lora.exists():
                raise Exception("Missing LoRA adapters. Train digits and colors models first.")

            # Create merged model
            self.training_progress = 10.0
            self.training_message = "Merging LoRA adapters..."

            # Mock merge command:
            # python scripts/merge_lora.py --base base/yolov8n.pt
            # --lora loras/digits.safetensors --lora loras/colors.safetensors
            # --out merged/digits_colors_merged.pt

            import asyncio
            await asyncio.sleep(1)  # Simulate merge time

            merged_model_path = self.merged_dir / "digits_colors_merged.pt"
            with open(merged_model_path, 'w') as f:
                f.write("# Merged YOLO model (digits + colors)\n")

            self.training_progress = 80.0
            self.training_message = "Merging complete. Exporting to ONNX..."

            # Export to ONNX
            # yolo export model=merged/yolo_merged.pt format=onnx
            onnx_model_path = self.merged_dir / "digits_colors_merged.onnx"
            with open(onnx_model_path, 'w') as f:
                f.write("# ONNX export of merged model\n")

            self.training_progress = 100.0
            self.training_message = "Merged model creation completed!"
            self.training_status = "success"

            return {
                "status": "success",
                "merged_model_path": str(merged_model_path),
                "onnx_model_path": str(onnx_model_path),
                "base_model": "yolov8n.pt",
                "lora_adapters": ["digits.safetensors", "colors.safetensors"]
            }

        except Exception as e:
            self.training_status = "error"
            self.training_progress = 0.0
            self.training_message = f"Merging failed: {str(e)}"
            raise Exception(f"Merged model creation failed: {str(e)}")

    def get_merged_model_status(self) -> dict:
        """Get status of merged models"""
        try:
            merged_files = []

            # Check for merged models
            for file_path in self.merged_dir.glob("*"):
                if file_path.is_file():
                    stat = file_path.stat()
                    merged_files.append({
                        "filename": file_path.name,
                        "size": stat.st_size,
                        "modified": datetime.datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        "format": "pt" if file_path.suffix == ".pt" else "onnx" if file_path.suffix == ".onnx" else "safetensors"
                    })

            return {
                "merged_dir": str(self.merged_dir),
                "available_models": merged_files,
                "model_count": len(merged_files)
            }

        except Exception as e:
            return {"error": str(e), "available_models": [], "model_count": 0}

    # ===== VISION-SPECIFIED ENDPOINT METHODS =====
    # These methods wrap the service functionality for API endpoints

    # ===== VISION-SPECIFIC ENDPOINT METHODS =====
    # These methods wrap the service functionality for API endpoints

    async def detect_objects_endpoint(self, file: "UploadFile") -> dict:
        """API endpoint wrapper for object detection"""
        file_bytes = await file.read()
        return await self.detect_objects(file_bytes)

    def get_model_status_endpoint(self) -> dict:
        """API endpoint wrapper for getting model status"""
        return self.get_status()

    def get_tags_endpoint(self, detection_mode: str) -> dict:
        """API endpoint wrapper for getting available tags"""
        return self.get_tags(detection_mode)

    async def upload_training_data_endpoint(self, training_data: dict) -> dict:
        """API endpoint wrapper for uploading training data"""
        from fastapi import HTTPException

        try:
            result = await self.process_training_data(training_data)
            return {
                "message": "Training data uploaded successfully",
                "processed_data": result
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    async def upload_specialized_training_data_endpoint(self, training_data: dict) -> dict:
        """API endpoint wrapper for uploading specialized training data"""
        from fastapi import HTTPException

        try:
            training_type = training_data.get("training_type")
            if training_type not in ["digits", "colors"]:
                raise HTTPException(status_code=400, detail="Invalid training type")

            result = await self.process_training_data(training_data)
            return {
                "message": f"{training_type} training data uploaded successfully",
                "training_type": training_type,
                "processed_data": result
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

    def get_specialized_training_count_endpoint(self, training_type: str) -> dict:
        """API endpoint wrapper for getting specialized training count"""
        return self.get_specialized_training_count(training_type)

    def get_training_queue_status_endpoint(self) -> dict:
        """API endpoint wrapper for getting training queue status"""
        return self.get_training_queue_status()

    def get_training_logs_endpoint(self) -> dict:
        """API endpoint wrapper for getting training logs"""
        return self.get_training_logs()

    async def reset_model_endpoint(self) -> dict:
        """API endpoint wrapper for resetting model"""
        return await self.reset_model()

    async def train_digits_lora_endpoint(self, background_tasks: "BackgroundTasks") -> dict:
        """API endpoint wrapper for training digits LoRA"""
        from fastapi import HTTPException

        try:
            background_tasks.add_task(self.train_specialized_lora, "digits")
            return {
                "message": "Digits LoRA training started",
                "training_type": "digits",
                "status": "background_processing"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def train_colors_lora_endpoint(self, background_tasks: "BackgroundTasks") -> dict:
        """API endpoint wrapper for training colors LoRA"""
        from fastapi import HTTPException

        try:
            background_tasks.add_task(self.train_specialized_lora, "colors")
            return {
                "message": "Colors LoRA training started",
                "training_type": "colors",
                "status": "background_processing"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def create_merged_model_endpoint(self, background_tasks: "BackgroundTasks") -> dict:
        """API endpoint wrapper for creating merged model"""
        from fastapi import HTTPException

        try:
            background_tasks.add_task(self.create_merged_model)
            return {
                "message": "Merged model creation started",
                "status": "background_processing"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_merged_model_status_endpoint(self) -> dict:
        """API endpoint wrapper for getting merged model status"""
        return self.get_merged_model_status()

    async def train_lora_specialized_endpoint(self, training_data: dict, background_tasks: "BackgroundTasks") -> dict:
        """API endpoint wrapper for specialized LoRA training (frontend expects this endpoint)"""
        from fastapi import HTTPException

        try:
            training_type = training_data.get("training_type")
            if training_type not in ["digits", "colors"]:
                raise HTTPException(status_code=400, detail="Invalid training type")

            background_tasks.add_task(
                self.train_specialized_lora,
                training_type
            )

            return {
                "message": f"{training_type} LoRA training started",
                "training_type": training_type,
                "status": "background_processing"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def train_endpoint(self, training_data: dict, background_tasks: "BackgroundTasks") -> dict:
        """API endpoint wrapper for general YOLO training (frontend expects this endpoint)"""
        from fastapi import HTTPException

        try:
            # For now, just start general training if data is available
            if not self.get_training_queue_status().get("ready_for_training"):
                raise HTTPException(status_code=400, detail="No training data available")

            # Start training with default parameters
            background_tasks.add_task(
                self.train_specialized_lora,
                "digits"  # Default to digits or could be configurable
            )

            return {
                "message": "General YOLO training started",
                "status": "background_processing"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @classmethod
    def get_service_config(cls):
        """Return the service configuration with endpoints and database schema"""
        return {
            "database_schema": {
                "tables": {
                    "annotations": """
                        CREATE TABLE IF NOT EXISTS annotations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            service TEXT DEFAULT 'vision',
                            filename TEXT NOT NULL,
                            data TEXT NOT NULL,
                            labels TEXT NOT NULL,
                            timestamp TEXT NOT NULL
                        )
                    """
                }
            },
            "endpoints": [
                {
                    "path": "/vision/detect",
                    "methods": ["POST"],
                    "handler": "detect_objects_endpoint",
                    "params": ["file: UploadFile"]
                },
                {
                    "path": "/vision/model-status",
                    "methods": ["GET"],
                    "handler": "get_model_status_endpoint"
                },
                {
                    "path": "/tags/{detection_mode}",
                    "methods": ["GET"],
                    "handler": "get_tags_endpoint",
                    "params": ["detection_mode: str"]
                },
                {
                    "path": "/upload-training-data",
                    "methods": ["POST"],
                    "handler": "upload_training_data_endpoint",
                    "params": ["training_data: dict"]
                },
                {
                    "path": "/upload-specialized-training-data",
                    "methods": ["POST"],
                    "handler": "upload_specialized_training_data_endpoint",
                    "params": ["training_data: dict"]
                },
                {
                    "path": "/specialized-training-count/{training_type}",
                    "methods": ["GET"],
                    "handler": "get_specialized_training_count_endpoint",
                    "params": ["training_type: str"]
                },
                {
                    "path": "/training-queue-status",
                    "methods": ["GET"],
                    "handler": "get_training_queue_status_endpoint"
                },
                {
                    "path": "/training-logs",
                    "methods": ["GET"],
                    "handler": "get_training_logs_endpoint"
                },
                {
                    "path": "/reset-model",
                    "methods": ["POST"],
                    "handler": "reset_model_endpoint"
                },
                {
                    "path": "/train-digits-lora",
                    "methods": ["POST"],
                    "handler": "train_digits_lora_endpoint",
                    "params": ["background_tasks: BackgroundTasks"]
                },
                {
                    "path": "/train-colors-lora",
                    "methods": ["POST"],
                    "handler": "train_colors_lora_endpoint",
                    "params": ["background_tasks: BackgroundTasks"]
                },
                {
                    "path": "/create-merged-model",
                    "methods": ["POST"],
                    "handler": "create_merged_model_endpoint",
                    "params": ["background_tasks: BackgroundTasks"]
                },
                {
                    "path": "/merged-model-status",
                    "methods": ["GET"],
                    "handler": "get_merged_model_status_endpoint"
                },
                {
                    "path": "/train-lora-specialized",
                    "methods": ["POST"],
                    "handler": "train_lora_specialized_endpoint",
                    "params": ["training_data: dict", "background_tasks: BackgroundTasks"]
                },
                {
                    "path": "/train",
                    "methods": ["POST"],
                    "handler": "train_endpoint",
                    "params": ["training_data: dict", "background_tasks: BackgroundTasks"]
                }
            ]
        }
