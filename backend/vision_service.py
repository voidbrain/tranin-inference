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
                }
            ]
        }
