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

        # Auto-load latest merged model on startup
        self._load_merged_model_if_available()

    async def detect_objects(self, image_data: bytes) -> dict:
        """Run object detection on image data"""
        if not self.model:
            raise Exception("YOLO model not loaded")

        try:
            # Convert bytes to image for processing
            import io
            from PIL import Image

            # Ensure we have valid image data
            if len(image_data) == 0:
                raise Exception("Empty image data received")

            image = Image.open(io.BytesIO(image_data))

            # Mock detection based on loaded model type
            model_type = self.model.get("type", "unknown") if isinstance(self.model, dict) else "unknown"
            training_type = self.model.get("training_type", "unknown") if isinstance(self.model, dict) else "unknown"

            # Generate mock detections based on model type
            detections = []

            if model_type == "merged" or training_type == "merged":
                # Merged model - detect both digits and colors
                detection_classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'red', 'blue', 'green', 'yellow', 'orange', 'purple']
                num_detections = 3  # Simulate detecting multiple items
                for i in range(num_detections):
                    class_name = detection_classes[i % len(detection_classes)]
                    detections.append({
                        "label": class_name,
                        "score": 0.82 + (i * 0.03),  # Varying confidence
                        "box": {
                            "xMin": 50 + (i * 80),
                            "yMin": 100 + (i * 40),
                            "xMax": 150 + (i * 80),
                            "yMax": 180 + (i * 40)
                        }
                    })
            elif training_type == "digits":
                # Digits model - detect only digits
                digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                num_detections = 2
                for i in range(num_detections):
                    class_name = digits[i % len(digits)]
                    detections.append({
                        "label": class_name,
                        "score": 0.85 + (i * 0.02),
                        "box": {
                            "xMin": 60 + (i * 100),
                            "yMin": 110,
                            "xMax": 140 + (i * 100),
                            "yMax": 170
                        }
                    })
            elif training_type == "colors":
                # Colors model - detect only colors
                colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']
                num_detections = 2
                for i in range(num_detections):
                    class_name = colors[i % len(colors)]
                    detections.append({
                        "label": class_name,
                        "score": 0.78 + (i * 0.04),
                        "box": {
                            "xMin": 70 + (i * 120),
                            "yMin": 120,
                            "xMax": 160 + (i * 120),
                            "yMax": 180
                        }
                    })
            else:
                # Base model or unknown - use generic placeholders
                generic_classes = ['object_1', 'object_2']
                for i, class_name in enumerate(generic_classes):
                    detections.append({
                        "label": class_name,
                        "score": 0.75,
                        "box": {
                            "xMin": 80 + (i * 140),
                            "yMin": 130,
                            "xMax": 170 + (i * 140),
                            "yMax": 190
                        }
                    })

            return {
                "detections": detections,
                "mock": True,
                "model_type": model_type,
                "training_type": training_type,
                "total_detections": len(detections)
            }
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

    def get_training_queue_status(self, page: int = 1, per_page: int = 20) -> dict:
        """Get training queue status with pagination and separate digit/color counts"""
        try:
            # Count images waiting for training - separate by type
            digits_waiting = len(list(self.digits_waiting_dir.glob("*.txt")))
            colors_waiting = len(list(self.colors_waiting_dir.glob("*.txt")))
            total_waiting = digits_waiting + colors_waiting

            # Count processed images - separate by type
            digits_processed = len(list(self.digits_processed_dir.glob("*.txt")))
            colors_processed = len(list(self.colors_processed_dir.glob("*.txt")))
            total_processed = digits_processed + colors_processed

            # Get waiting file details with pagination
            waiting_files = []
            for dir_type, dir_path in [("digits", self.digits_waiting_dir), ("colors", self.colors_waiting_dir)]:
                for annotation_file in dir_path.glob("*.txt"):
                    file_stat = annotation_file.stat()
                    waiting_files.append({
                        "filename": annotation_file.name,
                        "type": dir_type,
                        "size": file_stat.st_size,
                        "created": datetime.datetime.fromtimestamp(file_stat.st_ctime).isoformat(),
                        "path": str(annotation_file)
                    })

            # Sort by creation time (newest first) and paginate
            waiting_files.sort(key=lambda x: x["created"], reverse=True)
            total_files = len(waiting_files)
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            paginated_files = waiting_files[start_idx:end_idx]

            # Get processed file details with pagination
            processed_files = []
            for dir_type, dir_path in [("digits", self.digits_processed_dir), ("colors", self.colors_processed_dir)]:
                for annotation_file in dir_path.glob("*.txt"):
                    file_stat = annotation_file.stat()
                    processed_files.append({
                        "filename": annotation_file.name,
                        "type": dir_type,
                        "size": file_stat.st_size,
                        "processed": datetime.datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        "path": str(annotation_file)
                    })

            # Sort by processed time (newest first) and paginate
            processed_files.sort(key=lambda x: x["processed"], reverse=True)
            total_processed_files = len(processed_files)
            proc_start_idx = (page - 1) * per_page
            proc_end_idx = proc_start_idx + per_page
            paginated_processed = processed_files[proc_start_idx:proc_end_idx]

            return {
                # Aggregated counts (for backward compatibility)
                "images_waiting": total_waiting,
                "processed_images": total_processed,
                "total_images_annotated": total_processed,

                # Separate counts by type
                "digits_waiting": digits_waiting,
                "colors_waiting": colors_waiting,
                "digits_processed": digits_processed,
                "colors_processed": colors_processed,

                # Training status
                "training_sessions": 0,  # Not implemented yet
                "ready_for_training": {
                    "digits": digits_waiting > 0,
                    "colors": colors_waiting > 0,
                    "combined": total_waiting > 0
                },

                # Pagination info and data
                "waiting_files": {
                    "total": total_files,
                    "page": page,
                    "per_page": per_page,
                    "pages": (total_files + per_page - 1) // per_page,
                    "data": paginated_files
                },

                "processed_files": {
                    "total": total_processed_files,
                    "page": page,
                    "per_page": per_page,
                    "pages": (total_processed_files + per_page - 1) // per_page,
                    "data": paginated_processed
                },

                "last_training_time": None,
                "current_model": "YOLOv8n (Base)"
            }
        except Exception as e:
            return {"error": str(e)}

    def get_training_logs(self) -> dict:
        """Get training logs"""
        return {"logs": self.training_logs[-50:]}  # Last 50 logs

    async def process_training_data(self, training_data: dict):
        """Process training data with labels and bounding boxes, splitting by detection type"""
        try:
            # Extract image data (base64) and detections
            image_data = training_data.get("imageData", "")
            detections = training_data.get("detections", [])

            if not image_data.startswith("data:image/"):
                raise Exception("Invalid image data format")

            # Decode base64 image
            image_bytes = base64.b64decode(image_data.split(",")[1])
            image = Image.open(io.BytesIO(image_bytes))

            # Analyze detections to categorize by type
            digit_detections = []
            color_detections = []

            for detection in detections:
                label = detection.get("label", "")
                # Check if it's a digit (0-9)
                if label.isdigit() and label in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                    digit_detections.append(detection)
                # Check if it's a color
                elif label in ['red', 'blue', 'green', 'yellow', 'orange', 'purple']:
                    color_detections.append(detection)

            # Generate unique timestamp for this training data
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

            # Save data based on detection types found
            saved_files = {}

            # Process digit detections
            if digit_detections:
                digit_image_path = self.digits_waiting_dir / f"vision_{timestamp}_digits.jpg"
                digit_annotation_path = self.digits_waiting_dir / f"vision_{timestamp}_digits.txt"

                # Save the image (convert to RGB if necessary to ensure JPEG compatibility)
                if image.mode != 'RGB':
                    rgb_image = image.convert('RGB')
                    rgb_image.save(digit_image_path, 'JPEG', quality=95)
                else:
                    image.save(digit_image_path, 'JPEG', quality=95)

                # Save YOLO-format annotations for digits
                with open(digit_annotation_path, 'w') as f:
                    for detection in digit_detections:
                        label = detection["label"]
                        bbox = detection["bbox"]  # [x1, y1, x2, y2]
                        confidence = detection.get("confidence", 100)

                        # Convert to YOLO format: class x_center y_center width height
                        # Map digits to class indices (0-9)
                        class_id = int(label)

                        # Convert pixel coordinates to normalized [0,1] coordinates
                        x1, y1, x2, y2 = bbox
                        img_width, img_height = image.size

                        x_center = ((x1 + x2) / 2) / img_width
                        y_center = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height

                        confidence_normalized = confidence / 100.0

                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence_normalized:.6f}\n")

                saved_files["digits"] = {
                    "image": str(digit_image_path),
                    "annotation": str(digit_annotation_path),
                    "detections": len(digit_detections)
                }

            # Process color detections
            if color_detections:
                color_image_path = self.colors_waiting_dir / f"vision_{timestamp}_colors.jpg"
                color_annotation_path = self.colors_waiting_dir / f"vision_{timestamp}_colors.txt"

                # Save the image (convert to RGB if necessary to ensure JPEG compatibility)
                if image.mode != 'RGB':
                    rgb_image = image.convert('RGB')
                    rgb_image.save(color_image_path, 'JPEG', quality=95)
                else:
                    image.save(color_image_path, 'JPEG', quality=95)

                # Map color names to class indices
                color_to_id = {'red': 0, 'blue': 1, 'green': 2, 'yellow': 3, 'orange': 4, 'purple': 5}

                # Save YOLO-format annotations for colors
                with open(color_annotation_path, 'w') as f:
                    for detection in color_detections:
                        label = detection["label"]
                        bbox = detection["bbox"]
                        confidence = detection.get("confidence", 100)

                        class_id = color_to_id[label]

                        # Convert pixel coordinates to normalized [0,1] coordinates
                        x1, y1, x2, y2 = bbox
                        img_width, img_height = image.size

                        x_center = ((x1 + x2) / 2) / img_width
                        y_center = ((y1 + y2) / 2) / img_height
                        width = (x2 - x1) / img_width
                        height = (y2 - y1) / img_height

                        confidence_normalized = confidence / 100.0

                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {confidence_normalized:.6f}\n")

                saved_files["colors"] = {
                    "image": str(color_image_path),
                    "annotation": str(color_annotation_path),
                    "detections": len(color_detections)
                }

            processed_data = {
                "image_size": image.size,
                "total_detections": len(detections),
                "digit_detections": len(digit_detections),
                "color_detections": len(color_detections),
                "saved_files": saved_files,
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

    async def load_lora_adapter(self, training_data: dict) -> dict:
        """Load a LoRA adapter for specialized detection"""
        try:
            training_type = training_data.get("training_type")
            if training_type not in ["digits", "colors"]:
                raise Exception(f"Invalid training type: {training_type}")

            # Check if the LoRA adapter file exists
            lora_file = self.models_dir / "loras" / training_type / f"{training_type}.safetensors"
            if not lora_file.exists():
                raise Exception(f"LoRA adapter for {training_type} not found. Train the model first.")

            # Set the model to indicate we're using a specialized adapter
            self.model = {
                "type": "lora",
                "training_type": training_type,
                "lora_path": str(lora_file),
                "base_model": "YOLOv8n"
            }
            self.using_base_model = False

            return {
                "message": f"LoRA adapter for {training_type} loaded successfully",
                "training_type": training_type,
                "lora_path": str(lora_file),
                "model_status": "loaded"
            }

        except Exception as e:
            raise Exception(f"Failed to load LoRA adapter: {str(e)}")

    async def load_merged_model(self, training_data: dict) -> dict:
        """Load a merged model that combines digits and colors"""
        try:
            training_type = training_data.get("training_type")
            if training_type != "merged":
                raise Exception(f"Invalid training type for merged model: {training_type}")

            # Check if merged model files exist
            merged_pt_file = self.merged_dir / "digits_colors_merged.pt"
            merged_onnx_file = self.merged_dir / "digits_colors_merged.onnx"

            if not merged_pt_file.exists() and not merged_onnx_file.exists():
                raise Exception("Merged model not found. Create the merged model first.")

            # Set the model to indicate we're using a merged model
            self.model = {
                "type": "merged",
                "training_type": "merged",
                "merged_path": str(merged_pt_file) if merged_pt_file.exists() else str(merged_onnx_file),
                "onnx_path": str(merged_onnx_file) if merged_onnx_file.exists() else None,
                "base_model": "YOLOv8n"
            }
            self.using_base_model = False

            return {
                "message": "Merged model loaded successfully",
                "training_type": "merged",
                "model_path": str(merged_pt_file) if merged_pt_file.exists() else str(merged_onnx_file),
                "onnx_path": str(merged_onnx_file) if merged_onnx_file.exists() else None,
                "model_status": "loaded"
            }

        except Exception as e:
            raise Exception(f"Failed to load merged model: {str(e)}")

    async def load_lora_adapter_endpoint(self, training_data: dict) -> dict:
        """API endpoint wrapper for loading LoRA adapter or merged model"""
        from fastapi import HTTPException

        try:
            training_type = training_data.get("training_type")
            if training_type == "merged":
                return await self.load_merged_model(training_data)
            else:
                return await self.load_lora_adapter(training_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def load_merged_model_endpoint(self, training_data: dict) -> dict:
        """API endpoint wrapper for loading merged model specifically"""
        from fastapi import HTTPException

        try:
            return await self.load_merged_model(training_data)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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
        """Create merged models from specialized LoRA adapters"""
        try:
            import subprocess
            import asyncio

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

            # Create outputs
            created_models = []

            # 1. Create digits + colors merged model
            self.training_progress = 10.0
            self.training_message = "Creating digits + colors merged model..."

            merged_pt_path = self.merged_dir / "digits_colors_merged.pt"
            process = subprocess.Popen([
                "python", "models/vision/scripts/merge_lora.py",
                "--base", "yolov8n.pt",
                "--lora", str(digits_lora),
                "--lora", str(colors_lora),
                "--out", str(merged_pt_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            stdout, _ = process.communicate()
            if process.returncode != 0:
                raise Exception(f"Failed to create digits+colors merged model: {stdout}")

            created_models.append({
                "name": "digits_colors_merged",
                "pt_path": str(merged_pt_path),
                "onnx_path": str(merged_pt_path.with_suffix('.onnx'))
            })

            # 2. Create colors-only merged model
            self.training_progress = 40.0
            self.training_message = "Creating colors-only merged model..."

            colors_pt_path = self.merged_dir / "colors_merged.pt"
            process = subprocess.Popen([
                "python", "models/vision/scripts/merge_lora.py",
                "--base", "yolov8n.pt",
                "--lora", str(colors_lora),
                "--out", str(colors_pt_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            stdout, _ = process.communicate()
            if process.returncode != 0:
                raise Exception(f"Failed to create colors merged model: {stdout}")

            created_models.append({
                "name": "colors_merged",
                "pt_path": str(colors_pt_path),
                "onnx_path": str(colors_pt_path.with_suffix('.onnx'))
            })

            # 3. Create digits-only merged model
            self.training_progress = 70.0
            self.training_message = "Creating digits-only merged model..."

            digits_pt_path = self.merged_dir / "digits_merged.pt"
            process = subprocess.Popen([
                "python", "models/vision/scripts/merge_lora.py",
                "--base", "yolov8n.pt",
                "--lora", str(digits_lora),
                "--out", str(digits_pt_path)
            ], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            stdout, _ = process.communicate()
            if process.returncode != 0:
                raise Exception(f"Failed to create digits merged model: {stdout}")

            created_models.append({
                "name": "digits_merged",
                "pt_path": str(digits_pt_path),
                "onnx_path": str(digits_pt_path.with_suffix('.onnx'))
            })

            self.training_progress = 100.0
            self.training_message = "All merged models created successfully!"
            self.training_status = "success"

            return {
                "status": "success",
                "created_models": created_models,
                "total_models": len(created_models),
                "base_model": "yolov8n.pt",
                "lora_adapters_used": ["digits.safetensors", "colors.safetensors"],
                "message": f"Created {len(created_models)} merged models: digits+colors, colors-only, digits-only"
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

    def get_training_queue_status_endpoint(self, page: int = 1, per_page: int = 20) -> dict:
        """API endpoint wrapper for getting training queue status with pagination"""
        return self.get_training_queue_status(page, per_page)

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

    async def train_lora_specialized_endpoint(self, training_type: str, background_tasks: "BackgroundTasks") -> dict:
        """API endpoint wrapper for specialized LoRA training (frontend expects this endpoint)"""
        from fastapi import HTTPException

        try:
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

    def _load_merged_model_if_available(self):
        """Load the latest merged model if available at startup"""
        try:
            # Try to load the merged model that combines both digits and colors
            merged_pt_file = self.merged_dir / "digits_colors_merged.pt"

            if merged_pt_file.exists():
                print(f"Auto-loading merged model: {merged_pt_file}")
                self.model = {
                    "type": "merged",
                    "training_type": "merged",
                    "merged_path": str(merged_pt_file),
                    "onnx_path": str(merged_pt_file.with_suffix('.onnx')),
                    "base_model": "YOLOv8n"
                }
                self.using_base_model = False
                print(f"✓ Auto-loaded merged model: {merged_pt_file}")

            else:
                print("No merged model found at startup - starting with base model")
                self.model = None
                self.using_base_model = True

        except Exception as e:
            print(f"Warning: Failed to auto-load merged model: {e}")
            self.model = None
            self.using_base_model = True

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
                    "path": "/vision/tags/{detection_mode}",
                    "methods": ["GET"],
                    "handler": "get_tags_endpoint",
                    "params": ["detection_mode: str"]
                },
                {
                    "path": "/vision/model-status",
                    "methods": ["GET"],
                    "handler": "get_model_status_endpoint"
                },
                {
                    "path": "/vision/upload-training-data",
                    "methods": ["POST"],
                    "handler": "upload_training_data_endpoint",
                    "params": ["training_data: dict"]
                },
                {
                    "path": "/vision/specialized-training-count/{training_type}",
                    "methods": ["GET"],
                    "handler": "get_specialized_training_count_endpoint",
                    "params": ["training_type: str"]
                },
                {
                    "path": "/vision/training-queue-status",
                    "methods": ["GET"],
                    "handler": "get_training_queue_status_endpoint"
                },
                {
                    "path": "/vision/training-logs",
                    "methods": ["GET"],
                    "handler": "get_training_logs_endpoint"
                },
                {
                    "path": "/vision/reset-model",
                    "methods": ["POST"],
                    "handler": "reset_model_endpoint"
                },
                {
                    "path": "/vision/detect",
                    "methods": ["POST"],
                    "handler": "detect_objects_endpoint",
                    "params": ["file: UploadFile"]
                },
                {
                    "path": "/vision/train-digits-lora",
                    "methods": ["POST"],
                    "handler": "train_digits_lora_endpoint",
                    "params": ["background_tasks: BackgroundTasks"]
                },
                {
                    "path": "/vision/train-colors-lora",
                    "methods": ["POST"],
                    "handler": "train_colors_lora_endpoint",
                    "params": ["background_tasks: BackgroundTasks"]
                },
                {
                    "path": "/vision/create-merged-model",
                    "methods": ["POST"],
                    "handler": "create_merged_model_endpoint",
                    "params": ["background_tasks: BackgroundTasks"]
                },

                {
                    "path": "/vision/train-lora-specialized",
                    "methods": ["POST"],
                    "handler": "train_lora_specialized_endpoint",
                    "params": ["type: str", "background_tasks: BackgroundTasks"]
                },
                {
                    "path": "/vision/load-lora-adapter",
                    "methods": ["POST"],
                    "handler": "load_lora_adapter_endpoint",
                    "params": ["training_data: dict"]
                },
                {
                    "path": "/vision/train",
                    "methods": ["POST"],
                    "handler": "train_endpoint",
                    "params": ["training_data: dict", "background_tasks: BackgroundTasks"]
                }
            ]
        }
