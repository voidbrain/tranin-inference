"""
Vision Inference Module
Handles object detection and inference operations
"""
import io
from pathlib import Path
from PIL import Image


class VisionInferenceMixin:
    """Mixin class containing inference/detection functionality"""

    def _map_class_id_to_label(self, class_id: int, training_type: str, model_type: str) -> str:
        """Map YOLO class ID to human-readable label based on model type"""
        if training_type == "merged" or model_type == "merged":
            # Combined model: digits 0-9, then colors
            if class_id >= 0 and class_id <= 9:
                return str(class_id)  # 0-9
            elif class_id >= 10 and class_id <= 15:
                color_map = {10: 'red', 11: 'blue', 12: 'green', 13: 'yellow', 14: 'orange', 15: 'purple'}
                return color_map.get(class_id, f'class_{class_id}')
            else:
                return f'class_{class_id}'
        elif training_type == "digits":
            # Digits model: class IDs should map to 0-9
            return str(class_id) if class_id >= 0 and class_id <= 9 else f'class_{class_id}'
        elif training_type == "colors":
            # Colors model: class IDs map to color names
            color_map = {0: 'red', 1: 'blue', 2: 'green', 3: 'yellow', 4: 'orange', 5: 'purple'}
            return color_map.get(class_id, f'class_{class_id}')
        else:
            return f'class_{class_id}'

    def _get_mock_detections(self, training_type: str, model_type: str) -> tuple:
        """Fallback mock detections when real model inference fails"""
        digit_detections = []
        color_detections = []

        if model_type == "merged" or training_type == "merged":
            # Merged model - detect both digits and colors
            digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']

            # Add some digit detections
            for i in range(2):
                digit_detections.append({
                    "label": digits[i],
                    "score": 0.85 + (i * 0.02),
                    "box": {
                        "xMin": 60 + (i * 100),
                        "yMin": 110,
                        "xMax": 140 + (i * 100),
                        "yMax": 170
                    }
                })

            # Add some color detections
            for i in range(3):
                color_detections.append({
                    "label": colors[i],
                    "score": 0.78 + (i * 0.04),
                    "box": {
                        "xMin": 70 + (i * 120),
                        "yMin": 120,
                        "xMax": 160 + (i * 120),
                        "yMax": 180
                    }
                })
        elif training_type == "digits":
            # Digits model - detect only digits
            digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            for i in range(2):
                digit_detections.append({
                    "label": digits[i],
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
            for i in range(2):
                color_detections.append({
                    "label": colors[i],
                    "score": 0.78 + (i * 0.04),
                    "box": {
                        "xMin": 70 + (i * 120),
                        "yMin": 120,
                        "xMax": 160 + (i * 120),
                        "yMax": 180
                    }
                })
        # Base model returns empty detections (no mocks)

        return digit_detections, color_detections

    def _prioritize_digits_in_blue_box(self, digit_detections: list, blue_box_coords: dict) -> list:
        """Boost confidence scores for digit detections within or overlapping the blue box area"""
        if not blue_box_coords or not digit_detections:
            return digit_detections

        try:
            # Extract blue box coordinates
            blue_x = blue_box_coords.get('x', 0)
            blue_y = blue_box_coords.get('y', 0)
            blue_width = blue_box_coords.get('width', 0)
            blue_height = blue_box_coords.get('height', 0)

            blue_x2 = blue_x + blue_width
            blue_y2 = blue_y + blue_height

            # Process each digit detection
            prioritized_detections = []

            for detection in digit_detections:
                box = detection.get('box', {})
                digit_x1 = box.get('xMin', 0)
                digit_y1 = box.get('yMin', 0)
                digit_x2 = box.get('xMax', 0)
                digit_y2 = box.get('yMax', 0)

                # Calculate intersection area
                inter_x1 = max(digit_x1, blue_x)
                inter_y1 = max(digit_y1, blue_y)
                inter_x2 = min(digit_x2, blue_x2)
                inter_y2 = min(digit_y2, blue_y2)

                # Check if there's any intersection
                if inter_x2 > inter_x1 and inter_y2 > inter_y1:
                    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)

                    # Calculate digit box area
                    digit_area = (digit_x2 - digit_x1) * (digit_y2 - digit_y1)

                    # Calculate overlap percentage (intersection over digit area)
                    overlap_percentage = inter_area / digit_area if digit_area > 0 else 0

                    # Boost confidence for digits that overlap significantly with blue box
                    original_score = detection.get('score', 0)
                    boosted_score = original_score

                    if overlap_percentage > 0.2:  # More than 20% overlap
                        # Boost confidence, more for higher overlap
                        boost_factor = 0.1 + (overlap_percentage * 0.2)  # 0.1 to 0.3 boost
                        boosted_score = min(0.99, original_score + boost_factor)  # Cap at 0.99

                        print(f"Boosted digit '{detection.get('label', '?')}' confidence from {original_score:.2f} to {boosted_score:.2f} (overlap: {overlap_percentage:.2f})")

                    # Update detection with boosted score
                    boosted_detection = detection.copy()
                    boosted_detection['score'] = boosted_score
                    prioritized_detections.append(boosted_detection)
                else:
                    # No overlap, keep original detection
                    prioritized_detections.append(detection)

            return prioritized_detections

        except Exception as e:
            print(f"Error in blue box prioritization: {e}")
            # Return original detections if prioritization fails
            return digit_detections

    async def detect_objects(self, image_data: bytes, blue_box_coords: dict = None) -> dict:
        """Run object detection on image data using real YOLOv8 models"""
        if not self.model:
            raise Exception("YOLO model not loaded")

        try:
            # Convert bytes to image for processing
            image = Image.open(io.BytesIO(image_data))

            model_type = self.model.get("type", "unknown") if isinstance(self.model, dict) else "unknown"
            training_type = self.model.get("training_type", "unknown") if isinstance(self.model, dict) else "unknown"

            # Try to use real YOLO inference with loaded model
            detections = []
            use_mock = False  # Default to not using mocks

            # Determine which model file to use
            model_path = None
            detected_merge_type = None

            if model_type == "merged" or training_type == "merged":
                # Use digits_colors_merged model for full merged detection
                merged_file = self.merged_dir / "digits_colors_merged.pt"
                if merged_file.exists():
                    model_path = str(merged_file)
                    # Try to read metadata to determine merge type
                    metadata_file = merged_file.with_suffix('.metadata.json')
                    if metadata_file.exists():
                        try:
                            import json
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                detected_merge_type = metadata.get('merge_type', 'combined')
                                print(f"Using merged model: {model_path} (type: {detected_merge_type})")
                        except Exception as e:
                            print(f"Warning: Could not read metadata for {merged_file}: {e}")
                            print(f"Using merged model: {model_path}")
                    else:
                        print(f"Using merged model: {model_path}")
                else:
                    print(f"Merged model not found: {merged_file}, using base model")
                    use_mock = True
            elif training_type in ["digits", "colors"]:
                # Use individual merged models (base + individual LoRA)
                merged_file = self.merged_dir / f"{training_type}_merged.pt"
                if merged_file.exists():
                    model_path = str(merged_file)
                    # Try to read metadata
                    metadata_file = merged_file.with_suffix('.metadata.json')
                    if metadata_file.exists():
                        try:
                            import json
                            with open(metadata_file, 'r') as f:
                                metadata = json.load(f)
                                detected_merge_type = metadata.get('merge_type', training_type)
                                print(f"Using {training_type} merged model: {model_path} (type: {detected_merge_type})")
                        except Exception as e:
                            print(f"Warning: Could not read metadata for {merged_file}: {e}")
                            print(f"Using {training_type} merged model: {model_path}")
                    else:
                        print(f"Using {training_type} merged model: {model_path}")
                        detected_merge_type = training_type
                else:
                    print(f"{training_type} merged model not found: {merged_file}, using base model")
                    use_mock = True
            else:
                # Base model - use YOLOv8n directly
                base_model = self.models_dir / "base" / "yolov8n.pt"
                if base_model.exists():
                    model_path = str(base_model)
                    print(f"Using base YOLO model: {model_path}")
                else:
                    print(f"Base model not found: {base_model}, using mock detections")
                    use_mock = True

            # Update training_type based on detected merge type
            if detected_merge_type:
                training_type = detected_merge_type

            if not use_mock and model_path:
                try:
                    # Import YOLO and try real inference
                    from ultralytics import YOLO

                    print(f"Loading YOLO model: {model_path}")
                    yolo_model = YOLO(model_path)

                    print("Running inference...")
                    results = yolo_model(image, conf=0.25)  # Lower confidence threshold

                    # Process real YOLO results and separate by detection type
                    digit_detections = []
                    color_detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                # Get box coordinates
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                confidence = box.conf[0].cpu().numpy()
                                class_id = int(box.cls[0].cpu().numpy())

                                # Map class_id to label based on model type
                                label = self._map_class_id_to_label(class_id, training_type, model_type)

                                # Filter out invalid/unknown classes - only allow valid digits and colors
                                valid_digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
                                valid_colors = ['red', 'blue', 'green', 'yellow', 'orange', 'purple']

                                if label in valid_digits:
                                    digit_detections.append({
                                        "label": label,
                                        "score": float(confidence),
                                        "box": {
                                            "xMin": float(x1),
                                            "yMin": float(y1),
                                            "xMax": float(x2),
                                            "yMax": float(y2)
                                        }
                                    })
                                elif label in valid_colors:
                                    color_detections.append({
                                        "label": label,
                                        "score": float(confidence),
                                        "box": {
                                            "xMin": float(x1),
                                            "yMin": float(y1),
                                            "xMax": float(x2),
                                            "yMax": float(y2)
                                        }
                                    })

                    print(f"Real model detected: {len(digit_detections)} digits, {len(color_detections)} colors")

                except Exception as e:
                    print(f"Failed to use real YOLO model ({model_type}/{training_type}): {e}")
                    print("Falling back to mock detections...")
                    use_mock = True

            # Use mock detections if real model failed or no detections found
            if use_mock:
                digit_detections, color_detections = self._get_mock_detections(training_type, model_type)
                print(f"Using mock detections: {len(digit_detections)} digits, {len(color_detections)} colors")

            # Apply blue box prioritization for digit detections if coordinates provided
            if blue_box_coords:
                digit_detections = self._prioritize_digits_in_blue_box(digit_detections, blue_box_coords)

            return {
                "digitDetections": digit_detections,
                "colorDetections": color_detections,
                "mock": use_mock,
                "model_type": model_type,
                "training_type": training_type,
                "total_detections": len(digit_detections) + len(color_detections),
                "digital_count": len(digit_detections),
                "color_count": len(color_detections),
                "model_path": model_path if 'model_path' in locals() else None
            }

        except Exception as e:
            raise Exception(f"YOLO inference error: {e}")

    def get_tags(self, detection_mode: str) -> dict:
        """Get available tags for detection mode"""
        if detection_mode == "digits":
            return {"tags": ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']}
        elif detection_mode == "colors":
            return {"tags": ['red', 'blue', 'green', 'yellow', 'orange', 'purple']}
        else:
            return {"tags": []}

    def get_status(self) -> dict:
        """Get current status of YOLO service"""
        return {
            'loaded': self.model is not None,
            'model_type': 'Vision Service',
            'source': 'ultralytics'
        }
