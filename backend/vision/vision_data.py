"""
Vision Data Module
Handles training data management and processing
"""
import datetime
from pathlib import Path


class VisionDataMixin:
    """Mixin class containing data management functionality"""

    async def process_training_data(self, training_data: dict):
        """Process training data with labels and bounding boxes, splitting by detection type"""
        try:
            # Extract image data (base64) and detections
            image_data = training_data.get("imageData", "")
            detections = training_data.get("detections", [])

            if not image_data.startswith("data:image/"):
                raise Exception("Invalid image data format")

            # Decode base64 image
            import base64
            import io
            from PIL import Image

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

    def get_specialized_training_count_endpoint(self, training_type: str) -> dict:
        """API endpoint wrapper for getting specialized training count"""
        return self.get_specialized_training_count(training_type)

    def get_training_queue_status_endpoint(self, page: int = 1, per_page: int = 20) -> dict:
        """API endpoint wrapper for getting training queue status with pagination"""
        return self.get_training_queue_status(page, per_page)

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
