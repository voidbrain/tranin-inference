"""
Vision Endpoints Module
FastAPI endpoint wrappers for vision service functionality
"""


class VisionEndpointsMixin:
    """Mixin class containing API endpoint wrappers"""

    async def detect_objects_endpoint(self, request, model: str = "base") -> dict:
        """API endpoint wrapper for object detection"""
        from fastapi import Request
        # Parse multipart form data manually to get both file and blue_box
        form_data = await request.form()

        # Debug: Print all form fields
        print(f"🔍 FORM DATA DEBUG: Available fields: {list(form_data.keys())}")
        for key, value in form_data.items():
            print(f"🔍 FORM DATA DEBUG: {key} = {type(value)} (length: {len(str(value)) if hasattr(value, '__len__') else 'N/A'})")

        file = form_data.get("file")
        blue_box_str = form_data.get("blue_box")

        if not file:
            raise Exception("No file provided")

        file_bytes = await file.read()

        # Parse blue box coordinates if provided (from FormData)
        blue_box_coords = None
        if blue_box_str:
            try:
                # blue_box_str should be a string containing JSON
                if isinstance(blue_box_str, str):
                    blue_box_coords = __import__('json').loads(blue_box_str)
                    print(f"🔵 BLUE BOX DEBUG: Successfully parsed coordinates: {blue_box_coords}")
                else:
                    print(f"❌ BLUE BOX ERROR: Unexpected blue_box type: {type(blue_box_str)}, value: {repr(blue_box_str)}")
            except Exception as e:
                print(f"❌ BLUE BOX ERROR: Failed to parse blue box coordinates: {e}")
                print(f"❌ BLUE BOX ERROR: Raw blue_box value: {repr(blue_box_str)}")
                blue_box_coords = None
        else:
            print(f"⚠️  BLUE BOX WARNING: No blue_box parameter received in request")

        # Set the model based on the frontend selection
        if model == "base":
            self.model = None
            self.using_base_model = True
        elif model in ["digits", "colors", "merged"]:
            # Load the appropriate model
            if model == "merged":
                await self.load_merged_model({"training_type": "merged"})
            else:
                await self.load_lora_adapter({"training_type": model})

        return await self.detect_objects(file_bytes, blue_box_coords)

    def get_model_status_endpoint(self) -> dict:
        """API endpoint wrapper for getting model status"""
        return self.get_status()

    def get_tags_endpoint(self, detection_mode: str) -> dict:
        """API endpoint wrapper for getting available tags"""
        return self.get_tags(detection_mode)

    async def train_digits_lora_endpoint(self, background_tasks) -> dict:
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

    async def train_colors_lora_endpoint(self, background_tasks) -> dict:
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

    async def create_merged_model_endpoint(self, background_tasks) -> dict:
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

    async def train_lora_specialized_endpoint(self, training_data: dict, background_tasks) -> dict:
        """API endpoint wrapper for specialized LoRA training (frontend expects this endpoint)"""
        from fastapi import HTTPException

        try:
            training_type = training_data.get("training_type", "digits")
            if training_type not in ["digits", "colors", "combined"]:
                raise HTTPException(status_code=400, detail="Invalid training type")

            # Handle combined training (digits+colors)
            if training_type == "combined":
                background_tasks.add_task(self.train_combined_lora)
                return {
                    "message": "Combined LoRA training started",
                    "training_type": "combined",
                    "status": "background_processing"
                }

            # Handle regular specialized training
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

    async def train_combined_lora_endpoint(self, background_tasks) -> dict:
        """API endpoint wrapper for combined LoRA training (digits + colors)"""
        from fastapi import HTTPException

        try:
            background_tasks.add_task(self.train_combined_lora)
            return {
                "message": "Combined LoRA training started",
                "training_type": "combined",
                "status": "background_processing"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def reset_model_endpoint(self) -> dict:
        """API endpoint wrapper for resetting model"""
        return await self.reset_model()

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
