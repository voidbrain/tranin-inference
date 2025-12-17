"""
Vision Models Module
Handles model loading, saving, and management operations
"""


class VisionModelsMixin:
    """Mixin class containing model management functionality"""

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

    async def reset_model(self) -> dict:
        """Reset to base YOLO model"""
        self.model = None
        self.using_base_model = True
        return {"message": "Model reset to base YOLOv8n"}

    def get_merged_model_status(self) -> dict:
        """Get status of merged models"""
        try:
            merged_files = []

            # Check for merged models
            for file_path in self.merged_dir.glob("*"):
                if file_path.is_file():
                    import datetime
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

    def get_merged_model_status_endpoint(self) -> dict:
        """API endpoint wrapper for getting merged model status"""
        return self.get_merged_model_status()

    def get_training_logs_endpoint(self) -> dict:
        """API endpoint wrapper for getting training logs"""
        return self.get_training_logs()
