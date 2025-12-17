"""
Speech Models Module
Handles model loading, saving, and management operations
"""


class SpeechModelsMixin:
    """Mixin class containing model management functionality"""

    def _load_lora_adapter_if_available(self):
        """Try to load the most recent merged model if available"""
        try:
            # First try to load multilingual model if available
            multilang_file = self.merged_dir / "speech_multilang.pt"
            if multilang_file.exists():
                print(f"Auto-loading multilingual merged model: {multilang_file}")
                success = self.load_lora_adapter(str(multilang_file), "multilingual")
                if success:
                    self.using_lora = True
                    print(f"✓ Auto-loaded multilingual merged model")
                else:
                    print(f"✗ Failed to load multilingual merged model")
                return

            # Then try individual language models in preference order
            lang_order = ["en.pt", "it.pt"]  # English first, then Italian
            for lang_file in lang_order:
                lang_model = self.merged_dir / f"speech_{lang_file}"
                if lang_model.exists():
                    if lang_file == "en.pt":
                        language = "en"
                    elif lang_file == "it.pt":
                        language = "it"

                    print(f"Auto-loading {language} merged model: {lang_model}")
                    success = self.load_lora_adapter(str(lang_model), language)
                    if success:
                        self.using_lora = True
                        print(f"✓ Auto-loaded {language} merged model from {lang_model}")
                    else:
                        print(f"✗ Failed to load {language} merged model")
                    return

            # Fall back to old LoRA adapter loading
            print("No merged models found, trying old LoRA adapters...")
            lora_dirs = [d for d in self.models_dir.iterdir() if d.is_dir() and d.name.startswith("whisper-lora-")]
            if not lora_dirs:
                print("No LoRA adapters found - using base Whisper model")
                return

            # Sort by modification time to find the most recent
            lora_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
            latest_lora_dir = lora_dirs[0]
            model_path = latest_lora_dir / "final_model"

            if not model_path.exists():
                print(f"No final_model directory found in {latest_lora_dir}")
                return

            # Extract language from directory name
            language = "unknown"
            if "-en" in latest_lora_dir.name:
                language = "en"
            elif "-it" in latest_lora_dir.name:
                language = "it"

            success = self.load_lora_adapter(str(model_path), language)
            if success:
                self.using_lora = True
                print(f"Auto-loaded LoRA adapter: {language} from {latest_lora_dir}")
            else:
                print("Failed to auto-load LoRA adapter")

        except Exception as e:
            print(f"Warning: Could not auto-load LoRA adapter: {e}")

    def load_lora_adapter(self, model_path: str, language: str) -> bool:
        """Load a previously fine-tuned LoRA adapter for inference"""
        try:
            from peft import PeftModel
            from transformers import WhisperForConditionalGeneration

            if "openai" in str(model_path):
                raise Exception("Cannot load OpenAI Whisper with PEFT LoRA - incompatible formats")

            print(f"Loading LoRA adapter from: {model_path}")

            # Load the PEFT model
            fine_tuned_model = PeftModel.from_pretrained(
                WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny"),
                model_path
            )

            # Update our model reference (replace OpenAI Whisper with PEFT model)
            self.model = fine_tuned_model
            print(f"LoRA adapter loaded successfully for language: {language}")

            return True

        except Exception as e:
            print(f"Failed to load LoRA adapter: {e}")
            return False

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

    def load_language_model_endpoint(self, language: str) -> dict:
        """API endpoint wrapper for loading a language model"""
        try:
            # For specific languages, try to load the language-specific model first
            if language in ['en', 'it']:
                specific_model = self.merged_dir / f"speech_{language}.pt"
                if specific_model.exists() and specific_model.stat().st_size > 1000000:
                    print(f"Loading language-specific model for {language}: {specific_model}")
                    self._load_model(str(specific_model))
                    model_type = f"speech_{language}"
                    print(f"✓ Successfully loaded {language}-specific model")
                else:
                    print(f"No {language}-specific model found, loading base model")
                    self._load_model()
                    model_type = "whisper-tiny"
            else:
                # For multilingual or other cases, try multilingual model first
                multilang_model = self.merged_dir / "speech_multilang.pt"
                if multilang_model.exists() and multilang_model.stat().st_size > 1000000:
                    print(f"Loading multilingual model: {multilang_model}")
                    self._load_model(str(multilang_model))
                    model_type = "speech_multilang"
                    print(f"✓ Successfully loaded multilingual model")
                else:
                    print(f"No multilingual model found, loading base model")
                    self._load_model()
                    model_type = "whisper-tiny"

            if self.model is not None:
                return {
                    "message": f"Successfully loaded Whisper model for {language}",
                    "language": language,
                    "model_type": model_type,
                    "status": "success"
                }
            else:
                raise Exception("Failed to load Whisper model")

        except Exception as e:
            return {
                "message": f"Failed to load language model: {str(e)}",
                "language": language,
                "status": "error",
                "error": str(e)
            }

    async def load_whisper_lora_adapter_endpoint(self, adapter_path: str) -> dict:
        """API endpoint wrapper for loading a Whisper LoRA adapter"""
        from fastapi import HTTPException

        try:
            success = self.load_lora_adapter(adapter_path, "unknown")
            if success:
                return {"message": "LoRA adapter loaded successfully", "adapter_path": adapter_path}
            else:
                raise HTTPException(status_code=500, detail="Failed to load LoRA adapter")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading LoRA adapter: {str(e)}")

    def get_merged_model_status_endpoint(self) -> dict:
        """API endpoint wrapper for getting merged model status"""
        return self.get_merged_model_status()
