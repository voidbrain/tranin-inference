"""
Speech Endpoints Module
FastAPI endpoint wrappers for speech service functionality
"""


class SpeechEndpointsMixin:
    """Mixin class containing API endpoint wrappers"""

    async def transcribe_audio_endpoint(self, audio_file, language: str) -> dict:
        """API endpoint wrapper for transcribing audio files"""
        from fastapi import HTTPException

        try:
            if not audio_file:
                raise HTTPException(status_code=400, detail="No audio file provided")

            print(f"Transcription request received with language parameter: {language}")

            # Map frontend language codes to Whisper codes
            # For specific languages, force that language
            # For multilingual, allow auto-detection but restrict to supported languages
            if language == 'multi':
                whisper_lang = None  # Auto-detect
                restrict_to_supported = True
                print("Using multilingual mode")
            else:
                # Force the selected language
                lang_map = {
                    'en': 'en',
                    'it': 'it',
                }
                whisper_lang = lang_map.get(language, 'en')  # Default to English if unknown
                restrict_to_supported = False
                print(f"Using specific language mode: whisper_lang={whisper_lang}, restrict_to_supported={restrict_to_supported}")

            result = await self.transcribe_audio(audio_file, whisper_lang, restrict_to_supported_langs=restrict_to_supported)
            print(f"Transcription result: {result}")
            return result

        except Exception as e:
            print(f"Transcription endpoint error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Transcription endpoint error: {str(e)}")

    def get_whisper_status_endpoint(self) -> dict:
        """API endpoint wrapper for getting Whisper service status"""
        return self.get_status()

    def get_whisper_training_status_endpoint(self) -> dict:
        """API endpoint wrapper for getting Whisper training status"""
        return self.get_training_status()

    def get_whisper_training_status_details_endpoint(self) -> dict:
        """API endpoint wrapper for getting detailed Whisper training status"""
        return self.get_training_status_details()

    async def start_whisper_lora_fine_tuning_endpoint(self, request, background_tasks) -> dict:
        """API endpoint wrapper for starting LoRA fine-tuning"""
        from fastapi import HTTPException

        try:
            # Check if we have training data
            status = self.get_training_status()
            if status.get("training_samples", 0) == 0:
                raise HTTPException(
                    status_code=400,
                    detail="No speech training data found. Upload audio samples first."
                )

            # Determine language (use first available language from data)
            language = status.get("available_languages", ["en"])[0] if status.get("available_languages") else "en"

            # Start LoRA fine-tuning in background
            background_tasks.add_task(
                self.fine_tune_lora,
                language=language,
                epochs=request.epochs
            )

            return {
                "message": f"Whisper LoRA fine-tuning started for language: {language}",
                "language": language,
                "epochs": request.epochs,
                "status": "background_processing"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def train_language_lora_endpoint(self, language: str, background_tasks) -> dict:
        """API endpoint wrapper for training language LoRA"""
        from fastapi import HTTPException

        try:
            background_tasks.add_task(self.train_specialized_lora, language)
            return {
                "message": f"{language} LoRA training started",
                "language": language,
                "status": "background_processing"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def create_merged_model_endpoint(self, background_tasks) -> dict:
        """API endpoint wrapper for creating merged model"""
        from fastapi import HTTPException

        try:
            background_tasks.add_task(self.create_merged_models)
            return {
                "message": "Merged Whisper model creation started",
                "status": "background_processing"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def whisper_fine_tune_lora_endpoint(self, training_data: dict, background_tasks) -> dict:
        """API endpoint wrapper for LoRA fine-tuning (frontend expects whisper-fine-tune-lora)"""
        from fastapi import HTTPException

        try:
            language = training_data.get("language", "en")
            epochs = training_data.get("epochs", 5)

            background_tasks.add_task(
                self.fine_tune_lora,
                language=language,
                epochs=epochs,
                output_dir=None
            )

            return {
                "message": f"Whisper LoRA fine-tuning started for language: {language}",
                "language": language,
                "epochs": epochs,
                "status": "background_processing"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def whisper_training_status_details_endpoint(self) -> dict:
        """API endpoint wrapper for detailed training status (frontend expects whisper-training-status-details)"""
        return self.get_training_status_details()
