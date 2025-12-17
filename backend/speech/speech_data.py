"""
Speech Data Module
Handles training data management and processing
"""
import datetime
from pathlib import Path


class SpeechDataMixin:
    """Mixin class containing data management functionality"""

    async def process_speech_training_data(self, audio_blob: bytes, language: str, transcript: str, original_filename: str = None):
        """Process and store speech training data for LoRA fine-tuning"""
        # Save uploaded training data to waiting directory (not processed, since it's new/unused)
        try:
            # Create language-specific subdirectory in waiting
            lang_dir = self.train_dir / language
            lang_dir.mkdir(exist_ok=True)

            timestamp = Path(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))

            # Keep original file extension - prefer WebM for training data
            if original_filename and '.' in original_filename:
                original_ext = Path(original_filename).suffix.lower()
                if original_ext in ['.webm', '.mp3', '.wav', '.m4a', '.ogg', '.flac']:
                    audio_ext = original_ext
                else:
                    audio_ext = '.webm'  # fallback
            else:
                audio_ext = '.webm'  # default for training data

            audio_filename = f"speech_{timestamp}_{language}_training{audio_ext}"
            transcript_filename = f"speech_{timestamp}_{language}_training.txt"

            # Save audio file to language-specific waiting directory
            audio_path = lang_dir / audio_filename
            with open(audio_path, 'wb') as f:
                f.write(audio_blob)

            # Save transcript to language-specific waiting directory
            transcript_path = lang_dir / transcript_filename
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)

            return {
                'audio_path': str(audio_path),
                'transcript_path': str(transcript_path),
                'language': language
            }

        except Exception as e:
            raise Exception(f"Failed to process speech training data: {str(e)}")

    async def clear_training_data_endpoint(self, language: str) -> dict:
        """API endpoint wrapper for clearing training data for a specific language"""
        try:
            # Find and delete training files for the specified language
            lang_dir = self.train_dir / language
            if lang_dir.exists():
                deleted_count = 0
                # Only delete files that match the specific language pattern in this directory
                for file_path in lang_dir.glob(f"speech_*_{language}_training.*"):
                    if file_path.is_file():
                        file_path.unlink()  # Delete the file
                        deleted_count += 1

            # Keep the directory structure intact for future uploads
            # Only remove files, not directories

                return {
                    "message": f"Cleared {deleted_count} training files for language '{language}'",
                    "language": language,
                    "files_deleted": deleted_count
                }
            else:
                return {
                    "message": f"No training data found for language '{language}'",
                    "language": language,
                    "files_deleted": 0
                }

        except Exception as e:
            raise Exception(f"Failed to clear training data for language '{language}': {str(e)}")

    def get_speech_training_count_endpoint(self) -> dict:
        """API endpoint wrapper for getting count of speech training samples"""
        try:
            # Get training status which reads from file system
            status = self.get_training_status()
            count = status.get("training_samples", 0)
            language_counts = status.get("language_counts", {})
            return {
                "count": count,
                "language_counts": language_counts,
                "message": "Speech training data count"
            }
        except Exception as e:
            return {"error": str(e), "count": 0, "language_counts": {}}

    def speech_training_count_endpoint(self) -> dict:
        """Frontend expects this exact endpoint: speech-training-count (direct naming)"""
        return self.get_speech_training_count_endpoint()

    async def upload_speech_training_data_endpoint(self, audio_file, language: str = "Form(...)", transcript: str = "Form(...)") -> dict:
        """API endpoint wrapper for uploading speech training data"""
        from fastapi import HTTPException

        try:
            # Read the audio file
            audio_bytes = await audio_file.read()

            # Use the provided language directly, unless it's "multi"
            actual_language = language
            if language == "multi":
                # For multilingual uploads, try to detect language from transcription context
                # For now, default to English as we don't have transcription here
                actual_language = "en"

            # Process and store the speech training data
            result = await self.process_speech_training_data(
                audio_bytes, actual_language, transcript, audio_file.filename
            )

            return {
                "message": "Speech training data uploaded successfully",
                "audio_path": result["audio_path"],
                "transcript_path": result["transcript_path"],
                "language": result["language"]
            }

        except Exception as e:
            raise Exception(f"Speech training data upload failed: {str(e)}")

    async def upload_speech_training_data_endpoint_alt(self, request):
        """Alternative endpoint for speech training data upload (frontend expects upload-speech-training-data)"""
        from fastapi import HTTPException, Request

        try:
            # Parse multipart form data manually
            form_data = await request.form()
            file = form_data.get("audio_file")
            language = form_data.get("language") or "en"
            transcript = form_data.get("transcript") or ""

            # Handle UploadFile objects
            if hasattr(file, 'read'):
                audio_bytes = await file.read()
                filename = file.filename
            else:
                raise HTTPException(status_code=400, detail="No audio file provided")

            result = await self.process_speech_training_data(audio_bytes, language, transcript)
            return {
                "message": "Speech training data uploaded successfully",
                "filename": filename,
                "language": language,
                "transcript": transcript,
                "audio_size": len(audio_bytes),
                "result": result
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Speech training data upload failed: {str(e)}")
