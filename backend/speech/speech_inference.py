"""
Speech Inference Module
Handles audio transcription and inference operations
"""
import os
import tempfile
import subprocess
from pathlib import Path
from fastapi import UploadFile

# Lazy imports - loaded on first use to avoid import errors during configuration
_whisper = None
_librosa = None

def _import_ml_libraries():
    """Lazy import of ML libraries"""
    global _whisper, _librosa
    if _whisper is None:
        import whisper
        import librosa
        _whisper = whisper
        _librosa = librosa

def _get_whisper():
    _import_ml_libraries()
    return _whisper

def _get_librosa():
    _import_ml_libraries()
    return _librosa


class SpeechInferenceMixin:
    """Mixin class containing inference/transcription functionality"""

    async def transcribe_audio(self, audio_file: UploadFile, language: str = None, restrict_to_supported_langs: bool = False) -> dict:
        """Transcribe audio file using Whisper"""
        if not self.model:
            raise Exception("Whisper model not loaded")

        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_file:
            contents = await audio_file.read()  # Read as bytes (binary)
            temp_file.write(contents)
            temp_file_path = temp_file.name

        try:
            # Load audio with proper preprocessing
            # Whisper expects 16kHz mono audio
            # Handle WebM files by converting to WAV first

            # Check if file is WebM and convert to WAV if needed
            if temp_file_path.endswith('.webm'):
                wav_temp_path = temp_file_path.replace('.webm', '.wav')
                try:
                    # Use ffmpeg to extract audio from WebM
                    subprocess.run([
                        'ffmpeg', '-y', '-i', temp_file_path,
                        '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1',
                        wav_temp_path
                    ], check=True, capture_output=True)
                    temp_file_path = wav_temp_path  # Use the converted file
                except subprocess.CalledProcessError as e:
                    raise Exception(f"Failed to convert WebM to WAV: {e}")

            # Try soundfile first, fallback to librosa
            try:
                import soundfile as sf
                audio_array, sample_rate = sf.read(temp_file_path, dtype='float32')
                # Convert to mono if needed
                if len(audio_array.shape) > 1:
                    audio_array = audio_array.mean(axis=1)
                # Resample to 16kHz if needed
                if sample_rate != 16000:
                    librosa = _get_librosa()
                    audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            except ImportError:
                # Fallback to librosa
                librosa = _get_librosa()
                audio_array, _ = librosa.load(temp_file_path, sr=16000, mono=True, dtype='float32')

            # Ensure consistent dtype for Whisper
            import numpy as np
            audio_array = audio_array.astype(np.float32)

            # Run transcription
            options = {
                'task': 'transcribe',
                'language': language if language else None,
                'verbose': False
            }

            # If restricting to supported languages (en/it only), add language detection restriction
            if restrict_to_supported_langs and language is None:
                # For multilingual mode, we still allow auto-detection but should prefer en/it
                # Whisper doesn't have a direct "allowed languages" parameter, but we can post-process
                pass

            result = self.model.transcribe(audio_array, **options)

            # Post-process language detection to restrict to supported languages
            detected_lang = result.get('language', language)
            if restrict_to_supported_langs and detected_lang not in ['en', 'it']:
                # Force to English as fallback since it's more common
                print(f"Detected unsupported language '{detected_lang}', forcing to English for multilingual mode")
                detected_lang = 'en'
                # Re-run transcription with English forced
                options['language'] = 'en'
                result = self.model.transcribe(audio_array, **options)
            elif language is not None and not restrict_to_supported_langs:
                # For specific language selection, use the selected language instead of detected
                detected_lang = language
                print(f"Specific language '{language}' selected, using it instead of detected '{result.get('language', 'unknown')}'")

            return {
                'transcription': result['text'].strip(),
                'language': detected_lang,
                'confidence': result.get('confidence', 0.0),
                'duration': len(audio_array) / 16000,
                'filename': audio_file.filename
            }

        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
        finally:
            # Clean up temp files
            try:
                os.unlink(temp_file_path)
                # Also clean up converted WAV file if it exists
                if temp_file_path.endswith('.webm'):
                    wav_temp_path = temp_file_path.replace('.webm', '.wav')
                    if os.path.exists(wav_temp_path):
                        os.unlink(wav_temp_path)
            except:
                pass

    def _load_model(self, model_path: str = None):
        """Load Whisper model from local path or download if necessary"""
        try:
            if model_path and Path(model_path).exists():
                print(f"Loading local Whisper model: {model_path}")
                whisper = _get_whisper()
                self.model = whisper.load_model(model_path)
                print("Local Whisper model loaded successfully")
            else:
                print("Loading Whisper model (tiny) from remote...")
                whisper = _get_whisper()
                self.model = whisper.load_model("tiny")
                print("Remote Whisper model loaded successfully")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            raise

    def get_status(self) -> dict:
        """Get current status of Whisper service"""
        return {
            'status': 'Backend-powered',
            'model': f'Whisper {self.model.model_type}' if self.model else 'None',
            'ready': self.model is not None,
            'cache_dir': str(self.models_dir)
        }
