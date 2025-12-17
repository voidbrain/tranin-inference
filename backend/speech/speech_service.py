"""
Whisper Service for Speech Recognition
Main service class that combines all speech functionality via mixins
"""
import os
import datetime
from pathlib import Path

from .speech_inference import SpeechInferenceMixin, _get_whisper
from .speech_training import SpeechTrainingMixin
from .speech_data import SpeechDataMixin
from .speech_models import SpeechModelsMixin
from .speech_endpoints import SpeechEndpointsMixin


class SpeechService(SpeechInferenceMixin, SpeechTrainingMixin, SpeechDataMixin, SpeechModelsMixin, SpeechEndpointsMixin):
    """Service for handling Whisper speech recognition functionality"""

    def __init__(self, models_dir: str = "models/speech", data_dir: str = "data/speech"):
        # Initialize paths
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        # Create consistent subdirectory structure
        self.processed_dir = self.data_dir / "processed"
        self.train_dir = self.data_dir / "waiting"
        self.processed_dir.mkdir(exist_ok=True)
        self.train_dir.mkdir(exist_ok=True)

        # Configure cache directory for Whisper models
        whisper_cache_dir = self.models_dir / "merged"
        os.environ['WHISPER_CACHE_DIR'] = str(whisper_cache_dir)

        # Merged models directory
        self.merged_dir = Path("models/speech/merged")
        self.merged_dir.mkdir(exist_ok=True, parents=True)

        # Training logs directory
        self.train_logs_dir = self.models_dir / "train_logs"
        self.train_logs_dir.mkdir(exist_ok=True, parents=True)

        # Model state
        self.model = None
        self.using_lora = False

        # Training status tracking
        self.training_status = "idle"  # idle, running, success, error
        self.training_progress = 0.0
        self.training_message = ""
        self.training_logs = []
        self.training_start_time = None
        self.training_language = None

        # Auto-load best available model
        self._load_lora_adapter_if_available()

    @classmethod
    def get_service_config(cls):
        """Return the service configuration with endpoints"""
        return {
            "endpoints": [
                {
                    "path": "/speech/status",
                    "methods": ["GET"],
                    "handler": "get_whisper_status_endpoint"
                },
                {
                    "path": "/speech/training-count",
                    "methods": ["GET"],
                    "handler": "speech_training_count_endpoint"
                },
                {
                    "path": "/speech/training-status",
                    "methods": ["GET"],
                    "handler": "get_whisper_training_status_endpoint"
                },
                {
                    "path": "/speech/training-status-details",
                    "methods": ["GET"],
                    "handler": "get_whisper_training_status_details_endpoint"
                },
                {
                    "path": "/speech/lora-fine-tune",
                    "methods": ["POST"],
                    "handler": "start_whisper_lora_fine_tuning_endpoint",
                    "params": ["request", "background_tasks"]
                },
                {
                    "path": "/speech/load-lora-adapter",
                    "methods": ["POST"],
                    "handler": "load_whisper_lora_adapter_endpoint",
                    "params": ["adapter_path: str"]
                },
                {
                    "path": "/speech/train-language-lora/{language}",
                    "methods": ["POST"],
                    "handler": "train_language_lora_endpoint",
                    "params": ["language: str", "background_tasks"]
                },
                {
                    "path": "/speech/create-merged-model",
                    "methods": ["POST"],
                    "handler": "create_merged_model_endpoint",
                    "params": ["background_tasks"]
                },
                {
                    "path": "/speech/merged-model-status",
                    "methods": ["GET"],
                    "handler": "get_merged_model_status_endpoint"
                },
                {
                    "path": "/speech/load-language-model",
                    "methods": ["POST"],
                    "handler": "load_language_model_endpoint",
                    "params": ["language: str"]
                },
                {
                    "path": "/speech/transcribe-audio",
                    "methods": ["POST"],
                    "handler": "transcribe_audio_endpoint",
                    "params": ["audio_file: UploadFile", "language"]
                },
                {
                    "path": "/speech/upload-speech-training-data",
                    "methods": ["POST"],
                    "handler": "upload_speech_training_data_endpoint_alt",
                    "params": ["request"]
                },
                {
                    "path": "/speech/whisper-fine-tune-lora",
                    "methods": ["POST"],
                    "handler": "whisper_fine_tune_lora_endpoint",
                    "params": ["training_data: dict", "background_tasks"]
                },
                {
                    "path": "/speech/whisper-training-status-details",
                    "methods": ["GET"],
                    "handler": "whisper_training_status_details_endpoint"
                },
                {
                    "path": "/speech/clear-training-data/{language}",
                    "methods": ["DELETE"],
                    "handler": "clear_training_data_endpoint",
                    "params": ["language: str"]
                }
            ]
        }

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
