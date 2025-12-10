"""
Whisper Service for Speech Recognition
Handles speech transcription and audio processing
"""
import os
import shutil
import tempfile
import datetime
from pathlib import Path
from fastapi import UploadFile
import whisper
import torch
import librosa
import numpy as np

class SpeechService:
    """Service for handling Whisper speech recognition functionality"""

    def __init__(self, models_dir: str = "whisper_models", data_dir: str = "whisper_data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.data_dir.mkdir(exist_ok=True)

        # Create consistent subdirectory structure like YOLO
        self.processed_dir = self.data_dir / "processed"
        self.train_dir = self.data_dir / "train"
        self.processed_dir.mkdir(exist_ok=True)
        self.train_dir.mkdir(exist_ok=True)

        # Configure cache directory for Whisper models (doesn't work, use download_root instead)
        whisper_cache_dir = self.models_dir
        os.environ['WHISPER_CACHE_DIR'] = str(whisper_cache_dir)

        self.model = None
        self.using_lora = False

        # Training status tracking
        self.training_status = "idle"  # idle, running, success, error
        self.training_progress = 0.0
        self.training_message = ""
        self.training_logs = []
        self.training_start_time = None
        self.training_language = None

        self._load_model()
        self._load_lora_adapter_if_available()  # Try to load latest LoRA adapter if available

    def _load_model(self):
        """Load Whisper model, downloading if necessary"""
        try:
            print("Loading Whisper model (tiny)...")
            self.model = whisper.load_model("tiny")
            print("Whisper model loaded successfully")
        except Exception as e:
            print(f"Failed to load Whisper model: {e}")
            raise

    async def transcribe_audio(self, audio_file: UploadFile, language: str = None) -> dict:
        """Transcribe audio file using Whisper"""
        if not self.model:
            raise Exception("Whisper model not loaded")

        # Create temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            contents = await audio_file.read()
            temp_file.write(contents)
            temp_file_path = temp_file.name

        try:
            # Load audio with proper preprocessing
            # Whisper expects 16kHz mono audio
            audio_array, _ = librosa.load(temp_file_path, sr=16000, mono=True)

            # Run transcription
            options = {
                'task': 'transcribe',
                'language': language if language else None,
                'verbose': False
            }

            result = self.model.transcribe(audio_array, **options)

            return {
                'transcription': result['text'].strip(),
                'language': result.get('language', language),
                'confidence': result.get('confidence', 0.0),
                'duration': len(audio_array) / 16000,
                'filename': audio_file.filename
            }

        except Exception as e:
            raise Exception(f"Transcription failed: {str(e)}")
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_file_path)
            except:
                pass

    def get_status(self) -> dict:
        """Get current status of Whisper service"""
        return {
            'status': 'Backend-powered',
            'model': f'Whisper {self.model.model_type}' if self.model else 'None',
            'ready': self.model is not None,
            'cache_dir': str(self.models_dir)
        }

    async def process_speech_training_data(self, audio_blob: bytes, language: str, transcript: str):
        """Process and store speech training data for LoRA fine-tuning"""
        # Save uploaded training data to processed directory
        try:
            timestamp = Path(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
            audio_filename = f"speech_{timestamp}_{language}_training.wav"
            transcript_filename = f"speech_{timestamp}_{language}_training.txt"

            # Save audio file to processed directory
            audio_path = self.processed_dir / audio_filename
            with open(audio_path, 'wb') as f:
                f.write(audio_blob)

            # Save transcript to processed directory
            transcript_path = self.processed_dir / transcript_filename
            with open(transcript_path, 'w', encoding='utf-8') as f:
                f.write(transcript)

            return {
                'audio_path': str(audio_path),
                'transcript_path': str(transcript_path),
                'language': language
            }

        except Exception as e:
            raise Exception(f"Failed to process speech training data: {str(e)}")

    def get_training_status_details(self) -> dict:
        """Get detailed training status for frontend polling"""
        return {
            "status": self.training_status,
            "progress": self.training_progress,
            "message": self.training_message,
            "language": self.training_language,
            "start_time": self.training_start_time.isoformat() if self.training_start_time else None,
            "logs": self.training_logs[-20:] if self.training_logs else [],  # Last 20 logs
            "log_count": len(self.training_logs)
        }

    async def fine_tune_lora(self, language: str = "en", epochs: int = 5, output_dir: str = None) -> dict:
        """Fine-tune Whisper model using LoRA for specific language/dataset"""
        try:
            # Set training status
            self.training_status = "running"
            self.training_progress = 0.0
            self.training_message = "Initializing training..."
            self.training_start_time = datetime.datetime.now()
            self.training_language = language
            self.training_logs = []
            self._add_training_log("Starting Whisper LoRA fine-tuning")
            self._add_training_log(f"Language: {language}, Epochs: {epochs}")

            # Import required libraries for LoRA fine-tuning
            from transformers import WhisperProcessor
            from datasets import Dataset, Audio
            from peft import LoraConfig, get_peft_model
            import pandas as pd

            # 1. Move data from processed to train for training
            self._add_training_log("Moving training data from processed to train...")
            self._move_from_processed_to_train(language)

            # 2. Prepare dataset from stored training data
            self.training_message = "Preparing training dataset..."
            self._add_training_log("Preparing training dataset...")
            training_files = list(self.train_dir.glob(f"speech_*_{language}_training.txt"))

            if not training_files:
                raise Exception(f"No training data found for language: {language}")

            self._add_training_log(f"Found {len(training_files)} training samples")

            # Load and preprocess training data
            data = []
            for txt_file in training_files:
                # Find corresponding audio file
                audio_file = txt_file.with_suffix('.wav')
                if audio_file.exists():
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()

                    data.append({
                        "audio": str(audio_file),
                        "text": transcript
                    })

            if not data:
                raise Exception("No valid training pairs found")

            self._add_training_log(f"Loaded {len(data)} complete audio-transcript pairs")
            # Create HF dataset
            dataset = Dataset.from_pandas(pd.DataFrame(data))
            self.training_progress = 10.0

            # 2. Configure LoRA
            print("Configuring LoRA adapters...")
            lora_config = LoraConfig(
                r=32,  # rank dimension
                lora_alpha=64,  # scaling parameter
                target_modules=["q_proj", "v_proj"],  # attention layers to adapt
                lora_dropout=0.1,
                bias="none"
            )

            # 3. Load processor and model with LoRA
            print("Loading Whisper processor and model with LoRA...")
            processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

            # Convert openai-whisper to HF format for LoRA
            # We'll use transformers' Whisper implementation for PEFT compatibility
            from transformers import WhisperForConditionalGeneration

            model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
            model = get_peft_model(model, lora_config)

            # 4. Data preprocessing
            print("Preprocessing audio data...")
            def preprocess_function(batch):
                audio = batch["audio"]

                # Load audio
                audio_array, sample_rate = librosa.load(audio, sr=16000)
                input_features = processor(audio_array, sampling_rate=sample_rate).input_features[0]

                # Process text
                labels = processor(text=batch["text"]).input_ids[0]

                return {
                    "input_features": input_features,
                    "labels": labels
                }

            processed_dataset = dataset.map(preprocess_function)

            # 5. Training setup
            print("Setting up training environment...")
            from transformers import TrainingArguments, Trainer

            output_dir = output_dir or self.models_dir / f"whisper-lora-{language}"
            output_dir = Path(output_dir)
            output_dir.mkdir(exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(output_dir),
                num_train_epochs=epochs,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=4,
                save_steps=100,
                logging_steps=25,
                learning_rate=1e-5,
                warmup_steps=50,
                save_total_limit=2,
                eval_strategy="no",
                load_best_model_at_end=False,
                metric_for_best_model="loss",
                greater_is_better=False,
                dataloader_pin_memory=False
            )

            # 6. Initialize trainer
            print("Starting LoRA fine-tuning...")
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=processed_dataset,
                tokenizer=processor,
            )

            # 7. Train the model
            trainer.train()

            # 8. Save the fine-tuned model
            final_model_path = output_dir / "final_model"
            trainer.save_model(str(final_model_path))

            # Save LoRA adapters separately
            model.save_pretrained(str(final_model_path))

            self.training_progress = 95.0
            self.training_message = "Saving model..."
            self._add_training_log("Saving fine-tuned model")

            print(f"LoRA fine-tuning completed! Model saved to: {final_model_path}")

            # 9. Clean up old LoRA models to save disk space
            self._cleanup_old_lora_models(output_dir.name)

            # 10. Move processed training data to avoid reuse
            self._move_processed_training_data(language)

            # Set success status
            self.training_status = "success"
            self.training_progress = 100.0
            self.training_message = "Training completed successfully!"
            self._add_training_log("Training completed successfully!")

            return {
                "status": "success",
                "model_path": str(final_model_path),
                "language": language,
                "epochs_trained": epochs,
                "training_samples": len(data),
                "lora_config": str(lora_config)
            }

        except Exception as e:
            # Set error status
            self.training_status = "error"
            self.training_progress = 0.0
            self.training_message = f"Training failed: {str(e)}"
            self._add_training_log(f"Training failed: {str(e)}")
            print(f"Whisper LoRA fine-tuning failed: {e}")
            raise Exception(f"LoRA fine-tuning failed: {str(e)}")

    def _load_lora_adapter_if_available(self):
        """Try to load the most recent LoRA adapter if available"""
        try:
            # Look for LORA model directories
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

    def _cleanup_old_lora_models(self, current_model_name: str):
        """Keep only the most recent LoRA model, remove older ones to save disk space"""
        try:
            # Look for LORA model directories
            lora_dirs = [d for d in self.models_dir.iterdir() if d.is_dir() and d.name.startswith("whisper-lora-")]
            if len(lora_dirs) <= 1:
                return  # No cleanup needed

            # Sort by modification time (newest first)
            lora_dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)

            # Keep the newest one, delete others
            current_model_dir = self.models_dir / current_model_name
            deleted_count = 0

            for lora_dir in lora_dirs:
                if lora_dir != current_model_dir and lora_dir.name != current_model_name:
                    try:
                        shutil.rmtree(str(lora_dir))
                        deleted_count += 1
                        print(f"Cleaned up old LoRA model: {lora_dir.name}")
                    except Exception as e:
                        print(f"Warning: Failed to remove old LoRA directory {lora_dir.name}: {e}")

            if deleted_count > 0:
                print(f"LoRA cleanup: kept latest model, deleted {deleted_count} old models")

        except Exception as e:
            print(f"Warning: LoRA model cleanup failed: {e}")

    def _move_from_processed_to_train(self, language: str = "all"):
        """Move training data from processed directory to train directory before training"""
        try:
            moved_count = 0

            # Find and move training files from processed to train
            if language == "all":
                patterns = ["speech_*_training.wav", "speech_*_training.txt"]
            else:
                patterns = [f"speech_*_{language}_training.wav", f"speech_*_{language}_training.txt"]

            for pattern in patterns:
                for file_path in self.processed_dir.glob(pattern):
                    if file_path.is_file():
                        train_path = self.train_dir / file_path.name
                        shutil.move(str(file_path), str(train_path))
                        moved_count += 1

            if moved_count > 0:
                print(f"Moved {moved_count} training files from processed to train")

        except Exception as e:
            print(f"Warning: Failed to move training data from processed to train: {e}")

    def _add_training_log(self, message: str):
        """Add a training log message"""
        timestamp = datetime.datetime.now().isoformat()
        self.training_logs.append(f"[{timestamp}] {message}")

    def _move_processed_training_data(self, language: str = "all"):
        """Move training data from train directory to processed directory after fine-tuning"""
        try:
            moved_count = 0

            # Find and move training files from train to processed
            if language == "all":
                patterns = ["speech_*_training.wav", "speech_*_training.txt"]
            else:
                patterns = [f"speech_*_{language}_training.wav", f"speech_*_{language}_training.txt"]

            for pattern in patterns:
                for file_path in self.train_dir.glob(pattern):
                    if file_path.is_file():
                        processed_path = self.processed_dir / file_path.name
                        shutil.move(str(file_path), str(processed_path))
                        moved_count += 1

            if moved_count > 0:
                print(f"Moved {moved_count} training files from train to processed")

        except Exception as e:
            print(f"Warning: Failed to move processed training data: {e}")

    def get_training_status(self) -> dict:
        """Get status of training data availability"""
        try:
            training_files = list(self.data_dir.glob("speech_*_training.txt"))
            languages = set()

            for file in training_files:
                # Extract language from filename pattern: speech_TIMESTAMP_LANGUAGE_training.txt
                filename = file.name
                if '_training.txt' in filename:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        lang = parts[-2]  # language is second to last
                        languages.add(lang)

            return {
                "training_samples": len(training_files),
                "languages": list(languages),
                "available_languages": list(languages),
                "data_directory": str(self.data_dir)
            }

        except Exception as e:
            return {
                "error": str(e),
                "training_samples": 0,
                "languages": [],
                "available_languages": [],
                "data_directory": str(self.data_dir)
            }
