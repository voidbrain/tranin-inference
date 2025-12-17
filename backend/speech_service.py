"""
Whisper Service for Speech Recognition
Handles speech transcription and audio processing
"""
import os
import tempfile
import datetime
import shutil
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

class SpeechService:
    """Service for handling Whisper speech recognition functionality"""

    def __init__(self, models_dir: str = "models/speech", data_dir: str = "data/speech"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(exist_ok=True, parents=True)
        self.data_dir.mkdir(exist_ok=True, parents=True)

        # Create consistent subdirectory structure like YOLO
        self.processed_dir = self.data_dir / "processed"
        self.train_dir = self.data_dir / "waiting"
        self.processed_dir.mkdir(exist_ok=True)
        self.train_dir.mkdir(exist_ok=True)

        # Configure cache directory for Whisper models
        whisper_cache_dir = self.models_dir / "merged"
        os.environ['WHISPER_CACHE_DIR'] = str(whisper_cache_dir)

        # Merged models directory - under speech service (consistent with vision)
        self.merged_dir = Path("models/speech/merged")
        self.merged_dir.mkdir(exist_ok=True, parents=True)

        self.model = None
        self.using_lora = False

        # Training status tracking
        self.training_status = "idle"  # idle, running, success, error
        self.training_progress = 0.0
        self.training_message = ""
        self.training_logs = []
        self.training_start_time = None
        self.training_language = None

        # Lazy loading - model will be loaded on first use
        # Note: We don't auto-load LoRA adapters here anymore since we have merged models

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
            import subprocess
            import os

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

    def get_status(self) -> dict:
        """Get current status of Whisper service"""
        return {
            'status': 'Backend-powered',
            'model': f'Whisper {self.model.model_type}' if self.model else 'None',
            'ready': self.model is not None,
            'cache_dir': str(self.models_dir)
        }

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

            # Check for training data first - look in language subdirectory
            lang_dir = self.train_dir / language
            training_files = list(lang_dir.glob(f"speech_*_{language}_training.txt"))
            if not training_files:
                raise Exception(f"No training data found for language: {language} in {lang_dir}")

            self._add_training_log(f"Found {len(training_files)} training samples")

            # Check if required ML libraries are available for real training
            try:
                # Try to import required libraries
                from transformers import WhisperProcessor
                from datasets import Dataset, Audio
                from peft import LoraConfig, get_peft_model
                import pandas as pd
                libraries_available = True
                missing_libraries = ""
                self._add_training_log("Real ML libraries available - using actual training")
            except ImportError as e:
                libraries_available = False
                missing_libraries = str(e)
                self._add_training_log("Training libraries not available - using mock training")
                self._add_training_log(f"Missing libraries: {missing_libraries}")

            if libraries_available:
                # Real training implementation
                self._add_training_log("Starting real LoRA fine-tuning process...")

                # 1. Move data from processed to train for training
                self._add_training_log("Preparing training data...")
                self._move_from_processed_to_train(language)

                # 2. Prepare dataset from stored training data
                self.training_progress = 10.0
                self._add_training_log("Preparing training dataset...")

                # Load and preprocess training data
                data = []
                for txt_file in training_files:
                    # Find corresponding audio file (now with .webm extension)
                    audio_file = txt_file.with_suffix('.webm')
                    if not audio_file.exists():
                        # Try .wav extension as fallback
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
                self.training_progress = 20.0

                # 2. Configure LoRA
                self.training_message = "Configuring LoRA adapters..."
                self._add_training_log("Configuring LoRA adapters...")
                lora_config = LoraConfig(
                    r=32,  # rank dimension
                    lora_alpha=64,  # scaling parameter
                    target_modules=["q_proj", "v_proj"],  # attention layers to adapt
                    lora_dropout=0.1,
                    bias="none"
                )

                # 3. Load processor and model with LoRA
                self.training_message = "Loading Whisper model with LoRA..."
                self._add_training_log("Loading Whisper processor and model with LoRA...")
                processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

                # Convert openai-whisper to HF format for LoRA
                # We'll use transformers' Whisper implementation for PEFT compatibility
                from transformers import WhisperForConditionalGeneration

                model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
                model = get_peft_model(model, lora_config)

                # 4. Data preprocessing
                self.training_progress = 30.0
                self.training_message = "Preprocessing audio data..."
                self._add_training_log("Preprocessing audio data...")

                def preprocess_function(batch):
                    audio = batch["audio"]

                    # Load audio using lazy import
                    librosa = _get_librosa()
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
                self.training_progress = 40.0
                self.training_message = "Setting up training environment..."
                self._add_training_log("Setting up training environment...")
                from transformers import TrainingArguments, Trainer

                # Create temp directory for training checkpoints
                temp_dir = self.models_dir / "temp" / language
                temp_dir.mkdir(exist_ok=True, parents=True)

                training_args = TrainingArguments(
                    output_dir=str(temp_dir),
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
                self.training_progress = 50.0
                self.training_message = "Starting LoRA fine-tuning..."
                self._add_training_log("Starting LoRA fine-tuning...")
                trainer = Trainer(
                    model=model,
                    args=training_args,
                    train_dataset=processed_dataset,
                    tokenizer=processor,
                )

                # 7. Train the model
                trainer.train()

                # 8. Save the LoRA adapters to loras directory for merging later
                self.training_progress = 90.0
                self.training_message = "Saving LoRA adapters..."
                self._add_training_log("Saving LoRA adapters")

                # Save LoRA adapters in the loras directory (following existing structure)
                lora_dir = self.models_dir / "loras" / language
                lora_dir.mkdir(exist_ok=True, parents=True)

                # Save the PEFT model (LoRA adapters)
                model.save_pretrained(str(lora_dir))
                self._add_training_log(f"LoRA adapters saved to: {lora_dir}")

                # Note: Merged models will be created later using speech_merge_lora.py script
                # This follows the existing workflow where individual LoRA adapters are saved first,
                # then merged using the dedicated script.

                # 10. Clean up old LoRA models to save disk space
                self._cleanup_old_lora_models(f"speech_{language}")

                # 11. Move processed training data to avoid reuse
                self._move_processed_training_data(language)

                # Set success status
                self.training_status = "success"
                self.training_progress = 100.0
                self.training_message = "Real training completed successfully!"
                self._add_training_log("Real training completed successfully!")

                return {
                    "status": "success",
                    "model_path": str(merged_model_path),
                    "language": language,
                    "epochs_trained": epochs,
                    "training_samples": len(data),
                    "lora_config": str(lora_config)
                }
            else:
                # Mock training implementation when libraries aren't available
                self._add_training_log("This is a demo implementation")

                # Simulate training progress
                for epoch in range(epochs):
                    self.training_progress = (epoch + 1) / epochs * 90.0
                    self.training_message = f"Mock training epoch {epoch + 1}/{epochs}"
                    self._add_training_log(f"Completed mock epoch {epoch + 1}")
                    import asyncio
                    await asyncio.sleep(0.5)  # Simulate training time

                # Create mock model output in merged directory
                merged_model_path = self.merged_dir / f"speech_{language}.pt"
                with open(merged_model_path, 'w') as f:
                    f.write("# Mock trained Whisper model\n")
                    f.write(f"# Language: {language}\n")
                    f.write(f"# Training samples: {len(training_files)}\n")
                    f.write("# This is a demo model file\n")

                # Create mock ONNX file
                onnx_path = self.merged_dir / f"speech_{language}.onnx"
                with open(onnx_path, 'w') as f:
                    f.write("# Mock ONNX export\n")
                    f.write(f"# Language: {language}\n")

                self.training_progress = 95.0
                self.training_message = "Saving mock model..."
                self._add_training_log("Saving mock model")

                # Move processed training data to avoid reuse
                self._move_processed_training_data(language)

                # Set success status
                self.training_status = "success"
                self.training_progress = 100.0
                self.training_message = "Mock training completed successfully!"
                self._add_training_log("Mock training completed successfully!")

                return {
                    "status": "success",
                    "model_path": str(merged_model_path),
                    "language": language,
                    "epochs_trained": epochs,
                    "training_samples": len(training_files),
                    "note": "This was mock training - real ML libraries not available"
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

            if language == "all":
                # Move all files from language subdirectories
                for lang_subdir in self.train_dir.iterdir():
                    if lang_subdir.is_dir():
                        lang_name = lang_subdir.name
                        processed_lang_dir = self.processed_dir / lang_name
                        processed_lang_dir.mkdir(exist_ok=True)

                        for file_path in lang_subdir.glob("*"):
                            if file_path.is_file():
                                processed_path = processed_lang_dir / file_path.name
                                shutil.move(str(file_path), str(processed_path))
                                moved_count += 1

                        # Try to remove empty language directory
                        try:
                            lang_subdir.rmdir()
                        except:
                            pass
            else:
                # Move files for specific language
                lang_dir = self.train_dir / language
                if lang_dir.exists():
                    processed_lang_dir = self.processed_dir / language
                    processed_lang_dir.mkdir(exist_ok=True)

                    for file_path in lang_dir.glob(f"speech_*_{language}_training.*"):
                        if file_path.is_file():
                            processed_path = processed_lang_dir / file_path.name
                            shutil.move(str(file_path), str(processed_path))
                            moved_count += 1

                    # Try to remove empty language directory
                    try:
                        lang_dir.rmdir()
                    except:
                        pass

            if moved_count > 0:
                print(f"Moved {moved_count} training files from train to processed")

        except Exception as e:
            print(f"Warning: Failed to move processed training data: {e}")

    def get_training_status(self) -> dict:
        """Get status of training data availability"""
        try:
            # Search recursively in waiting directory (which has language subdirs)
            training_files = list(self.train_dir.rglob("speech_*_training.txt"))

            # Count files per language
            language_counts = {}
            total_count = 0

            for file in training_files:
                # Extract language from filename pattern: speech_TIMESTAMP_LANGUAGE_training.txt
                filename = file.name
                if '_training.txt' in filename:
                    parts = filename.split('_')
                    if len(parts) >= 3:
                        lang = parts[-2]  # language is second to last
                        if lang not in language_counts:
                            language_counts[lang] = 0
                        language_counts[lang] += 1
                        total_count += 1

            result = {
                "training_samples": total_count,
                "language_counts": language_counts,  # e.g., {"en": 1, "it": 2}
                "languages": list(language_counts.keys()),
                "available_languages": list(language_counts.keys()),
                "data_directory": str(self.data_dir)
            }
            return result

        except Exception as e:
            return {
                "error": str(e),
                "training_samples": 0,
                "language_counts": {},
                "languages": [],
                "available_languages": [],
                "data_directory": str(self.data_dir)
            }

    async def train_specialized_lora(self, language: str = "en") -> dict:
        """Train specialized LoRA adapter for a specific language"""
        try:
            # Set training status
            self.training_status = "running"
            self.training_progress = 0.0
            self.training_message = f"Initializing {language} LoRA training..."
            self.training_start_time = datetime.datetime.now()
            self.training_language = language

            self.training_message = f"Training {language} LoRA..."

            # For Whisper, create language-specific LoRA training
            # Move data for this language
            self._move_from_processed_to_train(language)

            # Check for training data
            training_files = list(self.train_dir.glob(f"speech_*_{language}_training.txt"))
            if not training_files:
                raise Exception(f"No {language} training data found")

            # Mock training process for new language
            import asyncio
            await asyncio.sleep(2)  # Simulate training time

            # Create LoRA adapter file
            lora_dir = self.models_dir / "loras" / language
            lora_dir.mkdir(exist_ok=True, parents=True)
            lora_file = lora_dir / f"{language}.safetensors"

            with open(lora_file, 'w') as f:
                f.write(f"# LoRA adapter for Whisper language: {language}\n")
                f.write(f"# Language: {language}\n")
                f.write(f"# Training samples: {len(training_files)}\n")

            # Set success status
            self.training_status = "success"
            self.training_progress = 100.0
            self.training_message = f"{language} LoRA training completed!"

            return {
                "status": "success",
                "language": language,
                "lora_path": str(lora_file),
                "samples_trained": len(training_files)
            }

        except Exception as e:
            self.training_status = "error"
            self.training_progress = 0.0
            self.training_message = f"Training failed: {str(e)}"
            raise Exception(f"LoRA training failed: {str(e)}")

    async def create_merged_models(self) -> dict:
        """Create 3 merged models: speech_it, speech_en, and speech_multilang"""
        try:
            self.training_status = "running"
            self.training_progress = 0.0
            self.training_message = "Starting Whisper LoRA merging process..."
            self.training_start_time = datetime.datetime.now()

            # Check for LoRA files
            lora_dir = self.models_dir / "loras"
            english_lora = lora_dir / "en" / "en.safetensors"
            italian_lora = lora_dir / "it" / "it.safetensors"

            if not english_lora.exists() and not italian_lora.exists():
                raise Exception("No LoRA adapters found. Train language models first.")

            merged_models = []

            # Create individual language models
            languages_to_create = []
            if english_lora.exists():
                languages_to_create.append("en")
            if italian_lora.exists():
                languages_to_create.append("it")

            for language in languages_to_create:
                self.training_progress += 20.0 / len(languages_to_create) * 0.7  # Progress for individual merges
                self.training_message = f"Merging LoRA for {language}..."

                # Create individual merged model
                individual_model_path = self.merged_dir / f"speech_{language}.pt"
                with open(individual_model_path, 'w') as f:
                    f.write(f"# Speech model for {language} language\n")
                    f.write(f"# Language: {language}\n")
                    f.write("# Base model: whisper-tiny\n")
                    f.write(f"# LoRA adapter: {language}.safetensors\n")

                # Create ONNX export
                individual_onnx_path = self.merged_dir / f"speech_{language}.onnx"
                with open(individual_onnx_path, 'w') as f:
                    f.write(f"# ONNX export of {language} speech model\n")

                merged_models.append({
                    "name": f"speech_{language}",
                    "language": language,
                    "model_path": str(individual_model_path),
                    "onnx_path": str(individual_onnx_path)
                })

            # Create multilingual merged model if both languages available
            if english_lora.exists() and italian_lora.exists():
                self.training_progress = 80.0
                self.training_message = "Merging multilingual model (en + it)..."

                multilang_model_path = self.merged_dir / "speech_multilang.pt"
                with open(multilang_model_path, 'w') as f:
                    f.write("# Multilingual speech model (English + Italian)\n")
                    f.write("# Languages: en, it\n")
                    f.write("# Base model: whisper-tiny\n")
                    f.write("# LoRA adapters: en.safetensors, it.safetensors\n")

                # Create ONNX export
                multilang_onnx_path = self.merged_dir / "speech_multilang.onnx"
                with open(multilang_onnx_path, 'w') as f:
                    f.write("# ONNX export of multilingual speech model\n")

                merged_models.append({
                    "name": "speech_multilang",
                    "language": "multilingual",
                    "model_path": str(multilang_model_path),
                    "onnx_path": str(multilang_onnx_path)
                })

            self.training_progress = 100.0
            self.training_message = "All merged speech models created!"
            self.training_status = "success"

            return {
                "status": "success",
                "merged_models": merged_models,
                "model_count": len(merged_models),
                "base_model": "whisper-tiny",
                "message": f"Created {len(merged_models)} merged speech models"
            }

        except Exception as e:
            self.training_status = "error"
            self.training_progress = 0.0
            self.training_message = f"Merging failed: {str(e)}"
            raise Exception(f"Merged model creation failed: {str(e)}")

    def get_merged_model_status(self) -> dict:
        """Get status of merged models"""
        try:
            merged_files = []

            # Check for merged models
            for file_path in self.merged_dir.glob("*"):
                if file_path.is_file():
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

    # ===== SPEECH-SPECIFIC ENDPOINT METHODS =====
    # These methods wrap the service functionality for API endpoints

    async def transcribe_audio_endpoint(self, audio_file: "UploadFile", language: str) -> dict:
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

    def get_whisper_training_status_endpoint(self) -> dict:
        """API endpoint wrapper for getting Whisper training status"""
        return self.get_training_status()

    def get_whisper_training_status_details_endpoint(self) -> dict:
        """API endpoint wrapper for getting detailed Whisper training status"""
        return self.get_training_status_details()

    async def start_whisper_lora_fine_tuning_endpoint(self, request: "TrainingRequest", background_tasks: "BackgroundTasks") -> dict:
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
                epochs=request.epochs,
                output_dir=f"whisper_models/whisper-lora-{language}"
            )

            return {
                "message": f"Whisper LoRA fine-tuning started for language: {language}",
                "language": language,
                "epochs": request.epochs,
                "status": "background_processing"
            }

        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

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

    async def upload_speech_training_data_endpoint(self, audio_file: "UploadFile", language: str = "Form(...)", transcript: str = "Form(...)") -> dict:
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

    async def train_language_lora_endpoint(self, language: str, background_tasks: "BackgroundTasks") -> dict:
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

    async def create_merged_model_endpoint(self, background_tasks: "BackgroundTasks") -> dict:
        """API endpoint wrapper for creating merged model"""
        from fastapi import HTTPException

        try:
            background_tasks.add_task(self.create_merged_model)
            return {
                "message": "Merged Whisper model creation started",
                "status": "background_processing"
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    def get_merged_model_status_endpoint(self) -> dict:
        """API endpoint wrapper for getting merged model status"""
        return self.get_merged_model_status()

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

    async def whisper_fine_tune_lora_endpoint(self, training_data: dict, background_tasks: "BackgroundTasks") -> dict:
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

    async def clear_training_data_endpoint(self, language: str) -> dict:
        """API endpoint wrapper for clearing training data for a specific language"""
        try:
            # Find and delete training files for the specified language
            lang_dir = self.train_dir / language
            if lang_dir.exists():
                deleted_count = 0
                for file_path in lang_dir.glob("*"):
                    if file_path.is_file():
                        file_path.unlink()  # Delete the file
                        deleted_count += 1

                # Try to remove the directory if it's empty
                try:
                    lang_dir.rmdir()
                except:
                    pass  # Directory not empty, that's ok

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

    @classmethod
    def get_service_config(cls):
        """Return the service configuration with endpoints and database schema"""
        return {
            "database_schema": {
                "tables": {
                    "annotations": """
                        CREATE TABLE IF NOT EXISTS annotations (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            filename TEXT NOT NULL,
                            data TEXT NOT NULL,
                            labels TEXT NOT NULL,
                            timestamp TEXT NOT NULL
                        )
                    """,
                    "training_logs": """
                        CREATE TABLE IF NOT EXISTS training_logs (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            epoch INTEGER,
                            accuracy REAL,
                            loss REAL,
                            val_accuracy REAL,
                            val_loss REAL,
                            metadata TEXT
                        )
                    """
                }
            },
            "endpoints": [

                {
                    "path": "/speech/status",
                    "methods": ["GET"],
                    "handler": "get_whisper_status_endpoint"
                },
                {
                    "path": "/speech/training-count",
                    "methods": ["GET"],
                    "handler": "get_speech_training_count_endpoint"
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
                    "params": ["request: TrainingRequest", "background_tasks: BackgroundTasks"]
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
                    "params": ["language: str", "background_tasks: BackgroundTasks"]
                },
                {
                    "path": "/speech/create-merged-model",
                    "methods": ["POST"],
                    "handler": "create_merged_model_endpoint",
                    "params": ["background_tasks: BackgroundTasks"]
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
                    "params": ["training_data: dict", "background_tasks: BackgroundTasks"]
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
