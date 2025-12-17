"""
Speech Training Module
Handles LoRA fine-tuning and training operations
"""
import datetime
import shutil
from pathlib import Path

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


class SpeechTrainingMixin:
    """Mixin class containing training and fine-tuning functionality"""

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

                    # Process audio with Whisper processor
                    inputs = processor(audio_array, sampling_rate=sample_rate, return_tensors="pt")
                    input_features = inputs.input_features.squeeze(0)  # Remove batch dimension

                    # Process text
                    labels = processor(text=batch["text"]).input_ids
                    # Handle different return types from processor
                    if isinstance(labels, list):
                        if len(labels) > 0:
                            # If it's a list of lists, take the first one
                            labels = labels[0] if isinstance(labels[0], list) else labels
                        else:
                            labels = []
                    elif hasattr(labels, 'tolist'):
                        labels = labels.tolist()
                        if isinstance(labels, list) and len(labels) > 0 and isinstance(labels[0], list):
                            labels = labels[0]
                    else:
                        # Fallback: convert to list if possible
                        try:
                            labels = list(labels)
                        except:
                            labels = []

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

                # Save LoRA adapter path for return
                lora_model_path = self.models_dir / "loras" / language

                return {
                    "status": "success",
                    "model_path": str(lora_model_path),
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

            # Move data for this language from processed to waiting
            self._move_from_processed_to_train(language)

            # Check for training data
            training_files = list(self.train_dir.glob(f"speech_*_{language}_training.txt"))
            if not training_files:
                raise Exception(f"No {language} training data found")

            # Create YOLO dataset YAML file
            dataset_yaml_path = self._create_whisper_dataset_yaml(language, self.train_dir / language)
            self._add_training_log(f"Created Whisper dataset YAML: {dataset_yaml_path}")

            # Run real Whisper LoRA training
            self._add_training_log("Starting real Whisper LoRA training...")
            lora_file = await self._run_whisper_lora_training(
                base_model_path=self.models_dir / "base" / "tiny.pt",
                dataset_yaml_path=dataset_yaml_path,
                lora_output_dir=self.models_dir / "loras" / language,
                language=language,
                epochs=5,  # Whisper LoRA typically uses fewer epochs
                batch_size=2
            )
            self._add_training_log(f"Real Whisper LoRA training completed: {lora_file}")

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

    async def _run_whisper_lora_training(self, base_model_path: Path, dataset_yaml_path: Path,
                                        lora_output_dir: Path, language: str, epochs: int,
                                        batch_size: int) -> Path:
        """Run real Whisper LoRA training using HuggingFace transformers"""
        try:
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            import torch
            from torch.utils.data import Dataset, DataLoader
            import json
            import asyncio

            print(f"🚀 Starting REAL Whisper LoRA training for {language}...")
            print(f"  Base model: {base_model_path}")
            print(f"  Dataset: {dataset_yaml_path}")
            print(f"  Epochs: {epochs}, batch size: {batch_size}")

            # Load processor and base model
            print("Loading Whisper processor and base model...")
            processor = WhisperProcessor.from_pretrained("openai/whisper-tiny", language=language, task="transcribe")

            # Load base model - try local first, then download
            try:
                model = WhisperForConditionalGeneration.from_pretrained(str(base_model_path))
                print("✓ Loaded local Whisper base model")
            except:
                print("Local base model not found, downloading from HuggingFace...")
                model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
                print("✓ Downloaded Whisper base model from HuggingFace")

            # Prepare model for LoRA
            model = prepare_model_for_kbit_training(model)

            # Configure LoRA
            lora_config = LoraConfig(
                r=32,  # rank dimension
                lora_alpha=64,  # scaling parameter
                target_modules=["q_proj", "v_proj", "k_proj", "out_proj"],  # attention layers
                lora_dropout=0.1,
                bias="none",
                task_type="SEQ2SEQ_LM"  # Whisper is a sequence-to-sequence model
            )

            # Apply LoRA to model
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()

            # Load training data
            print("Loading training dataset...")
            data_dir = dataset_yaml_path.parent
            training_files = list(data_dir.glob("speech_*_training.txt"))

            if not training_files:
                raise Exception(f"No training data found in {data_dir}")

            # Create dataset from training files
            training_data = []
            for txt_file in training_files:
                try:
                    # Find corresponding audio file
                    audio_file = txt_file.with_suffix('.webm')
                    if not audio_file.exists():
                        audio_file = txt_file.with_suffix('.wav')
                    if not audio_file.exists():
                        continue

                    # Read transcript
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcript = f.read().strip()

                    training_data.append({
                        "audio_path": str(audio_file),
                        "transcript": transcript
                    })

                except Exception as e:
                    print(f"Warning: Failed to load training pair {txt_file}: {e}")
                    continue

            if not training_data:
                raise Exception("No valid training data found")

            print(f"Loaded {len(training_data)} training samples")

            # Custom dataset class
            class WhisperDataset(Dataset):
                def __init__(self, data, processor):
                    self.data = data
                    self.processor = processor

                def __len__(self):
                    return len(self.data)

                def __getitem__(self, idx):
                    item = self.data[idx]

                    # Load and process audio
                    import librosa
                    audio_array, sample_rate = librosa.load(item["audio_path"], sr=16000)

                    # Process audio
                    input_features = self.processor(audio_array, sampling_rate=16000).input_features[0]

                    # Process text
                    labels = self.processor(text=item["transcript"]).input_ids

                    return {
                        "input_features": input_features,
                        "labels": labels
                    }

            # Create dataset and dataloader
            dataset = WhisperDataset(training_data, processor)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            # Training setup
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            print(f"Starting training on device: {device}")

            # Training loop
            model.train()
            for epoch in range(epochs):
                epoch_loss = 0
                for batch_idx, batch in enumerate(dataloader):
                    input_features = batch["input_features"].to(device)
                    labels = batch["labels"].to(device)

                    # Forward pass
                    outputs = model(input_features=input_features, labels=labels)
                    loss = outputs.loss

                    # Backward pass
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    epoch_loss += loss.item()

                    if batch_idx % 5 == 0:
                        print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}")

                avg_loss = epoch_loss / len(dataloader)
                print(f"Epoch {epoch+1}/{epochs} completed, Average Loss: {avg_loss:.4f}")

            # Save the trained LoRA model
            print("Saving trained LoRA model...")
            lora_file = lora_output_dir / f"{language}.pt"

            # Save the PEFT model
            model.save_pretrained(str(lora_output_dir))

            # Also save as PyTorch state dict for compatibility
            torch.save({
                'model_state_dict': model.state_dict(),
                'lora_config': lora_config,
                'language': language,
                'epochs_trained': epochs,
                'training_samples': len(training_data)
            }, lora_file)

            print(f"✅ REAL Whisper LoRA training completed!")
            print(f"  LoRA model saved to: {lora_file}")

            return lora_file

        except Exception as e:
            raise Exception(f"Real Whisper LoRA training failed: {str(e)}")

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

    def _add_training_log(self, message: str):
        """Add a training log message"""
        timestamp = datetime.datetime.now().isoformat()
        self.training_logs.append(f"[{timestamp}] {message}")

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

    def _create_whisper_dataset_yaml(self, language: str, data_dir: Path) -> Path:
        """Create Whisper dataset YAML file for training"""
        try:
            # Create dataset YAML content for Whisper
            yaml_content = f"""# Whisper Dataset Configuration for {language} training
# Auto-generated by SpeechService

# Dataset paths
path: {data_dir}  # dataset root dir
train: {data_dir}  # train data (relative to 'path')

# Language and task
language: {language}
task: transcribe

# Dataset format
format: json  # Whisper expects JSON format with audio paths and transcripts
"""

            # Save YAML file
            yaml_path = data_dir / f"{language}_whisper_dataset.yaml"
            with open(yaml_path, 'w') as f:
                f.write(yaml_content)

            print(f"Created Whisper dataset YAML: {yaml_path}")
            return yaml_path

        except Exception as e:
            raise Exception(f"Failed to create Whisper dataset YAML: {str(e)}")

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
