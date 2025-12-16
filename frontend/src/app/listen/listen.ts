import { Component, OnInit, OnDestroy, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { HttpClientModule } from '@angular/common/http';

@Component({
  selector: 'app-listen',
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './listen.html',
  styleUrl: './listen.scss',
})
export class Listen implements OnInit, OnDestroy {
  // Use localhost for development, backend service for production/Docker
  private backendUrl = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : 'http://backend:8000';

  // Language and recording state
  private _selectedLanguage = 'multi'; // Private backing for getter/setter

  get selectedLanguage(): string {
    return this._selectedLanguage;
  }

  set selectedLanguage(value: string) {
    if (this._selectedLanguage === value) return; // No change

    this._selectedLanguage = value;
    this.loadLanguageModel(value);
  }
  isRecording = false;
  transcript = '';
  status = 'Ready';
  whisperStatus = 'Backend-powered';
  lastMessage = '';
  lastMessageType: 'success' | 'error' | '' = '';

  // Add language mapping for UI display
  getLanguageDisplayName(langCode: string): string {
    const names: { [key: string]: string } = {
      'en': 'English',
      'it': 'Italian',
      'multi': 'Multilingual'
    };
    return names[langCode] || langCode;
  }

  // Audio state
  audioBlob: Blob | null = null;
  audioUrl: string | null = null;
  audioDuration = '0';
  audioSize = '0';
  isStartingRecording = false;

  // File upload state
  selectedAudioFile: File | null = null;
  audioSource: 'recorded' | 'uploaded' | null = null;

  // Transcription result details
  transcriptionLanguage = '';
  transcriptionConfidence = 0;
  transcriptionDuration = 0;

  // Processing states
  isTranscribing = false;
  isUploading = false;
  isTraining = false;

  // Training status polling
  trainingStatus = 'idle'; // idle, running, success, error
  trainingProgress = 0;
  trainingMessage = '';
  trainingLogs: string[] = [];
  trainingIntervalId: any = null;
  trainingDataCount = 0;
  languageCounts: { [key: string]: number } = {}; // e.g., {"en": 1, "it": 2}

  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private recordingStartTime: number = 0;
  private stream: MediaStream | null = null;

  constructor(private http: HttpClient, private cdr: ChangeDetectorRef) {}

  ngOnInit() {
    this.updateTrainingDataCount();
    // Load initial language model (multilingual by default)
    this.loadLanguageModel(this.selectedLanguage);
  }

  ngOnDestroy() {
    // Clean up media resources when component is destroyed
    this.cleanupRecording();

    // Stop training status polling
    this.stopTrainingStatusPolling();

    // Clean up audio URL
    if (this.audioUrl) {
      URL.revokeObjectURL(this.audioUrl);
      this.audioUrl = null;
    }
  }

  async startRecording() {
    if (this.isStartingRecording) return;

    this.isStartingRecording = true;

    try {
      // Clean up any existing recording
      await this.cleanupRecording();

      this.status = 'Requesting microphone access...';

      // Add timeout to prevent hanging if user doesn't respond to permission prompt
      const timeoutPromise = new Promise<never>((_, reject) =>
        setTimeout(() => reject(new Error('Recording permission timeout')), 15000)
      );

      this.stream = await Promise.race([
        navigator.mediaDevices.getUserMedia({
          audio: {
            echoCancellation: true,
            noiseSuppression: true,
            autoGainControl: true
          }
        }),
        timeoutPromise
      ]);

      this.mediaRecorder = new MediaRecorder(this.stream, {
        mimeType: 'audio/webm;codecs=opus'
      });

      this.audioChunks = [];
      this.recordingStartTime = Date.now();

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        this.updateAudioInfo();
        this.isRecording = false;
        this.isStartingRecording = false;

        // Use setTimeout to avoid ExpressionChangedAfterItHasBeenCheckedError
        setTimeout(() => {
          this.status = 'Audio recorded successfully';
          this.showMessage('Audio recorded successfully!', 'success');
        }, 0);

        // Note: We don't stop the stream here to allow immediate restart
        // Stream cleanup will happen on component destroy or next recording
      };

      this.mediaRecorder.onerror = (event) => {
        console.error('MediaRecorder error:', event);
        this.status = 'Recording error occurred';
        this.isRecording = false;
        this.isStartingRecording = false;
        this.showMessage('Recording error occurred', 'error');
      };

      // MediaRecorder should now be recording
      this.mediaRecorder.start(100); // Collect data every 100ms

      // At this point MediaRecorder is successfully started
      // Use ChangeDetectorRef to trigger Angular change detection immediately
      this.isRecording = true;
      this.isStartingRecording = false; // Recording has actually started
      this.status = 'Recording... (Click Stop when done)';
      this.clearMessages();
      this.cdr.detectChanges();

    } catch (error) {
      console.error('Recording error:', error);
      this.status = 'Recording failed: Permission denied or no microphone access';
      this.showMessage('Failed to start recording - check microphone permissions', 'error');
      this.isStartingRecording = false;
    }
  }

  stopRecording() {
    if (!this.isRecording) return;

    // Immediately disable recording flag to prevent multiple clicks
    this.isRecording = false;
    this.status = 'Stopping recording...';

    if (this.mediaRecorder) {
      this.mediaRecorder.stop();

      // Set a timeout to force stop if onstop callback doesn't fire
      setTimeout(() => {
        if (this.audioBlob === null && this.stream) {
          console.warn('MediaRecorder onstop callback did not fire, forcing cleanup');
          this.forceStopRecording();
        }
      }, 500);

    } else {
      // If mediaRecorder is not available but recording is supposedly active,
      // force cleanup
      this.forceStopRecording();
    }
  }

  private updateAudioInfo() {
    if (this.audioChunks.length > 0) {
      const duration = Math.round((Date.now() - this.recordingStartTime) / 1000);
      const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
      const size = (audioBlob.size / 1024 / 1024).toFixed(2); // MB

      // Clean up previous audio URL
      if (this.audioUrl) {
        URL.revokeObjectURL(this.audioUrl);
      }

      this.audioBlob = audioBlob;
      this.audioUrl = URL.createObjectURL(audioBlob);
      this.audioDuration = duration.toString();
      this.audioSize = size;
      this.audioSource = 'recorded';

      // Trigger change detection to update the UI
      this.cdr.detectChanges();
    }
  }

  private forceStopRecording() {
    console.warn('Forcing recording stop');
    this.isRecording = false;
    this.isStartingRecording = false;
    this.status = 'Recording stopped manually';

    // Stop the media stream to end recording
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }

    // Save any accumulated chunks as audio blob
    if (this.audioChunks.length > 0) {
      this.updateAudioInfo();
      this.status = 'Audio recorded successfully (forced stop)';
      this.showMessage('Audio recorded successfully!', 'success');
    } else {
      this.status = 'Recording stopped (no audio data)';
    }

    this.cleanupRecording();
  }

  async resetRecording() {
    // Stop recording if in progress
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
    }

    // Clean up media resources
    await this.cleanupRecording();

    // Clean up audio URL
    if (this.audioUrl) {
      URL.revokeObjectURL(this.audioUrl);
      this.audioUrl = null;
    }

    // Reset state
    this.audioBlob = null;
    this.audioUrl = null;
    this.audioDuration = '0';
    this.audioSize = '0';
    this.audioSource = null;
    this.transcript = '';
    this.status = 'Recording reset';
    this.clearMessages();
  }

  private async cleanupRecording() {
    // Stop all media tracks
    if (this.stream) {
      this.stream.getTracks().forEach(track => {
        track.stop();
      });
      this.stream = null;
    }

    // Clean up mediaRecorder
    if (this.mediaRecorder) {
      if (this.mediaRecorder.state === 'recording') {
        this.mediaRecorder.stop();
      }
      this.mediaRecorder = null;
    }

    this.audioChunks = [];
    this.isRecording = false;
  }

  private async loadLanguageModel(language: string) {
    try {
      // Map frontend language codes to backend model types
      const modelMap: { [key: string]: string } = {
        'en': 'en',      // English model
        'it': 'it',      // Italian model
        'multi': 'multilang'  // Multilingual model
      };

      const backendModelType = modelMap[language] || 'multilang'; // Default to multilingual

      this.status = `Switching to ${this.getLanguageDisplayName(language)} model...`;
      this.showMessage(`Loading ${this.getLanguageDisplayName(language)} Whisper model...`, 'success');

      // Call backend to load the appropriate model
      const response = await this.http.post(`${this.backendUrl}/speech/load-language-model`, {
        language: backendModelType
      }).toPromise();

      this.whisperStatus = `${this.getLanguageDisplayName(language)} Model Active`;
      this.status = `Ready - Using ${this.getLanguageDisplayName(language)} model`;

    } catch (error: any) {
      console.error(`Failed to load ${language} model:`, error);
      this.status = `Failed to load ${this.getLanguageDisplayName(language)} model`;
      this.showMessage(`Model loading failed: ${error.message}`, 'error');
    }
  }

  async transcribeAudio() {
    if (!this.audioBlob) return;

    console.log('Starting transcription...');
    this.isTranscribing = true;
    this.status = 'Transcribing audio...';
    this.cdr.detectChanges(); // Force UI update

    try {
      const formData = new FormData();
      const filename = `audio_${Date.now()}.webm`; // Use correct extension
      formData.append('audio_file', this.audioBlob, filename);
      formData.append('language', this.selectedLanguage); // Pass selected language

      const url = `${this.backendUrl}/speech/transcribe-audio?language=${encodeURIComponent(this.selectedLanguage)}`;
      console.log('FormData created, sending request to:', url);

      // Use fetch instead of Angular HTTP client for better debugging
      const response = await fetch(url, {
        method: 'POST',
        body: formData
      });

      console.log('Fetch response status:', response.status);

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const responseData = await response.json();
      console.log('Parsed response data:', responseData);

      // Store transcription details
      this.transcript = responseData?.transcription || 'Transcription failed';
      this.transcriptionLanguage = responseData?.language || this.selectedLanguage;
      this.transcriptionConfidence = responseData?.confidence || 0;
      this.transcriptionDuration = responseData?.duration || 0;

      const languageName = this.getLanguageDisplayName(this.transcriptionLanguage);
      this.status = `Transcription complete (${languageName})`;
      this.showMessage('Transcription completed successfully!', 'success');

      console.log('Transcription completed successfully');

    } catch (error: any) {
      console.error('Transcription error:', error);
      this.transcript = 'Error: Transcription failed';
      this.transcriptionLanguage = '';
      this.transcriptionConfidence = 0;
      this.transcriptionDuration = 0;
      this.status = 'Transcription failed';
      this.showMessage('Transcription failed: ' + error.message, 'error');
    } finally {
      this.isTranscribing = false;
      console.log('Setting isTranscribing to false');
      this.cdr.detectChanges(); // Force UI update
    }
  }

  async uploadForTraining() {
    if (!this.audioBlob) return;

    this.isUploading = true;
    this.status = 'Uploading audio for training...';
    this.cdr.detectChanges(); // Force UI update

    try {
      const formData = new FormData();
      const filename = `speech_${Date.now()}_${this.selectedLanguage}.webm`;
      formData.append('audio_file', this.audioBlob, filename);

      const url = `${this.backendUrl}/speech/upload-speech-training-data?language=${encodeURIComponent(this.selectedLanguage)}&transcript=${encodeURIComponent(this.transcript)}`;

      console.log('Uploading training data to:', url);
      const response = await this.http.post(url, formData).toPromise();

      console.log('Upload response:', response);
      this.status = 'Audio uploaded for speech training';
      this.showMessage('Audio uploaded for training successfully!', 'success');

      // Update training count after successful upload
      console.log('DEBUG: About to call updateTrainingDataCount...');
      try {
        await this.updateTrainingDataCount();
        console.log('DEBUG: updateTrainingDataCount completed successfully');
        console.log('DEBUG: Current trainingDataCount value:', this.trainingDataCount);
      } catch (error) {
        console.error('DEBUG: updateTrainingDataCount failed:', error);
      }

    } catch (error: any) {
      console.error('Upload error:', error);
      this.status = 'Upload failed';
      this.showMessage('Upload failed: ' + error.message, 'error');
    } finally {
      this.isUploading = false;
      this.cdr.detectChanges(); // Force UI update
    }
  }

  async startLoraTraining() {
    if (this.trainingDataCount === 0) return;

    this.isTraining = true;
    this.status = 'Starting LoRA fine-tuning...';

    try {
      const trainingRequest = {
        epochs: 5, // Default epochs for LoRA training
        use_lora: true // Always use LoRA for speech training
      };

      const response = await this.http.post(`${this.backendUrl}/speech/whisper-fine-tune-lora`, trainingRequest).toPromise();

      this.status = 'LoRA training started in background';
      this.showMessage('Whisper LoRA fine-tuning has started! Check logs for progress.', 'success');

      // Start polling for training status
      this.startTrainingStatusPolling();

    } catch (error: any) {
      console.error('Training start error:', error);
      this.status = 'Training start failed';
      this.showMessage('Failed to start training: ' + error.message, 'error');
    } finally {
      this.isTraining = false;
    }
  }

  private startTrainingStatusPolling() {
    // Poll every 2 seconds for training status updates
    this.trainingIntervalId = setInterval(async () => {
      try {
        const response = await this.http.get(`${this.backendUrl}/speech/whisper-training-status-details`).toPromise();
        const status: any = response;

        this.trainingStatus = status.status;
        this.trainingProgress = status.progress;
        this.trainingMessage = status.message;
        this.trainingLogs = status.logs;

        // Stop polling when training completes (success or error)
        if (status.status === 'success' || status.status === 'error') {
          this.stopTrainingStatusPolling();

          if (status.status === 'success') {
            this.showMessage('Whisper LoRA training completed successfully!', 'success');
          } else {
            this.showMessage(`Whisper LoRA training failed: ${status.message}`, 'error');
          }

          // Update training data count after completion
          this.updateTrainingDataCount();
        }
      } catch (error) {
        console.error('Failed to poll training status:', error);
      }
    }, 2000);
  }

  private stopTrainingStatusPolling() {
    if (this.trainingIntervalId) {
      clearInterval(this.trainingIntervalId);
      this.trainingIntervalId = null;
    }
  }

  async resetTrainingData(language: string = this.selectedLanguage) {
    const count = this.languageCounts[language] || 0;
    if (count === 0) return;

    try {
      // Clear training data for the specified language
      const response = await this.http.delete<{message: string, language: string, files_deleted: number}>(
        `${this.backendUrl}/speech/clear-training-data/${language}`
      ).toPromise();

      console.log('Clear training data response:', response);

      // Update the training data count after successful deletion
      await this.updateTrainingDataCount();

      const deletedCount = response?.files_deleted || 0;
      const langName = this.getLanguageDisplayName(language);
      this.showMessage(`Cleared ${deletedCount} training files for ${langName}`, 'success');

    } catch (error: any) {
      console.error('Clear training data error:', error);
      this.showMessage('Failed to clear training data: ' + error.message, 'error');
    }
  }

  async startEnglishTraining() {
    if ((this.languageCounts['en'] || 0) === 0) return;

    this.isTraining = true;
    this.status = 'Starting English LoRA fine-tuning...';
    this.cdr.detectChanges(); // Force UI update

    try {
      const trainingRequest = {
        language: 'en',
        epochs: 5, // Default epochs for LoRA training
        use_lora: true // Always use LoRA for speech training
      };

      const response = await this.http.post(`${this.backendUrl}/speech/whisper-fine-tune-lora`, trainingRequest).toPromise();

      this.status = 'English LoRA training started in background';
      this.showMessage('English Whisper LoRA fine-tuning has started! Check logs for progress.', 'success');

      // Start polling for training status
      this.startTrainingStatusPolling();

    } catch (error: any) {
      console.error('English training start error:', error);
      this.status = 'English training start failed';
      this.showMessage('Failed to start English training: ' + error.message, 'error');
      this.isTraining = false; // Reset training state on error
      this.cdr.detectChanges(); // Force UI update
    } finally {
      // Only reset if not already reset in catch block
      if (this.isTraining) {
        this.isTraining = false;
        this.cdr.detectChanges(); // Force UI update
      }
    }
  }

  async startItalianTraining() {
    if ((this.languageCounts['it'] || 0) === 0) return;

    this.isTraining = true;
    this.status = 'Starting Italian LoRA fine-tuning...';
    this.cdr.detectChanges(); // Force UI update

    try {
      const trainingRequest = {
        language: 'it',
        epochs: 5, // Default epochs for LoRA training
        use_lora: true // Always use LoRA for speech training
      };

      const response = await this.http.post(`${this.backendUrl}/speech/whisper-fine-tune-lora`, trainingRequest).toPromise();

      this.status = 'Italian LoRA training started in background';
      this.showMessage('Italian Whisper LoRA fine-tuning has started! Check logs for progress.', 'success');

      // Start polling for training status
      this.startTrainingStatusPolling();

    } catch (error: any) {
      console.error('Italian training start error:', error);
      this.status = 'Italian training start failed';
      this.showMessage('Failed to start Italian training: ' + error.message, 'error');
      this.isTraining = false; // Reset training state on error
      this.cdr.detectChanges(); // Force UI update
    } finally {
      // Only reset if not already reset in catch block
      if (this.isTraining) {
        this.isTraining = false;
        this.cdr.detectChanges(); // Force UI update
      }
    }
  }

  private async updateTrainingDataCount() {
    try {
      console.log('DEBUG: Fetching training count from backend...');
      const response = await this.http.get<{count: number, language_counts: {[key: string]: number}, message: string}>(`${this.backendUrl}/speech/training-count`).toPromise();
      console.log('DEBUG: Backend response:', response);
      const newCount = response?.count || 0;
      const languageCounts = response?.language_counts || {};

      console.log('DEBUG: Setting trainingDataCount to:', newCount);
      console.log('DEBUG: Setting languageCounts to:', languageCounts);

      // Update both properties
      this.trainingDataCount = newCount;
      this.languageCounts = languageCounts;

      // Force Angular change detection
      this.cdr.detectChanges();

      console.log('DEBUG: trainingDataCount is now:', this.trainingDataCount);
      console.log('DEBUG: languageCounts is now:', this.languageCounts);

    } catch (error) {
      console.error('DEBUG: Failed to update training count:', error);
      this.trainingDataCount = 0;
      this.languageCounts = {};
      this.cdr.detectChanges();
    }
  }

  private showMessage(message: string, type: 'success' | 'error') {
    this.lastMessage = message;
    this.lastMessageType = type;
    setTimeout(() => this.clearMessages(), 5000);
  }

  private clearMessages() {
    this.lastMessage = '';
    this.lastMessageType = '';
  }

  // Download recorded audio
  downloadAudio() {
    if (!this.audioBlob) return;

    const url = URL.createObjectURL(this.audioBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `recorded_audio_${Date.now()}.webm`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    this.showMessage('Audio downloaded successfully!', 'success');
  }

  // Handle file selection for upload
  onAudioFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) {
      this.selectedAudioFile = file;
      this.cdr.detectChanges();
    }
  }

  // Process uploaded audio file
  async processUploadedAudio() {
    if (!this.selectedAudioFile) return;

    try {
      // Clean up any existing recording
      await this.cleanupRecording();

      // Clean up previous audio URL
      if (this.audioUrl) {
        URL.revokeObjectURL(this.audioUrl);
        this.audioUrl = null;
      }

      // Convert file to blob
      this.audioBlob = new Blob([await this.selectedAudioFile.arrayBuffer()], {
        type: this.selectedAudioFile.type
      });

      this.audioUrl = URL.createObjectURL(this.audioBlob);
      this.audioDuration = 'N/A'; // We don't know duration for uploaded files
      this.audioSize = (this.selectedAudioFile.size / 1024 / 1024).toFixed(2);
      this.audioSource = 'uploaded';

      this.status = 'Audio file loaded successfully';
      this.showMessage(`Audio file "${this.selectedAudioFile.name}" loaded successfully!`, 'success');

      // Clear the file input
      this.selectedAudioFile = null;

      this.cdr.detectChanges();

    } catch (error) {
      console.error('File processing error:', error);
      this.status = 'Failed to load audio file';
      this.showMessage('Failed to load audio file', 'error');
    }
  }

  // Get language keys for template iteration
  getLanguageKeys(): string[] {
    return Object.keys(this.languageCounts);
  }

  // Format file size for display
  formatFileSize(bytes: number): string {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  }
}
