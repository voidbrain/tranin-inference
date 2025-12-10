import { Component, OnInit, OnDestroy } from '@angular/core';
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
  selectedLanguage = 'en';
  isRecording = false;
  transcript = '';
  status = 'Ready';
  whisperStatus = 'Backend-powered';
  lastMessage = '';
  lastMessageType: 'success' | 'error' | '' = '';

  // Audio state
  audioBlob: Blob | null = null;
  audioDuration = '0';
  audioSize = '0';
  isStartingRecording = false;

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

  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private recordingStartTime: number = 0;
  private stream: MediaStream | null = null;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.updateTrainingDataCount();
  }

  ngOnDestroy() {
    // Clean up media resources when component is destroyed
    this.cleanupRecording();

    // Stop training status polling
    this.stopTrainingStatusPolling();
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
        const duration = Math.round((Date.now() - this.recordingStartTime) / 1000);
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
        const size = (audioBlob.size / 1024 / 1024).toFixed(2); // MB

        this.audioBlob = audioBlob;
        this.audioDuration = duration.toString();
        this.audioSize = size;
        this.isRecording = false;
        this.isStartingRecording = false;

        this.status = 'Audio recorded successfully';
        this.showMessage('Audio recorded successfully!', 'success');

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

      this.mediaRecorder.start(100); // Collect data every 100ms
      this.isRecording = true;
      this.status = 'Recording... (Click Stop when done)';
      this.clearMessages();

    } catch (error) {
      console.error('Recording error:', error);
      this.status = 'Recording failed: Permission denied or no microphone access';
      this.showMessage('Failed to start recording - check microphone permissions', 'error');
      this.isStartingRecording = false;
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();

      // Set a timeout to force stop if onstop callback doesn't fire
      setTimeout(() => {
        if (this.isRecording) {
          console.warn('MediaRecorder onstop callback did not fire, forcing cleanup');
          this.forceStopRecording();
        }
      }, 500);

    } else if (this.isRecording) {
      // If mediaRecorder is not available but recording is supposedly active,
      // force cleanup
      this.forceStopRecording();
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
      const duration = Math.round((Date.now() - this.recordingStartTime) / 1000);
      const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
      const size = (audioBlob.size / 1024 / 1024).toFixed(2);

      this.audioBlob = audioBlob;
      this.audioDuration = duration.toString();
      this.audioSize = size;
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

    // Reset state
    this.audioBlob = null;
    this.audioDuration = '0';
    this.audioSize = '0';
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

  async transcribeAudio() {
    if (!this.audioBlob) return;

    this.isTranscribing = true;
    this.status = 'Transcribing audio...';

    try {
      const formData = new FormData();
      const filename = `audio_${Date.now()}.wav`;
      formData.append('audio', this.audioBlob, filename);

      const response = await this.http.post<{transcription: string}>(`${this.backendUrl}/transcribe-audio`, formData).toPromise();

      this.transcript = response?.transcription || 'Transcription failed';
      this.status = 'Transcription complete';
      this.showMessage('Transcription completed successfully!', 'success');

    } catch (error: any) {
      console.error('Transcription error:', error);
      this.transcript = 'Error: Transcription failed';
      this.status = 'Transcription failed';
      this.showMessage('Transcription failed: ' + error.message, 'error');
    } finally {
      this.isTranscribing = false;
    }
  }

  async uploadForTraining() {
    if (!this.audioBlob) return;

    this.isUploading = true;
    this.status = 'Uploading audio for training...';

    try {
      const formData = new FormData();
      const filename = `speech_${Date.now()}_${this.selectedLanguage}.wav`;
      formData.append('audio_file', this.audioBlob, filename);
      formData.append('language', this.selectedLanguage);
      formData.append('transcript', this.transcript);

      const response = await this.http.post(`${this.backendUrl}/upload-speech-training-data`, formData).toPromise();

      this.status = 'Audio uploaded for speech training';
      this.showMessage('Audio uploaded for training successfully!', 'success');
      this.updateTrainingDataCount();

    } catch (error: any) {
      console.error('Upload error:', error);
      this.status = 'Upload failed';
      this.showMessage('Upload failed: ' + error.message, 'error');
    } finally {
      this.isUploading = false;
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

      const response = await this.http.post(`${this.backendUrl}/whisper-fine-tune-lora`, trainingRequest).toPromise();

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
        const response = await this.http.get(`${this.backendUrl}/whisper-training-status-details`).toPromise();
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

  async resetTrainingData() {
    // This would need a separate endpoint to clear speech training data
    // For now, just reset the local count
    this.trainingDataCount = 0;
    this.showMessage('Training data count reset (backend cleanup needed)', 'success');
  }

  private async updateTrainingDataCount() {
    try {
      // For now, just show a static count until we implement the speech training backend
      const response = await this.http.get<{count: number}>(`${this.backendUrl}/speech-training-count`).toPromise();
      this.trainingDataCount = response?.count || 0;
    } catch {
      this.trainingDataCount = 0;
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
}
