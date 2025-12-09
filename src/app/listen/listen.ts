import { Component, OnInit } from '@angular/core';
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
export class Listen implements OnInit {
  private backendUrl = 'http://backend:8000';

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

  // Processing states
  isTranscribing = false;
  isUploading = false;
  isTraining = false;
  trainingDataCount = 0;

  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private recordingStartTime: number = 0;

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.updateTrainingDataCount();
  }

  async startRecording() {
    try {
      this.status = 'Requesting microphone access...';
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];
      this.recordingStartTime = Date.now();

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = () => {
        const duration = Math.round((Date.now() - this.recordingStartTime) / 1000);
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        const size = (audioBlob.size / 1024 / 1024).toFixed(2); // MB

        this.audioBlob = audioBlob;
        this.audioDuration = duration.toString();
        this.audioSize = size;

        this.status = 'Audio recorded successfully';
        this.showMessage('Audio recorded successfully!', 'success');

        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.status = 'Recording... (Click Stop when done)';
      this.clearMessages();

    } catch (error) {
      console.error('Recording error:', error);
      this.status = 'Recording failed: ' + (error as Error).message;
      this.showMessage('Failed to start recording', 'error');
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.isRecording = false;
    }
  }

  resetRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
    }

    this.audioBlob = null;
    this.audioDuration = '0';
    this.audioSize = '0';
    this.isRecording = false;
    this.transcript = '';
    this.status = 'Recording reset';
    this.clearMessages();
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

    } catch (error: any) {
      console.error('Training start error:', error);
      this.status = 'Training start failed';
      this.showMessage('Failed to start training: ' + error.message, 'error');
    } finally {
      this.isTraining = false;
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
