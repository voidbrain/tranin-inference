import { Component } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';

@Component({
  selector: 'app-listen',
  imports: [CommonModule, FormsModule],
  templateUrl: './listen.html',
  styleUrl: './listen.scss',
})
export class Listen {
  selectedLanguage = 'en';
  isRecording = false;
  transcript = '';
  status = 'Ready';

  private mediaRecorder: MediaRecorder | null = null;
  private audioChunks: Blob[] = [];
  private whisperPipeline: any = null;

  async ngOnInit() {
    await this.loadWhisperModel();
  }

  private async loadWhisperModel() {
    this.status = 'Loading Whisper model...';
    try {
      // Import the pipeline from transformers.js
      const { pipeline } = await import('@huggingface/transformers');

      this.whisperPipeline = await pipeline('automatic-speech-recognition', 'Xenova/whisper-tiny.en');
      this.status = 'Model loaded';
    } catch (error) {
      console.error('Error loading model:', error);
      this.status = 'Failed to load model: ' + (error as Error).message;
    }
  }

  async startRecording() {
    if (!this.whisperPipeline) {
      this.status = 'Model not loaded yet';
      return;
    }

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      this.mediaRecorder = new MediaRecorder(stream);
      this.audioChunks = [];

      this.mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          this.audioChunks.push(event.data);
        }
      };

      this.mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' });
        await this.transcribeAudio(audioBlob);
        // Stop all tracks
        stream.getTracks().forEach(track => track.stop());
      };

      this.mediaRecorder.start();
      this.isRecording = true;
      this.status = 'Recording...';
    } catch (error) {
      console.error('Error starting recording:', error);
      this.status = 'Failed to start recording: ' + (error as Error).message;
    }
  }

  stopRecording() {
    if (this.mediaRecorder && this.isRecording) {
      this.mediaRecorder.stop();
      this.isRecording = false;
      this.status = 'Processing audio...';
    }
  }

  private async transcribeAudio(audioBlob: Blob) {
    if (!this.whisperPipeline) {
      return;
    }

    this.status = 'Transcribing...';
    try {
      // Convert blob to array buffer
      const arrayBuffer = await audioBlob.arrayBuffer();
      const audioFile = new File([arrayBuffer], 'audio.wav', { type: 'audio/wav' });

      // Run transcription
      const result = await this.whisperPipeline(audioFile, {
        language: this.selectedLanguage,
        task: 'transcribe'
      });

      this.transcript = result.text;
      this.status = 'Transcription complete';
    } catch (error) {
      console.error('Error transcribing audio:', error);
      this.status = 'Transcription failed: ' + (error as Error).message;
    }
  }
}
