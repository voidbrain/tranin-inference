import { Component, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClientModule, HttpClient } from '@angular/common/http';

interface TrainingStatus {
  images_waiting: number;
  training_sessions: number;
  ready_for_training: boolean;
}

interface TrainingRequest {
  epochs: number;
  batch_size: number;
  val_split: number;
}

@Component({
  selector: 'app-training-queue',
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './training-queue.html',
  styleUrl: './training-queue.scss',
})
export class TrainingQueue {
  private backendUrl = 'http://localhost:8000';

  // Signals for reactive state
  trainingStatus = signal<TrainingStatus | null>(null);
  isLoading = signal(false);
  isTraining = signal(false);
  trainingError = signal<string | null>(null);
  trainingSuccess = signal<string | null>(null);

  // Training parameters
  epochs = signal(10);
  batch_size = signal(16);
  val_split = signal(0.2);

  constructor(private http: HttpClient) {
    this.loadTrainingStatus();
  }

  async loadTrainingStatus() {
    try {
      const response: any = await this.http.get(`${this.backendUrl}/training-queue-status`).toPromise();
      this.trainingStatus.set(response);
    } catch (error) {
      console.error('Failed to load training queue status:', error);
      this.trainingStatus.set({
        images_waiting: 0,
        training_sessions: 0,
        ready_for_training: false
      });
    }
  }

  async startTraining() {
    if (!this.trainingStatus()?.ready_for_training) {
      this.trainingError.set("No images ready for training");
      return;
    }

    this.isTraining.set(true);
    this.trainingError.set(null);
    this.trainingSuccess.set(null);

    try {
      const request: TrainingRequest = {
        epochs: this.epochs(),
        batch_size: this.batch_size(),
        val_split: this.val_split()
      };

      const response = await this.http.post(`${this.backendUrl}/train`, request).toPromise();
      console.log('Training started:', response);

      this.trainingSuccess.set("Training started successfully! Check training logs for progress.");
      this.loadTrainingStatus(); // Refresh status

    } catch (error: any) {
      console.error('Failed to start training:', error);
      this.trainingError.set(`Failed to start training: ${error.message}`);
    } finally {
      this.isTraining.set(false);
    }
  }

  // Reactive computed signals
  canStartTraining = () => this.trainingStatus()?.ready_for_training && !this.isTraining();
  hasError = () => this.trainingError() !== null;
  hasSuccess = () => this.trainingSuccess() !== null;
}
