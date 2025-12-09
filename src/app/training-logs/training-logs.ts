import { Component, OnInit } from '@angular/core';
import { CommonModule } from '@angular/common';
import { HttpClient } from '@angular/common/http';
import { HttpClientModule } from '@angular/common/http';

interface TrainingLog {
  id: number;
  timestamp: string;
  epoch?: number;
  accuracy?: number;
  loss?: number;
  val_accuracy?: number;
  val_loss?: number;
  metadata?: any;
}

@Component({
  selector: 'app-training-logs',
  imports: [CommonModule, HttpClientModule],
  templateUrl: './training-logs.html',
  styleUrl: './training-logs.scss',
})
export class TrainingLogs implements OnInit {
  logs: TrainingLog[] = [];
  chartData: any[] = [];
  isLoading = false;
  status = 'Ready';
  private backendUrl = 'http://localhost:8000'; // Direct backend URL

  constructor(private http: HttpClient) {}

  ngOnInit() {
    this.refreshLogs();
  }

  async refreshLogs() {
    this.isLoading = true;
    this.status = 'Loading training logs...';

    try {
      const response = await this.http.get<{logs: TrainingLog[]}>(`${this.backendUrl}/training-logs`).toPromise();
      if (response) {
        this.logs = response.logs;
        this.processChartData();
        this.status = `Loaded ${this.logs.length} log entries`;
      }
    } catch (error) {
      console.error('Error fetching logs:', error);
      this.status = 'Failed to load logs: ' + (error as any).message;
    } finally {
      this.isLoading = false;
    }
  }

  private processChartData() {
    // Filter only epoch metrics
    const epochLogs = this.logs.filter(log => log.metadata?.type === 'epoch_metrics');
    this.chartData = epochLogs;
  }

  getLogType(metadata: any): string {
    if (!metadata) return 'Unknown';
    switch (metadata.type) {
      case 'training_started':
        return 'Training Started';
      case 'epoch_metrics':
        return 'Epoch Metrics';
      case 'training_completed':
        return 'Training Completed';
      case 'training_error':
        return 'Training Error';
      default:
        return metadata.type || 'Unknown';
    }
  }
}
