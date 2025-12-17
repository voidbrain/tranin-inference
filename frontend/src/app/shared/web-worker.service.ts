import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';

export interface PollingMessage {
  type: 'start' | 'stop' | 'data' | 'error';
  endpoint: string;
  data?: any;
  error?: string;
}

@Injectable({
  providedIn: 'root'
})
export class WebWorkerService {
  private workers: Map<string, Worker> = new Map();
  private subjects: Map<string, Subject<PollingMessage>> = new Map();

  constructor() {}

  /**
   * Start polling for a specific endpoint
   * @param endpointId Unique identifier for this polling session
   * @param url The URL to poll
   * @param intervalMs Polling interval in milliseconds
   * @param headers Optional headers for the request
   */
  startPolling(endpointId: string, url: string, intervalMs: number = 3000, headers?: Record<string, string>): Observable<PollingMessage> {
    // Stop existing worker if it exists
    this.stopPolling(endpointId);

    // Create subject for this endpoint
    const subject = new Subject<PollingMessage>();
    this.subjects.set(endpointId, subject);

    // Create web worker
    const worker = new Worker(new URL('../workers/polling.worker', import.meta.url), { type: 'module' });

    // Handle messages from worker
    worker.onmessage = (event) => {
      const message: PollingMessage = event.data;
      subject.next(message);
    };

    worker.onerror = (error) => {
      subject.next({
        type: 'error',
        endpoint: endpointId,
        error: error.message
      });
    };

    // Start the worker
    worker.postMessage({
      type: 'start',
      endpointId,
      url,
      intervalMs,
      headers
    });

    this.workers.set(endpointId, worker);

    return subject.asObservable();
  }

  /**
   * Stop polling for a specific endpoint
   */
  stopPolling(endpointId: string): void {
    const worker = this.workers.get(endpointId);
    const subject = this.subjects.get(endpointId);

    if (worker) {
      worker.postMessage({ type: 'stop', endpointId });
      worker.terminate();
      this.workers.delete(endpointId);
    }

    if (subject) {
      subject.complete();
      this.subjects.delete(endpointId);
    }
  }

  /**
   * Stop all polling workers
   */
  stopAllPolling(): void {
    this.workers.forEach((worker, endpointId) => {
      this.stopPolling(endpointId);
    });
  }

  /**
   * Check if polling is active for an endpoint
   */
  isPollingActive(endpointId: string): boolean {
    return this.workers.has(endpointId);
  }
}
