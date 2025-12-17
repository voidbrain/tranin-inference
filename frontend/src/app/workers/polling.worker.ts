interface WorkerMessage {
  type: 'start' | 'stop';
  endpointId: string;
  url?: string;
  intervalMs?: number;
  headers?: Record<string, string>;
}

class PollingWorker {
  private intervalId: number | null = null;
  private endpointId: string = '';
  private url: string = '';
  private intervalMs: number = 3000;
  private headers: Record<string, string> = {};

  onmessage = (event: MessageEvent<WorkerMessage>) => {
    const message = event.data;

    switch (message.type) {
      case 'start':
        this.startPolling(message);
        break;
      case 'stop':
        this.stopPolling();
        break;
    }
  };

  private startPolling(message: WorkerMessage) {
    this.endpointId = message.endpointId!;
    this.url = message.url!;
    this.intervalMs = message.intervalMs || 3000;
    this.headers = message.headers || {};

    this.stopPolling(); // Clear any existing interval

    // Start polling
    this.intervalId = setInterval(() => {
      this.pollEndpoint();
    }, this.intervalMs);

    // Send initial start message
    postMessage({
      type: 'start',
      endpoint: this.endpointId
    });
  }

  private stopPolling() {
    if (this.intervalId) {
      clearInterval(this.intervalId);
      this.intervalId = null;

      postMessage({
        type: 'stop',
        endpoint: this.endpointId
      });
    }
  }

  private async pollEndpoint() {
    try {
      const response = await fetch(this.url, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
          ...this.headers
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();

      postMessage({
        type: 'data',
        endpoint: this.endpointId,
        data
      });
    } catch (error) {
      postMessage({
        type: 'error',
        endpoint: this.endpointId,
        error: (error as Error).message
      });
    }
  }
}

// Create and start the worker
const worker = new PollingWorker();

// Handle messages
onmessage = worker.onmessage;
