import { Injectable } from '@angular/core';
import { Observable, Subject, BehaviorSubject } from 'rxjs';

export interface WebSocketMessage {
  type: string;
  session_id: string;
  timestamp: string;
  progress?: number;
  message?: string;
  status?: string;
  success?: boolean;
}

@Injectable({
  providedIn: 'root'
})
export class WebSocketService {
  private sockets: Map<string, WebSocket> = new Map();
  private subjects: Map<string, Subject<WebSocketMessage>> = new Map();
  private connectionStatus: Map<string, BehaviorSubject<boolean>> = new Map();

  constructor() {}

  /**
   * Connect to a WebSocket endpoint
   * @param endpointId Unique identifier for this connection
   * @param url WebSocket URL
   */
  connect(endpointId: string, url: string): Observable<WebSocketMessage> {
    // Close existing connection if it exists
    this.disconnect(endpointId);

    // Create subject for this endpoint
    const subject = new Subject<WebSocketMessage>();
    const connectionStatus = new BehaviorSubject<boolean>(false);
    this.subjects.set(endpointId, subject);
    this.connectionStatus.set(endpointId, connectionStatus);

    // Create WebSocket connection
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}${url}`;

    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log(`WebSocket connected: ${endpointId}`);
      connectionStatus.next(true);
    };

    socket.onmessage = (event) => {
      try {
        const data: WebSocketMessage = JSON.parse(event.data);
        subject.next(data);
      } catch (error) {
        console.error(`Failed to parse WebSocket message for ${endpointId}:`, error);
      }
    };

    socket.onerror = (error) => {
      console.error(`WebSocket error for ${endpointId}:`, error);
      connectionStatus.next(false);
    };

    socket.onclose = () => {
      console.log(`WebSocket disconnected: ${endpointId}`);
      connectionStatus.next(false);
    };

    this.sockets.set(endpointId, socket);

    return subject.asObservable();
  }

  /**
   * Disconnect from a WebSocket endpoint
   */
  disconnect(endpointId: string): void {
    const socket = this.sockets.get(endpointId);
    const subject = this.subjects.get(endpointId);
    const connectionStatus = this.connectionStatus.get(endpointId);

    if (socket) {
      socket.close();
      this.sockets.delete(endpointId);
    }

    if (subject) {
      subject.complete();
      this.subjects.delete(endpointId);
    }

    if (connectionStatus) {
      connectionStatus.next(false);
      this.connectionStatus.delete(endpointId);
    }
  }

  /**
   * Get connection status observable
   */
  getConnectionStatus(endpointId: string): Observable<boolean> {
    if (!this.connectionStatus.has(endpointId)) {
      this.connectionStatus.set(endpointId, new BehaviorSubject<boolean>(false));
    }
    return this.connectionStatus.get(endpointId)!.asObservable();
  }

  /**
   * Check if connected to an endpoint
   */
  isConnected(endpointId: string): boolean {
    const connectionStatus = this.connectionStatus.get(endpointId);
    return connectionStatus ? connectionStatus.value : false;
  }

  /**
   * Check if connecting to an endpoint
   */
  isConnecting(endpointId: string): boolean {
    const socket = this.sockets.get(endpointId);
    return socket ? socket.readyState === WebSocket.CONNECTING : false;
  }

  /**
   * Get connection status string for template binding
   */
  getConnectionStatusString(endpointId: string): string {
    if (this.isConnected(endpointId)) {
      return 'connected';
    } else if (this.isConnecting(endpointId)) {
      return 'connecting';
    } else {
      return 'disconnected';
    }
  }

  /**
   * Check if backend is fully initialized and ready
   * This polls the backend health endpoint to determine if initialization is complete
   */
  private backendReady = false;
  private backendReadyCheckInterval: any = null;

  startBackendReadyCheck() {
    if (this.backendReadyCheckInterval) {
      clearInterval(this.backendReadyCheckInterval);
    }

    // Use the same backend URL logic as components
    const backendUrl = window.location.hostname === 'localhost'
      ? 'http://localhost:8000'
      : 'http://backend:8000';

    // Check backend readiness every 2 seconds
    this.backendReadyCheckInterval = setInterval(async () => {
      try {
        const response = await fetch(`${backendUrl}/health`);
        if (response.ok) {
          const data = await response.json();
          // Backend is ready when it returns healthy status and has services
          this.backendReady = data.status === 'healthy' && data.services && data.services.length > 0;
        } else {
          this.backendReady = false;
        }
      } catch (error) {
        this.backendReady = false;
      }
    }, 2000);
  }

  stopBackendReadyCheck() {
    if (this.backendReadyCheckInterval) {
      clearInterval(this.backendReadyCheckInterval);
      this.backendReadyCheckInterval = null;
    }
    this.backendReady = false;
  }

  isBackendReady(): boolean {
    return this.backendReady;
  }

  /**
   * Enhanced connection status that considers both WebSocket state and backend readiness
   */
  getEnhancedConnectionStatusString(endpointId: string): string {
    const socketState = this.getConnectionStatusString(endpointId);

    // If backend is not ready, show connecting regardless of WebSocket state
    if (!this.isBackendReady()) {
      return 'connecting';
    }

    // Backend is ready, return actual WebSocket connection status
    return socketState;
  }

  /**
   * Get connection status as a simple boolean value (for templates)
   */
  getConnectionStatusValue(endpointId: string): boolean {
    return this.isConnected(endpointId);
  }

  /**
   * Send a message through the WebSocket
   */
  sendMessage(endpointId: string, message: any): void {
    const socket = this.sockets.get(endpointId);
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      console.warn(`Cannot send message: WebSocket ${endpointId} is not connected`);
    }
  }

  /**
   * Disconnect all WebSocket connections
   */
  disconnectAll(): void {
    this.sockets.forEach((socket, endpointId) => {
      this.disconnect(endpointId);
    });
  }
}
