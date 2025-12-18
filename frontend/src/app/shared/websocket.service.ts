import { Injectable } from '@angular/core';
import { Observable, Subject, BehaviorSubject } from 'rxjs';
import { distinctUntilChanged } from 'rxjs/operators';

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

    // Create WebSocket connection - use backend URL instead of frontend host
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const backendHost = window.location.hostname === 'localhost'
      ? 'localhost:8000'
      : 'backend:8000';
    const wsUrl = `${protocol}//${backendHost}${url}`;

    const socket = new WebSocket(wsUrl);

    socket.onopen = () => {
      console.log(`WebSocket connected: ${endpointId}`);
      connectionStatus.next(true);
    };

    socket.onmessage = (event) => {
      try {
        const data: WebSocketMessage = JSON.parse(event.data);

        // Handle backend status updates
        if (data.type === 'backend_status_update') {
          this.handleBackendStatusUpdate(data);
        }

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
   * Backend connection states
   * disconnected: Cannot reach backend at all
   * connecting: Backend responding but not fully ready
   * connected: Backend fully ready
   */
  private backendConnectionState: 'disconnected' | 'connecting' | 'connected' = 'disconnected';
  private backendReadyCheckInterval: any = null;
  private backendStateSubject = new BehaviorSubject<'disconnected' | 'connecting' | 'connected'>('disconnected');

  startBackendReadyCheck() {
    if (this.backendReadyCheckInterval) {
      clearInterval(this.backendReadyCheckInterval);
    }

    // Use the same backend URL logic as components
    const backendUrl = window.location.hostname === 'localhost'
      ? 'http://localhost:8000'
      : 'http://backend:8000';

    // Check backend status every 2 seconds
    this.backendReadyCheckInterval = setInterval(async () => {
      try {
        const response = await fetch(`${backendUrl}/health`, {
          method: 'GET',
          signal: AbortSignal.timeout(5000) // 5 second timeout
        });

        if (response.ok) {
          const data = await response.json();

          // Determine new state based on backend response
          let newState: 'disconnected' | 'connecting' | 'connected';

          if (data.services_loaded === true && data.endpoints_registered === true) {
            newState = 'connected';
          } else {
            newState = 'connecting';
          }

          // Update state if changed
          if (newState !== this.backendConnectionState) {
            const oldState = this.backendConnectionState;
            this.backendConnectionState = newState;
            console.log(`Backend state changed: ${oldState} -> ${newState}`);
            this.backendStateSubject.next(newState);

            // If backend is now fully connected, stop polling
            if (newState === 'connected') {
              console.log('Backend is fully connected, stopping health polling');
              this.stopBackendReadyCheck();
            }
          }
        } else {
          // Health endpoint not returning 200 - backend is disconnected
          this.updateBackendState('disconnected');
        }
      } catch (error) {
        // Network error or timeout - backend is disconnected
        this.updateBackendState('disconnected');
      }
    }, 2000);
  }

  private updateBackendState(newState: 'disconnected' | 'connecting' | 'connected') {
    if (newState !== this.backendConnectionState) {
      const oldState = this.backendConnectionState;
      this.backendConnectionState = newState;
      console.log(`Backend state changed: ${oldState} -> ${newState}`);
      this.backendStateSubject.next(newState);

      // If backend becomes disconnected, restart polling
      if (newState === 'disconnected') {
        console.log('Backend disconnected, will continue polling');
      }
    }
  }

  stopBackendReadyCheck() {
    if (this.backendReadyCheckInterval) {
      clearInterval(this.backendReadyCheckInterval);
      this.backendReadyCheckInterval = null;
    }
  }

  /**
   * Get current backend connection state
   */
  getBackendConnectionState(): 'disconnected' | 'connecting' | 'connected' {
    return this.backendConnectionState;
  }

  /**
   * Check if backend is fully ready
   */
  isBackendReady(): boolean {
    return this.backendConnectionState === 'connected';
  }

  /**
   * Get backend state observable
   */
  getBackendState(): Observable<'disconnected' | 'connecting' | 'connected'> {
    return this.backendStateSubject.asObservable().pipe(distinctUntilChanged());
  }

  /**
   * Enhanced connection status that considers both WebSocket state and backend readiness
   */
  getEnhancedConnectionStatusString(endpointId: string): string {
    // Return backend connection state directly
    return this.backendConnectionState;
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
   * Handle backend status updates received via WebSocket
   */
  private handleBackendStatusUpdate(data: any): void {
    const phase = data.phase;
    const status = data.status;

    console.log(`Received backend status update: phase=${phase}, status=${status}`);

    // WebSocket status updates are primarily for informational purposes
    // The actual state transitions are handled by the HTTP polling mechanism
    // which checks the /health endpoint for the definitive state
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
