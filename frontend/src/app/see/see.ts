import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, signal, computed, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { HttpClientModule } from '@angular/common/http';
import { WebWorkerService, PollingMessage } from '../shared/web-worker.service';
import { WebSocketService, WebSocketMessage } from '../shared/websocket.service';

interface Detection {
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  id: number;
  mode: 'digits' | 'colors'; // Remember which mode created this detection
}

@Component({
  selector: 'app-see',
  imports: [CommonModule, FormsModule, HttpClientModule],
  templateUrl: './see.html',
  styleUrl: './see.scss',
})
export class See implements AfterViewInit, OnDestroy {
  @ViewChild('videoElement') videoElement!: ElementRef<HTMLVideoElement>;
  @ViewChild('canvasElement') canvasElement!: ElementRef<HTMLCanvasElement>;
  @ViewChild('blueBoxCanvas') blueBoxCanvas!: ElementRef<HTMLCanvasElement>;
  @ViewChild('staticImageElement') staticImageElement!: ElementRef<HTMLImageElement>;

  // Convert to Angular Signals for reactive state
  isComponentReady = signal(false);
  isCameraActive = signal(false);
  isDetecting = signal(false);
  isLoading = signal(false);
  isCorrectionMode = signal(false);
  addBoxMode = signal(false);
  newBoxLabel = signal('');
  status = signal('Ready');
  canvasWidth = signal(480);
  canvasHeight = signal(640);

  detections = signal<Detection[]>([]);

  // Computed signal for canSendToBackend
  canSendToBackend = computed(() => this.detections().length > 0);

  // Correction mode signals
  selectedBox = signal<Detection | null>(null);
  drawingNewBox = signal(false);
  newBox = signal({ x: 0, y: 0, width: 0, height: 0 });

  // Image display signals
  showStaticImage = signal(false);
  capturedImage = signal('');

  // Tab navigation
  activeTab = signal('detect');

  // Detection mode - need getter/setter for ngModel compatibility with signals
  private _detectionMode = signal<'digits' | 'colors'>('digits');

  get detectionMode(): string {
    return this._detectionMode();
  }

  set detectionMode(value: 'digits' | 'colors') {
    this._detectionMode.set(value);
    // Note: We don't call drawDetections() here anymore because we want existing boxes
    // to keep their original line style based on the mode they were created with
    this.loadAvailableTags();
  }

  // Available tags for quick selection - now loaded from backend
  availableTags = signal<string[]>([]);

  // Method to load tags from backend based on detection mode
  async loadAvailableTags() {
    const currentMode = this._detectionMode();
    if (!currentMode) {
      // If no detection mode is set yet, don't try to load tags
      return;
    }

    try {
      const response = await this.http.get<any>(`${this.backendUrl}/vision/tags/${currentMode}`).toPromise();
      this.availableTags.set(response.tags);
    } catch (error) {
      console.warn('Failed to load tags from backend:', error);
      // Fallback to hardcoded if backend unavailable
      const fallbackTags = currentMode === 'digits'
        ? ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        : ['red', 'blue', 'green', 'yellow', 'orange', 'purple'];
      this.availableTags.set(fallbackTags);
    }
  }

  // Training Queue functionality (merged from TrainingQueue component)
  trainingStatus = signal<any>(null);
  isTraining = signal(false);
  trainingError = signal<string | null>(null);
  trainingSuccess = signal<string | null>(null);

  // Training parameters
  epochs = signal(10);
  batch_size = signal(16);
  val_split = signal(0.2);
  learning_rate = signal(0.001);
  use_lora = signal(false);

  // Specialized LoRA training
  trainingType = signal<'digits' | 'colors' | 'combined' | ''>('');
  loraRank = signal(4); // Use rank 4 for smaller adapters as suggested
  loraLearningRate = signal(0.001);

  // Model selection for detection
  selectedModel = signal<'base' | 'colors' | 'digits' | 'merged'>('base');

  // Model loading methods
  async selectModel(modelType: 'base' | 'colors' | 'digits' | 'merged') {
    if (this.selectedModel() === modelType) {
      // Already selected, deselect to base model and clear detections
      this.selectedModel.set('base');
      this.clearDetections();
      this.status.set('Using base model for detection');
      try {
        await this.http.post(`${this.backendUrl}/vision/reset-model`, {}).toPromise();
      } catch (error: any) {
        console.error('Failed to reset to base model:', error);
      }
      return;
    }

    this.selectedModel.set(modelType);
    this.clearDetections(); // Clear existing detections when switching models
    this.status.set(`Switching to ${modelType} model...`);

    try {
      if (modelType === 'base') {
        await this.http.post(`${this.backendUrl}/vision/reset-model`, {}).toPromise();
        this.status.set('Using base YOLO model for detection');
      } else {
        // Load LoRA adapter or merged model
        await this.loadLoraAdapter(modelType);
        this.status.set(`Using ${modelType} specialized adapter for detection`);
      }
    } catch (error: any) {
      console.error(`Failed to switch to ${modelType} model:`, error);
      this.status.set(`Failed to load ${modelType} model: ${error.message}`);
      this.selectedModel.set('base'); // Revert to base model
      this.clearDetections(); // Clear detections on error
    }
  }

  getModelStatusText(): string {
    const model = this.selectedModel();
    return model === 'base' ? 'Base Model (base)' :
           model === 'colors' ? 'Colors LoRA (base + color)' :
           model === 'digits' ? 'Digits LoRA (base + digit)' :
           'Merged Model (base + color + digit)';
  }

  // Training Logs functionality (merged from TrainingLogs component)
  trainingLogs = signal<any[]>([]);

  // Real-time training status polling
  trainingStatusPollingInterval = signal<any>(null);

  // Box selection/dragging
  selectedBoxForDrag = signal<Detection | null>(null);
  isDraggingBox = signal(false);
  dragStartX = signal(0);
  dragStartY = signal(0);

  private stream: MediaStream | null = null;
  private yoloPipeline: any = null;
  private drawingBox = false;
  private boxStartX = 0;
  private boxStartY = 0;
  private nextId = 0;
  // Use localhost for development, backend service for production/Docker
  private backendUrl = window.location.hostname === 'localhost'
    ? 'http://localhost:8000'
    : 'http://backend:8000';

  // Web worker polling subscriptions
  private pollingSubscriptions: any[] = [];

  // WebSocket connection
  private webSocketSubscription: any = null;

  // Connection status
  backendConnected = signal(false);
  backendConnectionState = signal<'disconnected' | 'connecting' | 'connected'>('disconnected');
  private connectionStatusSubscription: any = null;
  private backendReadySubscription: any = null;

  constructor(private http: HttpClient, private webWorkerService: WebWorkerService, private webSocketService: WebSocketService) {}

  // Public getters for template access
  getWebSocketService(): WebSocketService {
    return this.webSocketService;
  }

  async ngAfterViewInit() {
    await this.loadYOLOModel();
    await this.loadAvailableTags();

    // Add onload event handler to static image element to draw blue box
    if (this.staticImageElement) {
      this.staticImageElement.nativeElement.onload = () => {
        this.drawBlueBox();
      };
    }

    // Start backend readiness checking
    this.webSocketService.startBackendReadyCheck();

    // Subscribe to WebSocket connection status
    this.connectionStatusSubscription = this.webSocketService.getConnectionStatus('vision-training').subscribe(
      (connected: boolean) => {
        this.backendConnected.set(connected);
      }
    );

    // Subscribe to backend state changes
    this.backendReadySubscription = this.webSocketService.getBackendState().subscribe(
      (state: 'disconnected' | 'connecting' | 'connected') => {
        console.log('Backend state changed:', state);
        this.backendConnectionState.set(state);
        // Force Angular change detection to update the LED status
        // This will trigger the enhanced connection status string to be recalculated
      }
    );

    this.isComponentReady.set(true);
  }

  ngOnDestroy() {
    this.stopCamera();
    this.stopTrainingStatusPolling();

    // Clean up all polling subscriptions
    this.pollingSubscriptions.forEach(sub => sub.unsubscribe());
    this.pollingSubscriptions = [];

    // Clean up WebSocket subscription
    if (this.webSocketSubscription) {
      this.webSocketSubscription.unsubscribe();
      this.webSocketSubscription = null;
    }

    // Clean up connection status subscription
    if (this.connectionStatusSubscription) {
      this.connectionStatusSubscription.unsubscribe();
      this.connectionStatusSubscription = null;
    }

    // Clean up backend ready subscription
    if (this.backendReadySubscription) {
      this.backendReadySubscription.unsubscribe();
      this.backendReadySubscription = null;
    }

    // Stop backend readiness checking
    this.webSocketService.stopBackendReadyCheck();

    // Disconnect WebSocket
    this.webSocketService.disconnect('vision-training');
  }

  private async loadYOLOModel() {
    this.status.set('Initializing YOLO model (offline mode)...');

    // Use mock implementation directly for offline functionality
    // No network requests to HuggingFace - completely local/offline
    this.yoloPipeline = {
      mock: true,
      async detect(imageData: any) {
        // Return realistic mock detections for testing UI functionality
        const objects = ['person', 'car', 'bicycle', 'dog', 'cat', 'chair', 'phone', 'book'];
        const numBoxes = Math.floor(Math.random() * 4) + 1; // 1-4 random boxes
        const results = [];

        for (let i = 0; i < numBoxes; i++) {
          const x1 = Math.floor(Math.random() * 300) + 50; // Random position
          const y1 = Math.floor(Math.random() * 250) + 50;
          const width = Math.floor(Math.random() * 120) + 60; // Random size
          const height = Math.floor(Math.random() * 120) + 80;
          const confidence = (Math.random() * 0.3 + 0.7).toFixed(2); // 0.7-1.0 range

          results.push({
            label: objects[Math.floor(Math.random() * objects.length)],
            score: parseFloat(confidence),
            box: {
              xMin: x1,
              yMin: y1,
              xMax: x1 + width,
              yMax: y1 + height
            }
          });
        }

        // Simulate some processing delay
        await new Promise(resolve => setTimeout(resolve, 500));

        return results;
      }
    };

    this.status.set('YOLO model ready (offline mode)');
    console.log('Offline YOLO model initialized');
  }

  async startCamera() {
    try {
      this.isLoading.set(true);
      this.status.set('Requesting camera access...');

      if (!this.videoElement) {
        this.status.set('Error: Video element not found');
        this.isLoading.set(false);
        return;
      }

      this.stream = await navigator.mediaDevices.getUserMedia({
        video: { width: this.canvasWidth(), height: this.canvasHeight() }
      });

      this.videoElement.nativeElement.srcObject = this.stream;

      await new Promise((resolve) => {
        this.videoElement.nativeElement.onloadedmetadata = resolve;
      });

      // Force immediate UI update
      this.isCameraActive.set(true);
      this.isLoading.set(false);
      this.status.set('Camera active');

      // Clear any existing detections and draw blue box when camera starts
      this.detections.set([]);
      this.clearCanvas();

      console.log("About to check blueBoxCanvas availability...");
      console.log("blueBoxCanvas exists:", !!this.blueBoxCanvas);

      if (!this.blueBoxCanvas) {
        console.log("blueBoxCanvas ViewChild not available yet!");
        // Try again after a longer delay
        setTimeout(() => {
          console.log("Retrying blue box draw after longer delay...");
          console.log("blueBoxCanvas now exists:", !!this.blueBoxCanvas);
          if (this.blueBoxCanvas) {
            this.drawBlueBox();
            console.log("Blue box drawn after retry");
          } else {
            console.log("blueBoxCanvas still not available!");
          }
        }, 500);
      } else {
        console.log("blueBoxCanvas available, drawing immediately");
        this.drawBlueBox();
        console.log("Blue box drawn immediately");
      }

      console.log('Camera started successfully, isCameraActive:', this.isCameraActive());

    } catch (error) {
      console.error('Error accessing camera:', error);
      this.isLoading.set(false);
      this.status.set('Failed to access camera: ' + (error as Error).message);
    }
  }

  stopCamera() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    this.isCameraActive.set(false);
    this.detections.set([]);
    this.clearCanvas();
    // Reset to live video mode
    this.showStaticImage.set(false);
    this.capturedImage.set('');
    this.status.set('Camera stopped');
  }

  // Handle image upload
  onImageUpload(event: any) {
    const file = event.target.files[0];
    if (!file) return;

    // Validate file type
    if (!file.type.startsWith('image/')) {
      this.status.set('Error: Please select a valid image file');
      return;
    }

    // Validate file size (max 10MB)
    const maxSize = 10 * 1024 * 1024; // 10MB
    if (file.size > maxSize) {
      this.status.set('Error: Image file is too large (max 10MB)');
      return;
    }

    this.status.set('Loading uploaded image...');

    const reader = new FileReader();
    reader.onload = (e) => {
      const imageData = e.target?.result as string;
      this.capturedImage.set(imageData);
      this.showStaticImage.set(true);
      this.isCameraActive.set(false); // Disable camera mode when using uploaded image

      // Clear any existing detections
      this.detections.set([]);
      this.clearCanvas();

      this.status.set('Image uploaded successfully. Ready for detection.');

      // Draw blue box on the uploaded image
      if (this.staticImageElement) {
        this.staticImageElement.nativeElement.onload = () => {
          this.drawBlueBox();
        };
      }
    };

    reader.onerror = () => {
      this.status.set('Error: Failed to read the image file');
    };

    reader.readAsDataURL(file);
  }

  // just stop the camera stream but keep the UI in static image mode
  private stopCameraStream() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    // Note: Don't set isCameraActive to false to avoid messing with UI state
  }

  async detectObjects() {
    this.isDetecting.set(true);
    this.status.set('Capturing image and detecting objects...');

    try {
      let imageBlob: Blob;
      let capturedFrameData: string;

      // If we already have a captured image, use it instead of re-capturing
      if (this.capturedImage() && this.showStaticImage()) {
        // Reuse the existing captured image
        this.status.set('Using previously captured image...');
        const imageResponse = await fetch(this.capturedImage());
        imageBlob = await imageResponse.blob();
        capturedFrameData = this.capturedImage();
      } else {
        // First time - capture from video
        if (!this.isCameraActive() || !this.videoElement) {
          this.status.set('Camera not active - cannot capture image');
          return;
        }

        // Capture the current video frame
        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = this.videoElement.nativeElement.videoWidth;
        captureCanvas.height = this.videoElement.nativeElement.videoHeight;
        const captureCtx = captureCanvas.getContext('2d')!;

        if (captureCanvas.width === 0 || captureCanvas.height === 0) {
          this.status.set('Video feed not available - cannot capture image');
          return;
        }

        captureCtx.drawImage(this.videoElement.nativeElement, 0, 0);

        // Convert canvas to blob for FormData upload
        imageBlob = await new Promise<Blob>((resolve) => {
          captureCanvas.toBlob((blob) => {
            resolve(blob!);
          }, 'image/jpeg', 0.95);
        });

        // Store the captured frame for display
        capturedFrameData = captureCanvas.toDataURL('image/jpeg');
        this.capturedImage.set(capturedFrameData);

        // Stop the camera stream and switch to static image mode
        this.stopCameraStream();
        this.showStaticImage.set(true);
      }

      // Create FormData to send image to backend
      const formData = new FormData();
      formData.append('file', imageBlob, 'captured_image.jpg');

      // Add blue box coordinates for digit prioritization
      const blueBoxCoords = {
        x: 140,  // Center of 480px canvas minus half of 200px box
        y: 220,  // Center of 640px canvas minus half of 200px box
        width: 200,
        height: 200
      };
      formData.append('blue_box', JSON.stringify(blueBoxCoords));

      // Call backend detection API with model as query parameter
      this.status.set('Sending image to backend for detection...');
      const url = `${this.backendUrl}/vision/detect?model=${this.selectedModel()}`;

      // Use fetch instead of Angular HTTP client for multipart form data to avoid UTF-8 decoding issues
      const response = await fetch(url, {
        method: 'POST',
        body: formData
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const responseData = await response.json();

      // Process backend response with separate digit/color detections
      const digitDetections = (responseData.digitDetections || []).map((detection: any) => ({
        label: detection.label,
        confidence: Math.round(detection.score * 100),
        x: Math.round(detection.box.xMin),
        y: Math.round(detection.box.yMin),
        width: Math.round(detection.box.xMax - detection.box.xMin),
        height: Math.round(detection.box.yMax - detection.box.yMin),
        id: this.nextId++,
        mode: 'digits' // Digits always get solid lines
      }));

      const colorDetections = (responseData.colorDetections || []).map((detection: any) => ({
        label: detection.label,
        confidence: Math.round(detection.score * 100),
        x: Math.round(detection.box.xMin),
        y: Math.round(detection.box.yMin),
        width: Math.round(detection.box.xMax - detection.box.xMin),
        height: Math.round(detection.box.yMax - detection.box.yMin),
        id: this.nextId++,
        mode: 'colors' // Colors always get dashed lines
      }));

      // Combine both arrays for the final detections (workaround since frontend expects single array)
      const newDetectionResults = [...digitDetections, ...colorDetections];

      // Replace existing detections for fresh analysis instead of accumulating
      this.detections.set(newDetectionResults);
      this.drawDetections();
      this.status.set(`Detected ${responseData.digital_count || 0} digits, ${responseData.color_count || 0} colors using ${this.getModelStatusText()} (Real Model: ${!responseData.mock})`);
    } catch (error) {
      console.error('Backend detection error:', error);
      this.status.set('Detection failed: ' + (error as Error).message);
    } finally {
      this.isDetecting.set(false);
    }
  }

  private getLabelName(label: string): string {
    // Map YOLO classes to readable names (simplified mapping)
    const labels: { [key: string]: string } = {
      '0': 'person',
      '1': 'bicycle',
      '2': 'car',
      '3': 'motorcycle',
      '4': 'airplane',
      '5': 'bus',
      '6': 'train',
      '7': 'truck',
      // Add more mappings as needed
    };
    return labels[label] || label;
  }

  private drawDetections() {
    const canvas = this.canvasElement.nativeElement;
    const ctx = canvas.getContext('2d')!;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Always redraw the blue box (when camera is active OR when we have a captured image)
    if (this.isCameraActive() || this.showStaticImage()) {
      this.drawBlueBox();
    }

    this.detections().forEach((detection: Detection) => {
      // Draw box
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 2;
      if (detection.mode === 'colors') {
        ctx.setLineDash([5, 5]); // Dashed line for colors
      } else {
        ctx.setLineDash([]); // Solid line for digits
      }
      ctx.strokeRect(detection.x, detection.y, detection.width, detection.height);

      // Draw label
      ctx.fillStyle = '#00FF00';
      ctx.font = '16px Arial';
      ctx.fillText(`${detection.label} (${detection.confidence}%)`, detection.x, detection.y - 5);
    });
  }

  private clearCanvas() {
    if (!this.canvasElement) return;
    const canvas = this.canvasElement.nativeElement;
    const ctx = canvas.getContext('2d')!;
    if (ctx) {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  }

  private drawBlueBox() {
    if (!this.blueBoxCanvas) return;
    const canvas = this.blueBoxCanvas.nativeElement;
    const ctx = canvas.getContext('2d')!;
    if (!ctx) return;

    // Clear the blue box canvas first
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Calculate center position for 200x200 box on 480x640 canvas
    const boxWidth = 200;
    const boxHeight = 200;
    const centerX = (canvas.width - boxWidth) / 2;  // (480 - 200) / 2 = 140
    const centerY = (canvas.height - boxHeight) / 2; // (640 - 200) / 2 = 220

    // Draw blue rectangle
    ctx.strokeStyle = '#0000FF'; // Blue color
    ctx.lineWidth = 3;
    ctx.strokeRect(centerX, centerY, boxWidth, boxHeight);

    // Optional: Add a label
    ctx.fillStyle = '#0000FF';
    ctx.font = '16px Arial';
    ctx.fillText('Center Box (200x200)', centerX, centerY - 5);
    console.log("Blue box drawn on canvas", ctx);
  }

  enterCorrectionMode() {
    this.isCorrectionMode.set(true);
    this.status.set('Correction mode: Edit boxes');
  }

  async sendToBackend() {
    try {
      const trainingData = {
        imageData: this.getCurrentFrameData(),
        detections: this.detections().map((d: Detection) => ({
          label: d.label,
          bbox: [d.x, d.y, d.x + d.width, d.y + d.height], // Convert to [x1,y1,x2,y2] format
          confidence: d.confidence
        }))
      };

      const response = await this.http.post(`${this.backendUrl}/vision/upload-training-data`, trainingData).toPromise();
      console.log('Backend response:', response);
      this.status.set('Data uploaded successfully. Training can now be initiated.');

      // Refresh training status to show updated counts
      await this.loadTrainingStatus();

    } catch (error) {
      console.error('Error sending to backend:', error);
      this.status.set('Failed to send data to backend: ' + (error as any).message);
    }
  }

  async sendSpecializedToBackend(trainingType: 'digits' | 'colors') {
    try {
      const specializedData = {
        training_type: trainingType,
        imageData: this.getCurrentFrameData(),
        detections: this.detections().map((d: Detection) => ({
          label: d.label,
          bbox: [d.x, d.y, d.x + d.width, d.y + d.height], // Convert to [x1,y1,x2,y2] format
          confidence: d.confidence
        }))
      };

      const response = await this.http.post(`${this.backendUrl}/vision/upload-specialized-training-data`, specializedData).toPromise();
      console.log(`Specialized ${trainingType} backend response:`, response);
      this.status.set(`${trainingType.charAt(0).toUpperCase() + trainingType.slice(1)} training data uploaded successfully.`);

      // Refresh specialized training counts
      await this.loadSpecializedCount(trainingType);

    } catch (error) {
      console.error(`Error sending ${trainingType} to backend:`, error);
      this.status.set(`Failed to send ${trainingType} data to backend: ${error}`);
    }
  }

  async startSpecializedTraining(trainingType?: 'digits' | 'colors') {
    // Use the selected training type from the UI if not explicitly passed
    const typeToTrain = trainingType || this.trainingType();
    if (!typeToTrain || typeToTrain === '' as any) {
      this.trainingError.set('Please select a training type first.');
      return;
    }

    try {
      // Only check training data for individual types (digits and colors)
      // Combined training merges existing digits+colors data
      if (typeToTrain !== 'combined') {
        const isReady = await this.checkSpecializedTrainingReady(typeToTrain as 'digits' | 'colors');
        if (!isReady) {
          this.trainingError.set(`No ${typeToTrain} training data found. Upload training data first.`);
          return;
        }
      }

      this.isTraining.set(true);
      this.trainingError.set(null);
      this.trainingSuccess.set(null);

      const loraRequest = {
        training_type: typeToTrain,
        epochs: 60, // LoRA training typically needs more epochs
        lora_rank: this.loraRank(),
        learning_rate: this.loraLearningRate()
      };

      const response = await this.http.post(`${this.backendUrl}/vision/train-lora-specialized`, loraRequest).toPromise();
      console.log(`LoRA training started for ${typeToTrain}:`, response);

      this.trainingSuccess.set(`LoRA training started for ${typeToTrain}! Check training logs for progress.`);

      // Start polling for specialized training status updates (only for individual types)
      if (typeToTrain !== 'combined') {
        this.startSpecializedTrainingPolling(typeToTrain as 'digits' | 'colors');
      }

    } catch (error: any) {
      console.error(`Failed to start specialized training for ${typeToTrain}:`, error);
      this.trainingError.set(`Failed to start ${typeToTrain} training: ${error.message}`);
    } finally {
      this.isTraining.set(false);
    }
  }

  async loadSpecializedCount(trainingType: 'digits' | 'colors') {
    try {
      const response: any = await this.http.get(`${this.backendUrl}/vision/specialized-training-count/${trainingType}`).toPromise();
      return response.count || 0;
    } catch (error) {
      console.error(`Failed to load ${trainingType} count:`, error);
      return 0;
    }
  }

  async loadLoraAdapter(trainingType: 'digits' | 'colors' | 'merged' | '' | undefined) {
    if (!trainingType || trainingType === '' as any) {
      this.trainingError.set('Please select a training type first.');
      return;
    }

    try {
      this.trainingError.set(null);
      this.trainingSuccess.set(null);

      const response = await this.http.post(`${this.backendUrl}/vision/load-lora-adapter`, {
        training_type: trainingType
      }).toPromise();

      console.log(`${trainingType} model loaded:`, response);
      const modelText = trainingType === 'merged' ? 'Merged model' : `LoRA adapter for ${trainingType}`;
      this.trainingSuccess.set(`${modelText} loaded successfully!`);
      // Model loaded successfully

    } catch (error: any) {
      console.error(`Failed to load ${trainingType} model:`, error);
      const modelText = trainingType === 'merged' ? 'merged model' : `${trainingType} LoRA adapter`;
      this.trainingError.set(`Failed to load ${modelText}: ${error.message}`);
    }
  }

  async startCombinedModelTraining() {
    try {
      this.trainingError.set(null);
      this.trainingSuccess.set(null);

      // Check if both LoRA adapters exist first
      const digitsLoraExists = await this.checkLoraAdapterExists('digits');
      const colorsLoraExists = await this.checkLoraAdapterExists('colors');

      if (!digitsLoraExists || !colorsLoraExists) {
        const missing = [];
        if (!digitsLoraExists) missing.push('digits');
        if (!colorsLoraExists) missing.push('colors');
        this.trainingError.set(`Missing LoRA adapter(s): ${missing.join(', ')}. Train both digits and colors first.`);
        return;
      }

      // Start combined model training
      const response = await this.http.post(`${this.backendUrl}/vision/create-merged-model`, {}).toPromise();
      console.log('Combined model training started:', response);

      this.trainingSuccess.set('Combined model training started! Check training logs for progress.');
    } catch (error: any) {
      console.error('Failed to start combined model training:', error);
      this.trainingError.set(`Failed to start combined model training: ${error.message}`);
    }
  }

  private async checkLoraAdapterExists(trainingType: 'digits' | 'colors'): Promise<boolean> {
    try {
      // Check if LoRA adapter file exists on server
      const statusResponse = await this.http.get(`${this.backendUrl}/vision/training-queue-status`).toPromise();
      // This is a simplified check - in real implementation, you'd check file system or model status
      return true; // Assume exists for now
    } catch (error) {
      return false;
    }
  }

  private async checkSpecializedTrainingReady(trainingType: 'digits' | 'colors'): Promise<boolean> {
    const count = await this.loadSpecializedCount(trainingType);
    return count > 0;
  }

  private startSpecializedTrainingPolling(trainingType: 'digits' | 'colors') {
    const pollingSubscription = this.webWorkerService.startPolling(
      `vision-training-logs-${trainingType}`,
      `${this.backendUrl}/vision/training-logs`,
      3000
    ).subscribe((message: PollingMessage) => {
      if (message.type === 'data') {
        const logs = message.data.logs || [];
        this.trainingLogs.set(logs);

        // Check if specialized training is completed
        const lastLog = logs[logs.length - 1];
        if (lastLog && lastLog.metadata &&
            lastLog.metadata.type === 'training_completed' &&
            lastLog.metadata.specialization === trainingType) {
          this.stopTrainingStatusPolling();
          this.trainingSuccess.set(`LoRA training for ${trainingType} completed successfully!`);
          this.trainingError.set(null);
        } else if (lastLog && lastLog.metadata &&
                   lastLog.metadata.type === 'training_error' &&
                   lastLog.metadata.specialization === trainingType) {
          this.stopTrainingStatusPolling();
          this.trainingError.set(`LoRA training for ${trainingType} failed: ${lastLog.metadata.error}`);
          this.trainingSuccess.set(null);
        }
      } else if (message.type === 'error') {
        console.error('Failed to poll specialized training logs:', message.error);
      }
    });

    this.pollingSubscriptions.push(pollingSubscription);
  }

  private getCurrentFrameData(): string {
    // Use the captured image data instead of trying to capture from video (which might not be streaming)
    if (this.capturedImage()) {
      return this.capturedImage(); // Return the already-captured frame
    } else if (this.videoElement) {
      // Fallback: try to capture from video element (though it might fail after stream is stopped)
      const canvas = document.createElement('canvas');
      canvas.width = this.videoElement.nativeElement.videoWidth || this.canvasWidth();
      canvas.height = this.videoElement.nativeElement.videoHeight || this.canvasHeight();
      const ctx = canvas.getContext('2d')!;
      ctx.drawImage(this.videoElement.nativeElement, 0, 0);
      return canvas.toDataURL('image/jpeg');
    } else {
      return '';
    }
  }

  trackByIndex(index: number): number {
    return index;
  }

  // Tag selection methods
  selectTag(tag: string) {
    this.newBoxLabel.set(tag);
  }

  toggleAddBoxMode() {
    this.addBoxMode.set(!this.addBoxMode());
    if (this.addBoxMode()) {
      this.newBoxLabel.set(this.newBoxLabel() || this.availableTags()[0] || 'unknown'); // Default to first available tag
      this.status.set('Click and drag on image to add boxes (multiple allowed)');
    } else {
      this.drawingNewBox.set(false);
      this.status.set('Add box mode disabled');
    }
  }

  clearAllBoxes() {
    this.detections.set([]);
    this.drawDetections();
    this.status.set('All boxes cleared');
  }

  private clearDetections() {
    this.detections.set([]);
    this.drawDetections();
    this.status.set('Detections cleared for model change');
  }

  getLabelsList(): string {
    return this.detections().map(d => d.label).join(', ') || 'none';
  }

  // Mouse event handlers for correction mode
  onMouseDown(event: MouseEvent) {
    if (!this.isCorrectionMode() || !this.canvasElement) return;

    const rect = this.canvasElement.nativeElement.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;

    // Check if clicking on existing box
    const clickedBox = this.detections().find(detection =>
      x >= detection.x && x <= detection.x + detection.width &&
      y >= detection.y && y <= detection.y + detection.height
    );

    if (clickedBox) {
      // If right-click, remove the box (same as left-click for now)
      if (event.button === 2 || event.ctrlKey) {
        const currentDetections = this.detections();
        const updatedDetections = currentDetections.filter(d => d.id !== clickedBox.id);
        this.detections.set(updatedDetections);
        this.drawDetections();
        this.status.set(`Removed ${clickedBox.label} box`);
      } else {
        // Start dragging the box
        this.selectedBoxForDrag.set(clickedBox);
        this.isDraggingBox.set(true);
        this.dragStartX.set(x - clickedBox.x);
        this.dragStartY.set(y - clickedBox.y);
        this.status.set(`Dragging ${clickedBox.label} box...`);
      }
    } else if (this.addBoxMode() && !this.isDraggingBox()) {
      // Start drawing a new box
      this.drawingNewBox.set(true);
      this.newBox.set({ x, y, width: 0, height: 0 });
      this.status.set('Drawing new box... (drag to size)');
    }
  }

  onMouseMove(event: MouseEvent) {
    if (this.isDraggingBox() && this.canvasElement) {
      // Handle dragging existing box
      const rect = this.canvasElement.nativeElement.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;

      const draggedBox = this.selectedBoxForDrag();
      if (draggedBox) {
        // Update box position
        const newX = Math.max(0, x - this.dragStartX());
        const newY = Math.max(0, y - this.dragStartY());

        // Keep box within canvas bounds
        const newDetection = {
          ...draggedBox,
          x: Math.min(newX, this.canvasWidth() - draggedBox.width),
          y: Math.min(newY, this.canvasHeight() - draggedBox.height)
        };

        // Update the box in detections
        const currentDetections = this.detections();
        const updatedDetections = currentDetections.map(d =>
          d.id === draggedBox.id ? newDetection : d
        );
        this.detections.set(updatedDetections);
        this.drawDetections();

        // Highlight the dragging box
        this.highlightDraggedBox(newDetection);
      }
    } else if (this.drawingNewBox()) {
      // Handle drawing new box
      const rect = this.canvasElement.nativeElement.getBoundingClientRect();
      const x = event.clientX - rect.left;
      const y = event.clientY - rect.top;
      const currentBox = this.newBox();

      const width = Math.max(0, x - currentBox.x);
      const height = Math.max(0, y - currentBox.y);

      this.newBox.set({ x: currentBox.x, y: currentBox.y, width, height });
      this.drawDetections();
      this.drawCurrentBox();
    }
  }

  onMouseUp(event: MouseEvent) {
    // Handle finishing drag operation
    if (this.isDraggingBox()) {
      this.isDraggingBox.set(false);
      this.selectedBoxForDrag.set(null);
      this.status.set('Finished moving box');
      return;
    }

    // Handle finishing new box drawing
    if (!this.drawingNewBox()) return;

    this.drawingNewBox.set(false);
    const box = this.newBox();

    if (box.width > 10 && box.height > 10 && this.newBoxLabel()) {
      // Add the new box to detections
      const newDetection: Detection = {
        label: this.newBoxLabel(),
        confidence: 100, // User-drawn boxes are manually confirmed
        x: box.x || 0, // Ensure x is not undefined, default to 0,0 if needed
        y: box.y || 0,
        width: box.width,
        height: box.height,
        id: this.nextId++,
        mode: this._detectionMode() // Store the mode this box was created with
      };

      console.log('Adding new detection:', newDetection);

      const currentDetections = this.detections();
      const updatedDetections = [...currentDetections, newDetection];
      this.detections.set(updatedDetections);

      console.log('Total detections now:', updatedDetections.length);

      this.newBoxLabel.set('');
      this.status.set(`Added new "${newDetection.label}" box at (${newDetection.x}, ${newDetection.y})`);

      // Redraw immediately to show the new box
      setTimeout(() => this.drawDetections(), 10);
    }

    this.newBox.set({ x: 0, y: 0, width: 0, height: 0 });
    this.drawDetections();
  }

  private drawCurrentBox() {
    if (!this.drawingNewBox()) return;

    const canvas = this.canvasElement.nativeElement;
    const ctx = canvas.getContext('2d')!;
    const box = this.newBox();

    // Draw the box being drawn
    ctx.strokeStyle = '#FF0000';
    ctx.lineWidth = 2;
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // Fill with semi-transparent red
    ctx.fillStyle = 'rgba(255, 0, 0, 0.1)';
    ctx.fillRect(box.x, box.y, box.width, box.height);
  }

  private highlightDraggedBox(box: Detection) {
    const canvas = this.canvasElement.nativeElement;
    const ctx = canvas.getContext('2d')!;

    // Draw highlighted border
    ctx.strokeStyle = '#FF6B35';
    ctx.lineWidth = 3;
    ctx.strokeRect(box.x - 2, box.y - 2, box.width + 4, box.height + 4);
  }

  // ===== TRAINING QUEUE FUNCTIONALITY (Merged from TrainingQueue) =====

  async ngOnInit() {
    // Load training status on component init
    await this.loadTrainingStatus();
    await this.loadTrainingLogs();
  }

  async loadTrainingStatus() {
    try {
      const response: any = await this.http.get(`${this.backendUrl}/vision/training-queue-status`).toPromise();
      this.trainingStatus.set(response);
    } catch (error) {
      console.error('Failed to load training queue status:', error);
      this.trainingStatus.set({
        images_waiting: 0,
        training_sessions: 0,
        ready_for_training: false,
        last_training_time: null,
        current_model: "YOLOv8n (Base)",
        processed_images: 0,
        total_images_annotated: 0
      });
    }
  }

  async resetModel() {
    try {
      this.trainingError.set(null);
      this.trainingSuccess.set(null);

      const response = await this.http.post(`${this.backendUrl}/vision/reset-model`, {}).toPromise();
      console.log('Model reset:', response);

      this.trainingSuccess.set("Model reset to base YOLOv8n successfully!");
      await this.loadTrainingStatus();

    } catch (error: any) {
      console.error('Failed to reset model:', error);
      this.trainingError.set(`Failed to reset model: ${error.message}`);
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
      const trainingRequest = {
        epochs: this.epochs(),
        batch_size: this.batch_size(),
        val_split: this.val_split(),
        use_lora: this.use_lora(),
        learning_rate: this.learning_rate()
      };

      const response = await this.http.post(`${this.backendUrl}/vision/train`, trainingRequest).toPromise();
      console.log('Training started:', response);

      this.trainingSuccess.set("Training started successfully! Check training logs for progress.");

      // Start polling for training status updates
      this.startTrainingStatusPolling();

      await this.loadTrainingStatus();
      await this.loadTrainingLogs();

    } catch (error: any) {
      console.error('Failed to start training:', error);
      this.trainingError.set(`Failed to start training: ${error.message}`);
    } finally {
      this.isTraining.set(false);
    }
  }

  canStartTraining = () => this.trainingStatus()?.ready_for_training && !this.isTraining();
  hasError = () => this.trainingError() !== null;
  hasSuccess = () => this.trainingSuccess() !== null;

  // ===== TRAINING LOGS FUNCTIONALITY (Merged from TrainingLogs) =====

  async loadTrainingLogs() {
    try {
      const response: any = await this.http.get(`${this.backendUrl}/vision/training-logs`).toPromise();
      this.trainingLogs.set(response.logs || []);
    } catch (error) {
      console.error('Failed to load training logs:', error);
      this.trainingLogs.set([]);
    }
  }

  formatDateTime(timestamp: string | null): string {
    if (!timestamp) return 'Never';

    try {
      const date = new Date(timestamp);
      return date.toLocaleString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit'
      });
    } catch (error) {
      return timestamp;
    }
  }

  trackByLog(index: number, item: any): any {
    return item.id || index;
  }

  // ===== REAL-TIME TRAINING STATUS (WebSocket) =====

  private startTrainingStatusPolling() {
    // Connect to WebSocket for real-time training updates
    this.webSocketSubscription = this.webSocketService.connect(
      'vision-training',
      '/ws/vision/training'
    ).subscribe((message: WebSocketMessage) => {
      if (message.type === 'training_update') {
        // Update training progress in real-time
        this.status.set(`Training: ${message.message || 'Processing...'}`);

        // Update logs if available (for backward compatibility, also load from HTTP endpoint)
        this.loadTrainingLogs();

        // Check for completion
        if (message.status === 'success') {
          this.stopTrainingStatusPolling();
          this.loadTrainingStatus().then(() => {
            const currentModelStatus = this.trainingStatus();
            if (currentModelStatus?.current_model && currentModelStatus.current_model !== "YOLOv8n (Base)") {
              this.trainingSuccess.set("YOLO training completed successfully!");
              this.trainingError.set(null);
            } else {
              this.trainingSuccess.set("YOLO training completed - check model status");
              this.trainingError.set(null);
            }
          });
        } else if (message.status === 'error') {
          this.stopTrainingStatusPolling();
          this.trainingError.set(`YOLO training failed: ${message.message || 'Unknown error'}`);
          this.trainingSuccess.set(null);
        }
      }
    });
  }

  private stopTrainingStatusPolling() {
    if (this.webSocketSubscription) {
      this.webSocketSubscription.unsubscribe();
      this.webSocketSubscription = null;
    }
    this.webSocketService.disconnect('vision-training');
  }
}
