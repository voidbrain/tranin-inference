import { Component, ElementRef, ViewChild, AfterViewInit, OnDestroy, signal, computed } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { HttpClient } from '@angular/common/http';
import { HttpClientModule } from '@angular/common/http';

interface Detection {
  label: string;
  confidence: number;
  x: number;
  y: number;
  width: number;
  height: number;
  id: number;
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

  // Convert to Angular Signals for reactive state
  isComponentReady = signal(false);
  isCameraActive = signal(false);
  isDetecting = signal(false);
  isLoading = signal(false);
  isCorrectionMode = signal(false);
  addBoxMode = signal(false);
  newBoxLabel = signal('');
  status = signal('Ready');
  canvasWidth = signal(640);
  canvasHeight = signal(480);

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
    // When mode changes, update border styles immediately
    this.drawDetections();
    this.loadAvailableTags();
  }

  // Available tags for quick selection - now loaded from backend
  availableTags = signal<string[]>([]);

  // Method to load tags from backend based on detection mode
  async loadAvailableTags() {
    try {
      const response = await this.http.get<any>(`http://backend:8000/tags/${this._detectionMode()}`).toPromise();
      this.availableTags.set(response.tags);
    } catch (error) {
      console.warn('Failed to load tags from backend:', error);
      // Fallback to hardcoded if backend unavailable
      const fallbackTags = this._detectionMode() === 'digits'
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
  trainingType = signal<'digits' | 'colors' | ''>('');
  loraRank = signal(8);
  loraLearningRate = signal(0.001);

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
  private backendUrl = 'http://backend:8000'; // Use backend service directly in dev mode

  constructor(private http: HttpClient) {}

  async ngAfterViewInit() {
    await this.loadYOLOModel();
    await this.loadAvailableTags();
    this.isComponentReady.set(true);
  }

  ngOnDestroy() {
    this.stopCamera();
    this.stopTrainingStatusPolling();
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

  // just stop the camera stream but keep the UI in static image mode
  private stopCameraStream() {
    if (this.stream) {
      this.stream.getTracks().forEach(track => track.stop());
      this.stream = null;
    }
    // Note: Don't set isCameraActive to false to avoid messing with UI state
  }

  async detectObjects() {
    if (!this.yoloPipeline || !this.isCameraActive() || !this.videoElement) return;

    this.isDetecting.set(true);
    this.status.set('Capturing image and detecting objects...');

    try {
      // Capture the current video frame BEFORE running detection
      const captureCanvas = document.createElement('canvas');
      captureCanvas.width = this.videoElement.nativeElement.videoWidth;
      captureCanvas.height = this.videoElement.nativeElement.videoHeight;
      const captureCtx = captureCanvas.getContext('2d')!;
      captureCtx.drawImage(this.videoElement.nativeElement, 0, 0);

      // Store the captured frame
      const capturedFrameData = captureCanvas.toDataURL('image/jpeg');
      console.log('Captured frame data length:', capturedFrameData.length);
      console.log('Captured frame data starts with:', capturedFrameData.substring(0, 50));

      this.capturedImage.set(capturedFrameData);
      console.log('Set capturedImage signal, current value starts with:', this.capturedImage().substring(0, 50));

      // Stop the camera stream and switch to static image mode
      this.stopCameraStream();
      this.showStaticImage.set(true); // Show captured image instead of video

      console.log('Set showStaticImage to true, should display static image now');

      // Update status to reflect stream is stopped but working on image
      this.status.set('detection-finished-stream-paused');

      // Convert to image data for processing
      const imageData = captureCtx.getImageData(0, 0, captureCanvas.width, captureCanvas.height);

      // Run detection - handle both real pipeline and mock
      let result;
      if (this.yoloPipeline.mock) {
        // Mock pipeline
        result = await this.yoloPipeline.detect(imageData);
      } else {
        // Real transformers pipeline
        result = await this.yoloPipeline(imageData);
      }

      // Process results
      const detectionResults = result.map((detection: any, index: number) => ({
        label: this.getLabelName(detection.label),
        confidence: Math.round(detection.score * 100),
        x: detection.box.xMin,
        y: detection.box.yMin,
        width: detection.box.xMax - detection.box.xMin,
        height: detection.box.yMax - detection.box.yMin,
        id: this.nextId++
      }));

      this.detections.set(detectionResults);
      this.drawDetections();
      this.status.set(`Detected ${detectionResults.length} objects (on captured image)`);
    } catch (error) {
      console.error('Error detecting objects:', error);
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

    this.detections().forEach((detection: Detection) => {
      // Draw box
      ctx.strokeStyle = '#00FF00';
      ctx.lineWidth = 2;
      if (this.detectionMode === 'colors') {
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

      const response = await this.http.post(`${this.backendUrl}/upload-training-data`, trainingData).toPromise();
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

      const response = await this.http.post(`${this.backendUrl}/upload-specialized-training-data`, specializedData).toPromise();
      console.log(`Specialized ${trainingType} backend response:`, response);
      this.status.set(`${trainingType.charAt(0).toUpperCase() + trainingType.slice(1)} training data uploaded successfully.`);

      // Refresh specialized training counts
      await this.loadSpecializedCount(trainingType);

    } catch (error) {
      console.error(`Error sending ${trainingType} to backend:`, error);
      this.status.set(`Failed to send ${trainingType} data to backend: ${error}`);
    }
  }

  async startSpecializedTraining(trainingType: 'digits' | 'colors') {
    try {
      const isReady = await this.checkSpecializedTrainingReady(trainingType);
      if (!isReady) {
        this.trainingError.set(`No ${trainingType} training data found. Upload training data first.`);
        return;
      }

      this.isTraining.set(true);
      this.trainingError.set(null);
      this.trainingSuccess.set(null);

      const loraRequest = {
        training_type: trainingType,
        epochs: 60, // LoRA training typically needs more epochs
        lora_rank: this.loraRank(),
        learning_rate: this.loraLearningRate()
      };

      const response = await this.http.post(`${this.backendUrl}/train-lora-specialized`, loraRequest).toPromise();
      console.log(`LoRA training started for ${trainingType}:`, response);

      this.trainingSuccess.set(`LoRA training started for ${trainingType}! Check training logs for progress.`);

      // Start polling for specialized training status updates
      this.startSpecializedTrainingPolling(trainingType);

    } catch (error: any) {
      console.error(`Failed to start specialized training for ${trainingType}:`, error);
      this.trainingError.set(`Failed to start ${trainingType} training: ${error.message}`);
    } finally {
      this.isTraining.set(false);
    }
  }

  async loadSpecializedCount(trainingType: 'digits' | 'colors') {
    try {
      const response: any = await this.http.get(`${this.backendUrl}/specialized-training-count/${trainingType}`).toPromise();
      return response.count || 0;
    } catch (error) {
      console.error(`Failed to load ${trainingType} count:`, error);
      return 0;
    }
  }

  async loadLoraAdapter(trainingType: 'digits' | 'colors') {
    try {
      this.trainingError.set(null);
      this.trainingSuccess.set(null);

      const response = await this.http.post(`${this.backendUrl}/load-lora-adapter`, {
        training_type: trainingType
      }).toPromise();

      console.log(`LoRA adapter loaded for ${trainingType}:`, response);
      this.trainingSuccess.set(`LoRA adapter for ${trainingType} loaded successfully!`);
      // LoRA adapter loaded successfully

    } catch (error: any) {
      console.error(`Failed to load LoRA adapter for ${trainingType}:`, error);
      this.trainingError.set(`Failed to load ${trainingType} LoRA adapter: ${error.message}`);
    }
  }

  private async checkSpecializedTrainingReady(trainingType: 'digits' | 'colors'): Promise<boolean> {
    const count = await this.loadSpecializedCount(trainingType);
    return count > 0;
  }

  private startSpecializedTrainingPolling(trainingType: 'digits' | 'colors') {
    this.trainingStatusPollingInterval.set(setInterval(async () => {
      try {
        const response: any = await this.http.get(`${this.backendUrl}/training-logs`).toPromise();
        const logs = response.logs || [];
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
      } catch (error) {
        console.error('Failed to poll specialized training logs:', error);
      }
    }, 3000));
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
        id: this.nextId++
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
      const response: any = await this.http.get(`${this.backendUrl}/training-queue-status`).toPromise();
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

      const response = await this.http.post(`${this.backendUrl}/reset-model`, {}).toPromise();
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

      const response = await this.http.post(`${this.backendUrl}/train`, trainingRequest).toPromise();
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
      const response: any = await this.http.get(`${this.backendUrl}/training-logs`).toPromise();
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

  // ===== REAL-TIME TRAINING STATUS POLLING (Added) =====

  private startTrainingStatusPolling() {
    // Poll every 3 seconds for training status updates
    this.trainingStatusPollingInterval.set(setInterval(async () => {
      try {
        const response: any = await this.http.get(`${this.backendUrl}/training-logs`).toPromise();
        const logs = response.logs || [];
        this.trainingLogs.set(logs);

        // Check if training is completed by looking for completion message
        const lastLog = logs[logs.length - 1];
        if (lastLog && lastLog.metadata && lastLog.metadata.type === 'training_completed') {
          this.stopTrainingStatusPolling();

          // Check if there's a trained_*.pt file (indicating success)
          await this.loadTrainingStatus();
          const currentModelStatus = this.trainingStatus();
          if (currentModelStatus?.current_model && currentModelStatus.current_model !== "YOLOv8n (Base)") {
            this.trainingSuccess.set("YOLO training completed successfully!");
            this.trainingError.set(null);
          } else {
            this.trainingSuccess.set("YOLO training completed - check model status");
            this.trainingError.set(null);
          }
        } else if (lastLog && lastLog.metadata && lastLog.metadata.type === 'training_error') {
          this.stopTrainingStatusPolling();
          this.trainingError.set(`YOLO training failed: ${lastLog.metadata.error}`);
          this.trainingSuccess.set(null);
        }
      } catch (error) {
        console.error('Failed to poll training logs:', error);
      }
    }, 3000));
  }

  private stopTrainingStatusPolling() {
    if (this.trainingStatusPollingInterval()) {
      clearInterval(this.trainingStatusPollingInterval());
      this.trainingStatusPollingInterval.set(null);
    }
  }
}
