FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies for ML workloads including audio processing
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    libglib2.0-0 \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for data and models
RUN mkdir -p data models train_logs

# Create a script to ensure directories exist with correct permissions
RUN echo '#!/bin/bash\nmkdir -p /tmp/librosa_cache /app/data/speech/waiting/en /app/data/speech/waiting/it /app/data/speech/processed /app/data/vision/colors/waiting /app/data/vision/colors/processed /app/data/vision/digits/waiting /app/data/vision/digits/processed /app/models /app/train_logs && echo "Directories created successfully"' > /app/create_dirs.sh && chmod +x /app/create_dirs.sh

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start the application with directory creation
CMD ["/bin/bash", "-c", "export LIBROSA_CACHE_DIR=/tmp/librosa_cache && export LIBROSA_CACHE_LEVEL=0 && export NUMBA_CACHE_DIR=/tmp/numba_cache && /app/create_dirs.sh && uvicorn main:app --host 0.0.0.0 --port 8000"]
