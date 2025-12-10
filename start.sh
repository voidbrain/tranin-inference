#!/bin/bash

echo "Starting AI Training Application with Docker Compose..."

# Check if docker and docker-compose are available
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed or not in PATH"
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed or not in PATH"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "Stopping containers..."
    docker-compose down
    exit 0
}

# Set trap for Ctrl+C
trap cleanup INT TERM

# Use development setup by default
DOCKER_FILE="docker-compose.yml"

# Check for --dev flag
if [ "$1" = "--dev" ]; then
    DOCKER_FILE="docker-compose.dev.yml"
    echo "Using development configuration..."
fi

# Start services
echo "Building and starting containers..."
docker-compose -f $DOCKER_FILE up --build

# Cleanup (if normal exit)
cleanup
