#!/bin/bash

# Face Recognition System - Build Script

set -e

echo "=================================="
echo "Building Face Recognition System"
echo "=================================="

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Warning: docker-compose not found, using 'docker compose' instead"
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p data logs output models config

# Copy environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo "Please edit .env file with your configuration"
fi

# Build Docker image
echo "Building Docker image..."
$DOCKER_COMPOSE build

echo "=================================="
echo "Build complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "1. Edit .env file with your RTSP URL and settings"
echo "2. Run: ./scripts/run.sh"
echo ""
