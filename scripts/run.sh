#!/bin/bash

# Face Recognition System - Run Script

set -e

echo "=================================="
echo "Starting Face Recognition System"
echo "=================================="

# Check if .env exists
if [ ! -f .env ]; then
    echo "Error: .env file not found"
    echo "Please run ./scripts/build.sh first"
    exit 1
fi

# Check if Docker is running
if ! docker info &> /dev/null; then
    echo "Error: Docker is not running"
    exit 1
fi

# Determine docker-compose command
if command -v docker-compose &> /dev/null; then
    DOCKER_COMPOSE="docker-compose"
else
    DOCKER_COMPOSE="docker compose"
fi

# Allow X11 forwarding (Linux only)
if [ "$(uname)" == "Linux" ]; then
    echo "Enabling X11 forwarding..."
    xhost +local:docker
fi

# Start the system
echo "Starting containers..."
$DOCKER_COMPOSE up

# Cleanup on exit
if [ "$(uname)" == "Linux" ]; then
    echo "Disabling X11 forwarding..."
    xhost -local:docker
fi

echo "=================================="
echo "System stopped"
echo "=================================="
