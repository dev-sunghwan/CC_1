# Face Recognition System - Dockerfile
# Optimized for CPU-based inference with OpenCV RTSP support

FROM ubuntu:22.04

# Prevent interactive prompts during build
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    # Python and build tools
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    # OpenCV dependencies
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libglib2.0-0 \
    # FFmpeg for RTSP streaming
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    # Network tools
    curl \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/output /app/models

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose ports (optional, for future web interface)
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import cv2; import numpy; print('OK')" || exit 1

# Default command
CMD ["python3", "src/main.py"]
