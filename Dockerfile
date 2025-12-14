# Face Recognition System - Single-Stage Dockerfile
# Optimized for CPU-based inference

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    # OpenCV dependencies
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    libglib2.0-0 \
    # FFmpeg for RTSP streaming
    ffmpeg \
    libavcodec58 \
    libavformat58 \
    libswscale5 \
    # Network tools
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Copy and install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/

# Create necessary directories
RUN mkdir -p /app/data /app/logs /app/output /app/models

# Set Python path
ENV PYTHONPATH=/app/src:$PYTHONPATH

# Expose web interface port
EXPOSE 8080

# Health check using the /health endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health', timeout=5)" || exit 1

# Default command
CMD ["python3", "src/main.py"]
