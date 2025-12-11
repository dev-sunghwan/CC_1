# Face Recognition System - Dockerfile
# Optimized for CPU-based inference with GStreamer RTSP support

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
    cmake \
    pkg-config \
    # Cairo and gobject for PyGObject
    libcairo2-dev \
    libgirepository1.0-dev \
    # GStreamer and plugins
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-rtsp \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    python3-gst-1.0 \
    gir1.2-gstreamer-1.0 \
    gir1.2-gst-plugins-base-1.0 \
    # OpenCV dependencies
    libopencv-dev \
    python3-opencv \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    # Network tools
    curl \
    wget \
    ca-certificates \
    # X11 support for display (optional)
    libx11-6 \
    libxcb1 \
    libxau6 \
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
    CMD python3 -c "import gi; gi.require_version('Gst', '1.0'); from gi.repository import Gst; Gst.init(None)" || exit 1

# Default command
CMD ["python3", "src/main.py"]
