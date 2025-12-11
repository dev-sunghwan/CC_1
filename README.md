# Face Recognition System

A real-time face recognition system using InsightFace and GStreamer for RTSP stream processing. Designed for multi-person tracking and identification with Docker deployment.

## Features

- **Real-time RTSP Stream Processing** using GStreamer (optimized for NAT/port-forwarded environments)
- **Face Detection** with InsightFace SCRFD (high accuracy, CPU-optimized)
- **5-Point Facial Landmark Extraction**
- **Face Alignment** and normalization
- **ArcFace Embedding Generation** (512-dimensional vectors)
- **Face Recognition** with similarity matching
- **Multi-Person Tracking** using BYTETrack algorithm
- **Persistent Face Database** with easy management
- **Multi-threaded Architecture** for high performance
- **Docker Deployment** with full isolation

## System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   RTSP Stream Input                     │
│              (GStreamer + rtspsrc)                      │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Frame Capture Thread                       │
│           (Queue-based buffering)                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│         Detection & Recognition Thread                  │
│    (InsightFace: SCRFD + Landmarks + ArcFace)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Tracking Thread                            │
│           (BYTETrack Algorithm)                         │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Display & Logging                              │
│      (OpenCV + Video Writer)                            │
└─────────────────────────────────────────────────────────┘
```

## Hardware Requirements

Tested on:
- **CPU**: Intel Core i7-13700 (20 threads) or equivalent
- **RAM**: 16GB minimum, 32GB+ recommended
- **GPU**: Not required (CPU inference is sufficient)
- **OS**: Windows 11 with Docker Desktop (WSL2) or Linux

## Quick Start

### Windows

1. **Install Prerequisites**
   - [Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Git (optional)

2. **Clone and Build**
   ```cmd
   git clone <repository-url>
   cd CC_1
   scripts\build.bat
   ```

3. **Configure**
   - Edit `.env` file with your RTSP URL and settings

4. **Run**
   ```cmd
   scripts\run.bat
   ```

### Linux

1. **Install Prerequisites**
   ```bash
   sudo apt-get update
   sudo apt-get install docker.io docker-compose git
   ```

2. **Clone and Build**
   ```bash
   git clone <repository-url>
   cd CC_1
   chmod +x scripts/*.sh
   ./scripts/build.sh
   ```

3. **Configure**
   ```bash
   nano .env  # Edit with your settings
   ```

4. **Run**
   ```bash
   ./scripts/run.sh
   ```

## Configuration

Edit `.env` file (created from `.env.example`):

```bash
# RTSP Stream
RTSP_URL=rtsp://your-camera-ip:554/profile2/media.smp

# Detection
DETECTION_THRESHOLD=0.5    # Face detection confidence (0-1)
DETECTION_INTERVAL=1       # Process every Nth frame

# Recognition
RECOGNITION_THRESHOLD=0.4  # Face matching threshold (0-1)

# Display
DISPLAY_ENABLED=true       # Show visualization window
SAVE_VIDEO=false          # Save output to video file
```

## Face Database Management

### Register New Faces

**Method 1: Interactive Registration**
```bash
# Inside container
docker exec -it face_recognition_system python3 src/database_manager.py
```

**Method 2: Programmatic**
```python
from database_manager import FaceDatabaseManager
import numpy as np

db = FaceDatabaseManager()
db.add_identity(
    person_id="john_doe",
    embedding=embedding_vector,  # 512-dim numpy array
    metadata={
        "name": "John Doe",
        "role": "Employee",
        "department": "Engineering"
    }
)
db.save()
```

### List Identities
```python
db = FaceDatabaseManager()
print(db.list_identities())
print(db.get_statistics())
```

### Export/Import
```python
# Export to JSON
db.export_json("backup.json")

# Import from JSON
db.import_json("backup.json", merge=True)
```

## Usage Examples

### Basic Usage
```bash
# Run with default settings
docker-compose up

# Run with custom RTSP URL
docker-compose run face-recognition python3 src/main.py --rtsp-url rtsp://192.168.1.100:554/stream
```

### Advanced Options
```bash
# Disable display (headless mode)
python3 src/main.py --no-display

# Save output video
python3 src/main.py --save-video --output /app/output/recording.mp4

# Process every 2nd frame (faster, lower accuracy)
python3 src/main.py --detection-interval 2
```

### Keyboard Controls (when display is enabled)

- **`q`**: Quit application
- **`s`**: Save snapshot
- **`r`**: Reset tracker

## Project Structure

```
CC_1/
├── src/
│   ├── main.py                      # Main application
│   ├── stream_capture.py            # GStreamer RTSP capture
│   ├── face_recognition_pipeline.py # InsightFace pipeline
│   ├── tracker.py                   # BYTETrack implementation
│   ├── config.py                    # Configuration management
│   └── database_manager.py          # Face database tools
├── scripts/
│   ├── build.sh / build.bat         # Build scripts
│   └── run.sh / run.bat             # Run scripts
├── data/                            # Runtime data
├── logs/                            # Log files
├── output/                          # Output videos/snapshots
├── models/                          # Model files (auto-downloaded)
├── Dockerfile                       # Container definition
├── docker-compose.yml               # Docker orchestration
├── requirements.txt                 # Python dependencies
├── .env.example                     # Environment template
└── README.md                        # This file
```

## Technical Details

### Why GStreamer?

GStreamer was selected over OpenCV and PyAV for critical reasons:

- **RTSP Stability**: Reliable handling of NAT/port-forwarded streams
- **FU-A NAL Support**: Automatic fragmentation unit handling for H.264
- **Reconnection Logic**: Built-in automatic reconnection
- **TCP Fallback**: Graceful fallback from UDP to TCP
- **Docker Compatibility**: Full support in containerized environments

### InsightFace Models

- **Detector**: SCRFD (Sample and Computation Redistribution Face Detector)
- **Landmark**: 5-point facial landmark detection
- **Embedding**: ArcFace (512-dimensional embeddings)
- **Model Pack**: buffalo_l (balanced accuracy/speed)

### Tracking Algorithm

BYTETrack implementation with:
- Kalman filter for motion prediction
- IoU + embedding similarity matching
- Multi-hypothesis tracking
- Occlusion handling

## Performance Optimization

### CPU Optimization
- Multi-threaded pipeline (3 worker threads)
- Frame skipping (configurable interval)
- Queue-based buffering
- Efficient numpy operations

### Memory Management
- Bounded queues (max 10 frames)
- Automatic oldest-frame dropping
- Periodic garbage collection

### Recommended Settings

**High Accuracy (Slower)**
```
DETECTION_INTERVAL=1
DETECTION_THRESHOLD=0.5
RECOGNITION_THRESHOLD=0.45
```

**Balanced**
```
DETECTION_INTERVAL=2
DETECTION_THRESHOLD=0.5
RECOGNITION_THRESHOLD=0.4
```

**High Speed (Lower Accuracy)**
```
DETECTION_INTERVAL=3
DETECTION_THRESHOLD=0.6
RECOGNITION_THRESHOLD=0.35
```

## Troubleshooting

### Stream Not Connecting
```bash
# Test RTSP connection directly
docker run --rm -it --network=host \
  jrottenberg/ffmpeg:4.1-alpine \
  -rtsp_transport tcp -i rtsp://your-url -frames:v 1 -f null -
```

### Display Not Working (Linux)
```bash
# Enable X11 forwarding
xhost +local:docker
export DISPLAY=:0
```

### Low FPS
- Increase `DETECTION_INTERVAL` to process fewer frames
- Reduce `DETECTION_SIZE` in config
- Check CPU usage: `docker stats`

### Face Not Recognized
- Lower `RECOGNITION_THRESHOLD` (more lenient)
- Ensure face is well-lit and frontal during registration
- Re-register with multiple angles

## API Reference

### RTSPStreamCapture
```python
capture = RTSPStreamCapture(rtsp_url, queue_size=10)
capture.start()
frame = capture.get_frame(timeout=1.0)
is_alive = capture.is_alive()
capture.stop()
```

### FaceRecognitionPipeline
```python
pipeline = FaceRecognitionPipeline(det_thresh=0.5)
faces = pipeline.detect_and_extract(frame)
faces = pipeline.recognize_faces(faces, threshold=0.4)
annotated = pipeline.draw_results(frame, faces)
```

### BYTETracker
```python
tracker = BYTETracker(det_thresh=0.5)
tracks = tracker.update(detections)
tracker.reset()
```

## Development

### Running Tests
```bash
# Test stream capture
docker exec -it face_recognition_system python3 src/stream_capture.py

# Test face detection
docker exec -it face_recognition_system python3 src/face_recognition_pipeline.py

# Test tracker
docker exec -it face_recognition_system python3 src/tracker.py
```

### Adding Custom Models
```python
# In face_recognition_pipeline.py
self.app = FaceAnalysis(
    name='your_model_pack',
    providers=['CPUExecutionProvider']
)
```

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please submit issues and pull requests.

## Contact

For questions and support, please open an issue on GitHub.

---

**Built with:**
- [InsightFace](https://github.com/deepinsight/insightface) - Face recognition
- [GStreamer](https://gstreamer.freedesktop.org/) - Media streaming
- [OpenCV](https://opencv.org/) - Computer vision
- [Docker](https://www.docker.com/) - Containerization
