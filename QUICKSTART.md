# Quick Start Guide

Get the Face Recognition System running in under 10 minutes!

## Prerequisites

- **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
- 16GB+ RAM recommended
- RTSP camera URL

## 5-Minute Setup

### 1. Clone or Download

```bash
git clone <repository-url>
cd CC_1
```

### 2. Configure

**Windows:**
```cmd
copy .env.example .env
notepad .env
```

**Linux/Mac:**
```bash
cp .env.example .env
nano .env
```

Edit the RTSP URL:
```bash
RTSP_URL=rtsp://your-camera-ip:554/profile2/media.smp
```

### 3. Build

**Windows:**
```cmd
scripts\build.bat
```

**Linux/Mac:**
```bash
chmod +x scripts/*.sh
./scripts/build.sh
```

This will:
- Create required directories
- Build Docker image (~5-10 minutes first time)
- Download InsightFace models

### 4. Run

**Windows:**
```cmd
scripts\run.bat
```

**Linux/Mac:**
```bash
./scripts/run.sh
```

### 5. Test

You should see:
- Video stream window opening
- Faces being detected with bounding boxes
- FPS counter in top-left corner

**Controls:**
- Press `q` to quit
- Press `s` to save snapshot
- Press `r` to reset tracker

## Next Steps

### Register Known Faces

1. **Stop the system** (Ctrl+C or press 'q')

2. **Run registration tool:**
   ```bash
   docker-compose run face-recognition python3 src/database_manager.py
   ```

3. **Position face in camera and press 's'**

4. **Enter details:**
   - Person ID: `john_doe`
   - Name: `John Doe`
   - Role: `Employee`

5. **Restart the system:**
   ```bash
   docker-compose up
   ```

Now the system will recognize registered faces!

## Common Issues

### "Cannot connect to RTSP stream"

**Solution:**
```bash
# Test RTSP URL directly
ffmpeg -rtsp_transport tcp -i YOUR_RTSP_URL -frames:v 1 test.jpg
```

If this fails, check:
- Camera IP address and port
- Network connectivity
- Firewall settings

### "Display window not showing" (Linux)

**Solution:**
```bash
# Enable X11 forwarding
xhost +local:docker
export DISPLAY=:0
```

### "Low FPS / High CPU usage"

**Solution - Edit .env:**
```bash
DETECTION_INTERVAL=2    # Process every 2nd frame
DETECTION_SIZE=320      # Reduce detection size
```

### "Docker not found"

**Windows:**
- Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)

**Linux:**
```bash
sudo apt-get update
sudo apt-get install docker.io docker-compose
```

## Architecture Overview

```
┌─────────────┐
│ RTSP Camera │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│   GStreamer     │  ← Stable stream capture
│   (TCP/RTSP)    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  InsightFace    │  ← Face detection
│  (SCRFD)        │     & recognition
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   BYTETrack     │  ← Multi-person
│   (Tracking)    │     tracking
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Display/Logging │
└─────────────────┘
```

## Configuration Quick Reference

### `.env` File

```bash
# Camera
RTSP_URL=rtsp://camera-ip:554/stream

# Performance (adjust for your system)
DETECTION_INTERVAL=1     # 1=every frame, 2=every 2nd frame
DETECTION_THRESHOLD=0.5  # Higher = fewer false positives
RECOGNITION_THRESHOLD=0.4 # Higher = stricter matching

# Features
DISPLAY_ENABLED=true     # Set false for headless mode
SAVE_VIDEO=false         # Set true to record output
```

### Performance Presets

**High Accuracy (Slower)**
```bash
DETECTION_INTERVAL=1
DETECTION_SIZE=640
```

**Balanced**
```bash
DETECTION_INTERVAL=2
DETECTION_SIZE=640
```

**High Speed (Lower Accuracy)**
```bash
DETECTION_INTERVAL=3
DETECTION_SIZE=320
```

## Testing the System

Run the test suite:

```bash
docker-compose run face-recognition python3 test_system.py
```

This will verify:
- ✓ All dependencies installed correctly
- ✓ GStreamer working
- ✓ InsightFace models loaded
- ✓ All modules functioning

## What's Next?

1. **Register more faces** - Build your face database
2. **Tune performance** - Adjust settings for your hardware
3. **Set up monitoring** - Track system metrics
4. **Deploy to production** - See [DEPLOYMENT.md](DEPLOYMENT.md)

## Help & Support

- **Full documentation:** [README.md](README.md)
- **Deployment guide:** [DEPLOYMENT.md](DEPLOYMENT.md)
- **Logs:** Check `logs/face_recognition.log`
- **Issues:** Open a GitHub issue

## Quick Commands Reference

```bash
# Build system
./scripts/build.sh        # Linux/Mac
scripts\build.bat         # Windows

# Run system
./scripts/run.sh          # Linux/Mac
scripts\run.bat           # Windows

# Register faces
docker-compose run face-recognition python3 src/database_manager.py

# View logs
docker-compose logs -f

# Stop system
docker-compose down

# Rebuild after changes
docker-compose build --no-cache

# Test system
docker-compose run face-recognition python3 test_system.py
```

---

**Ready to go?** Just run `./scripts/build.sh` (Linux/Mac) or `scripts\build.bat` (Windows) and you'll be detecting faces in minutes!
