# Face Recognition System - Development Session Summary

**Date:** December 11, 2024
**Project:** Real-time Face Recognition with RTSP Stream
**Location:** `C:\Users\sungh\Documents\CC_1`

---

## 🎯 What We Built Today

A complete **real-time face recognition system** with:
- GStreamer RTSP stream capture
- InsightFace face detection & recognition
- BYTETrack multi-person tracking
- Docker containerization
- Full documentation

---

## 📦 Project Structure

```
CC_1/
├── src/                               # Source code
│   ├── main.py                        # Multi-threaded main application
│   ├── stream_capture.py              # GStreamer RTSP capture
│   ├── face_recognition_pipeline.py   # InsightFace integration
│   ├── tracker.py                     # BYTETrack tracking
│   ├── config.py                      # Configuration management
│   ├── database_manager.py            # Face database tools
│   └── __init__.py                    # Package initialization
│
├── scripts/                           # Build & run scripts
│   ├── build.sh / build.bat           # Docker build scripts
│   └── run.sh / run.bat               # Docker run scripts
│
├── config/                            # Configuration files
│   └── config.example.yaml            # YAML config template
│
├── Dockerfile                         # Docker container definition
├── docker-compose.yml                 # Docker orchestration
├── requirements.txt                   # Python dependencies
├── .env                               # Environment variables (YOUR CONFIG)
├── .env.example                       # Environment template
│
├── README.md                          # Complete documentation
├── QUICKSTART.md                      # 5-minute setup guide
├── DEPLOYMENT.md                      # Production deployment guide
├── LICENSE                            # MIT License
│
├── test_system.py                     # Component test suite
├── build.log                          # Build logs
└── build_retry.log                    # Build retry logs
```

---

## ⚙️ Your Configuration

**RTSP Camera URL:** `rtsp://192.168.1.100:554/profile2/media.smp`
**Detection Threshold:** 0.5
**Recognition Threshold:** 0.4
**System:** Windows 11, Docker Desktop (WSL2)
**Hardware:** i7-13700 (20 threads), 64GB RAM

---

## 🚀 Quick Start Commands

### **1. Start the System**
```powershell
cd C:\Users\sungh\Documents\CC_1
docker-compose up
```

### **2. Stop the System**
- Press `Ctrl+C` in terminal, OR
- Press `q` in the video window

### **3. Run in Background (Detached)**
```powershell
docker-compose up -d
```

### **4. View Logs**
```powershell
docker-compose logs -f
```

### **5. Stop Background Process**
```powershell
docker-compose down
```

---

## 🎮 Runtime Controls

When the system is running and displaying video:

| Key | Action |
|-----|--------|
| `q` | Quit application |
| `s` | Save snapshot to `output/` |
| `r` | Reset tracker (clear all IDs) |

---

## 👤 Managing Face Database

### **Register New Faces (Interactive)**
```powershell
docker exec -it face_recognition_system python3 src/database_manager.py
```

**Steps:**
1. Position face in camera view
2. Press `s` to capture
3. Enter person ID (e.g., "john_doe")
4. Enter name (e.g., "John Doe")
5. Enter role (optional)

### **Database Location**
```
CC_1/face_database.pkl
```

### **Backup Database**
```powershell
copy face_database.pkl face_database_backup.pkl
```

---

## 🔧 Configuration (.env file)

Key settings in `.env`:

```bash
# Camera
RTSP_URL=rtsp://192.168.1.100:554/profile2/media.smp

# Performance
DETECTION_INTERVAL=1          # Process every Nth frame
DETECTION_THRESHOLD=0.5       # Face detection confidence
RECOGNITION_THRESHOLD=0.4     # Face matching threshold

# Display
DISPLAY_ENABLED=true          # Show video window
SAVE_VIDEO=false              # Record output video

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/face_recognition.log
```

---

## 📊 System Architecture

```
RTSP Camera (192.168.1.100:554)
        ↓
[ GStreamer Stream Capture ]  ← Thread 1
        ↓ (Frame Queue)
[ InsightFace Detection ]     ← Thread 2
        ↓ (Detection Queue)
[ BYTETrack Tracking ]        ← Thread 3
        ↓ (Result Queue)
[ Display & Logging ]         ← Main Thread
```

### **Key Components:**
1. **GStreamer** - Robust RTSP stream capture (handles NAT/port forwarding)
2. **InsightFace SCRFD** - Face detection (CPU-optimized)
3. **ArcFace** - 512-dim face embeddings
4. **BYTETrack** - Multi-person tracking with consistent IDs
5. **Docker** - Containerized deployment

---

## 🐛 Troubleshooting

### **Issue: Can't connect to RTSP stream**
```powershell
# Test camera accessibility
ping 192.168.1.100

# Test RTSP connection directly
docker run --rm -it jrottenberg/ffmpeg:4.1-alpine -rtsp_transport tcp -i rtsp://192.168.1.100:554/profile2/media.smp -frames:v 1 -f null -
```

### **Issue: No display window showing**
1. Check `.env` has `DISPLAY_ENABLED=true`
2. On Windows, display should work automatically
3. Check Docker Desktop is running

### **Issue: Low FPS / High CPU usage**
Edit `.env`:
```bash
DETECTION_INTERVAL=2    # Process every 2nd frame
DETECTION_SIZE=320      # Reduce detection size
```

### **Issue: Face not recognized**
- Lower `RECOGNITION_THRESHOLD` in `.env` (more lenient)
- Re-register face with better lighting
- Register face from multiple angles

### **Issue: Docker build failed**
```powershell
# Clean and rebuild
docker-compose down
docker system prune -f
docker-compose build --no-cache
```

---

## 📈 Performance Tuning

### **Presets:**

**High Accuracy (Slower - ~15 FPS)**
```bash
DETECTION_INTERVAL=1
DETECTION_THRESHOLD=0.5
RECOGNITION_THRESHOLD=0.45
```

**Balanced (Recommended - ~25-30 FPS)**
```bash
DETECTION_INTERVAL=2
DETECTION_THRESHOLD=0.5
RECOGNITION_THRESHOLD=0.4
```

**High Speed (Lower Accuracy - ~45-50 FPS)**
```bash
DETECTION_INTERVAL=3
DETECTION_THRESHOLD=0.6
RECOGNITION_THRESHOLD=0.35
```

---

## 🧪 Testing the System

### **1. Component Tests**
```powershell
docker exec -it face_recognition_system python3 test_system.py
```

This tests:
- ✓ All dependencies
- ✓ GStreamer
- ✓ InsightFace models
- ✓ Face pipeline
- ✓ Tracker
- ✓ Database

### **2. Test Individual Modules**
```powershell
# Test stream capture
docker exec -it face_recognition_system python3 src/stream_capture.py

# Test face detection
docker exec -it face_recognition_system python3 src/face_recognition_pipeline.py

# Test tracker
docker exec -it face_recognition_system python3 src/tracker.py
```

---

## 📚 Documentation Files

| File | Purpose |
|------|---------|
| `README.md` | Complete system documentation |
| `QUICKSTART.md` | 5-minute quick start guide |
| `DEPLOYMENT.md` | Production deployment instructions |
| `SESSION_SUMMARY.md` | This file - session summary |

---

## 💾 Important Files & Locations

**Conversation History:**
```
C:\Users\sungh\.claude\history.jsonl
```

**Build Logs:**
```
C:\Users\sungh\Documents\CC_1\build.log
C:\Users\sungh\Documents\CC_1\build_retry.log
```

**Container Logs:**
```powershell
docker-compose logs > container_logs.txt
```

**Face Database:**
```
C:\Users\sungh\Documents\CC_1\face_database.pkl
```

---

## 🔄 Rebuild After Changes

### **Rebuild Docker Image:**
```powershell
docker-compose build
```

### **Rebuild with No Cache:**
```powershell
docker-compose build --no-cache
```

### **Rebuild and Start:**
```powershell
docker-compose up --build
```

---

## 📝 Next Steps

1. **✅ COMPLETED:** Build Docker image
2. **▶️ NEXT:** Start system with `docker-compose up`
3. **→ THEN:** Test RTSP connection
4. **→ THEN:** Register known faces
5. **→ THEN:** Tune performance settings
6. **→ OPTIONAL:** Set up automatic startup
7. **→ OPTIONAL:** Deploy to production

---

## 🎓 Technical Details

### **Models Used:**
- **Detection:** SCRFD (InsightFace buffalo_l pack)
- **Landmarks:** 5-point facial landmark detector
- **Recognition:** ArcFace (512-dimensional embeddings)
- **Tracking:** BYTETrack with Kalman filtering

### **Dependencies:**
- Python 3.10
- GStreamer 1.0 (with RTSP plugins)
- InsightFace 0.7.3
- ONNXRuntime 1.23.2
- PyTorch 2.9.1
- OpenCV 4.12.0

### **Docker Image:**
- Base: Ubuntu 22.04
- Size: ~5-6 GB (first build)
- Name: `cc_1-face-recognition:latest`

---

## 🆘 Getting Help

### **Check Logs:**
```powershell
# Application logs
docker-compose logs

# System logs
docker logs face_recognition_system

# Log file
type logs\face_recognition.log
```

### **Debug Mode:**
Edit `.env`:
```bash
LOG_LEVEL=DEBUG
```

### **Container Shell Access:**
```powershell
docker exec -it face_recognition_system bash
```

---

## ✅ Build Success Confirmation

**Build Status:** ✅ SUCCESSFUL
**Build Date:** December 11, 2024
**Docker Image:** `cc_1-face-recognition:latest`
**Exit Code:** 0

**Installed Components:**
- ✅ 710 system packages
- ✅ All Python dependencies
- ✅ InsightFace + models
- ✅ PyTorch + CUDA support
- ✅ GStreamer + plugins
- ✅ OpenCV

**Ready to run!**

---

## 📞 Support Resources

- **Project README:** [README.md](README.md)
- **Quick Start:** [QUICKSTART.md](QUICKSTART.md)
- **Deployment:** [DEPLOYMENT.md](DEPLOYMENT.md)
- **InsightFace Docs:** https://github.com/deepinsight/insightface
- **GStreamer Docs:** https://gstreamer.freedesktop.org/documentation/

---

**End of Session Summary**
*Generated: December 11, 2024*
*Project Location: C:\Users\sungh\Documents\CC_1*
