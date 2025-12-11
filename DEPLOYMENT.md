# Deployment Guide

## Production Deployment

### Prerequisites

1. **Hardware Requirements**
   - CPU: 8+ cores (16+ threads recommended)
   - RAM: 16GB minimum, 32GB recommended
   - Disk: 20GB for Docker images and data
   - Network: Stable connection to RTSP source

2. **Software Requirements**
   - Docker Engine 20.10+
   - Docker Compose 2.0+
   - (Linux) X11 for display (optional)

### Step-by-Step Deployment

#### 1. Initial Setup

```bash
# Clone repository
git clone <repository-url>
cd CC_1

# Create required directories
mkdir -p data logs output models config

# Copy environment template
cp .env.example .env
```

#### 2. Configuration

Edit `.env` with production settings:

```bash
# RTSP Stream - Replace with your camera URL
RTSP_URL=rtsp://your-camera-ip:554/profile2/media.smp

# Performance tuning
DETECTION_INTERVAL=2          # Process every 2nd frame for better performance
DETECTION_THRESHOLD=0.6       # Higher threshold = fewer false positives
RECOGNITION_THRESHOLD=0.45    # Adjust based on accuracy needs

# Logging
LOG_LEVEL=INFO
LOG_FILE=/app/logs/face_recognition.log

# Disable display in headless mode
DISPLAY_ENABLED=false
```

#### 3. Build Docker Image

```bash
# Linux
./scripts/build.sh

# Windows
scripts\build.bat

# Or manually
docker-compose build
```

#### 4. Register Known Faces

Before running in production, populate the face database:

```bash
# Start interactive registration
docker-compose run face-recognition python3 src/database_manager.py

# Or prepare faces offline and import
docker-compose run face-recognition python3 -c "
from database_manager import FaceDatabaseManager
import numpy as np

db = FaceDatabaseManager()
# Add faces programmatically
db.save()
"
```

#### 5. Test Run

```bash
# Test with display (if available)
docker-compose up

# Test headless mode
docker-compose run -e DISPLAY_ENABLED=false face-recognition

# Check logs
docker-compose logs -f
```

#### 6. Production Run

```bash
# Start in background (detached mode)
docker-compose up -d

# View logs
docker-compose logs -f face-recognition

# Check status
docker-compose ps

# Stop
docker-compose down
```

### Linux Systemd Service

Create `/etc/systemd/system/face-recognition.service`:

```ini
[Unit]
Description=Face Recognition System
Requires=docker.service
After=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/path/to/CC_1
ExecStart=/usr/bin/docker-compose up -d
ExecStop=/usr/bin/docker-compose down
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl daemon-reload
sudo systemctl enable face-recognition
sudo systemctl start face-recognition
sudo systemctl status face-recognition
```

### Windows Service

Use [NSSM (Non-Sucking Service Manager)](https://nssm.cc/):

```cmd
# Install NSSM
choco install nssm

# Create service
nssm install FaceRecognition "C:\Program Files\Docker\Docker\resources\bin\docker-compose.exe"
nssm set FaceRecognition AppDirectory "C:\Users\user\CC_1"
nssm set FaceRecognition AppParameters "up"
nssm set FaceRecognition AppStdout "C:\Users\user\CC_1\logs\service.log"
nssm set FaceRecognition AppStderr "C:\Users\user\CC_1\logs\service_error.log"

# Start service
nssm start FaceRecognition
```

## Monitoring

### Health Checks

```bash
# Check container health
docker inspect --format='{{.State.Health.Status}}' face_recognition_system

# View health check logs
docker inspect face_recognition_system | jq '.[0].State.Health'
```

### Log Monitoring

```bash
# Real-time logs
docker-compose logs -f

# Last 100 lines
docker-compose logs --tail=100

# Save logs to file
docker-compose logs > logs/deployment_$(date +%Y%m%d).log
```

### Performance Monitoring

```bash
# Container resource usage
docker stats face_recognition_system

# System resource usage
htop  # Linux
taskmgr  # Windows
```

### Application Metrics

Check logs for:
- FPS (frames per second)
- Frames processed
- Faces detected
- Active tracks
- Database size

Example log line:
```
Processed 1000 frames | FPS: 18.5 | Active tracks: 3
```

## Scaling

### Horizontal Scaling (Multiple Cameras)

Edit `docker-compose.yml`:

```yaml
services:
  camera-1:
    build: .
    environment:
      - RTSP_URL=rtsp://camera1-ip:554/stream
    volumes:
      - ./data/camera1:/app/data
    container_name: face_rec_camera1

  camera-2:
    build: .
    environment:
      - RTSP_URL=rtsp://camera2-ip:554/stream
    volumes:
      - ./data/camera2:/app/data
    container_name: face_rec_camera2
```

### Vertical Scaling (Performance Tuning)

1. **Increase Detection Interval**
   ```bash
   DETECTION_INTERVAL=3  # Process every 3rd frame
   ```

2. **Reduce Detection Size**
   ```bash
   DETECTION_SIZE=320    # Smaller = faster but less accurate
   ```

3. **CPU Allocation**
   ```yaml
   # docker-compose.yml
   deploy:
     resources:
       limits:
         cpus: '20'      # Use more CPU cores
   ```

4. **Memory Allocation**
   ```yaml
   deploy:
     resources:
       limits:
         memory: 48G    # Increase memory limit
   ```

## Backup and Recovery

### Database Backup

```bash
# Backup face database
docker cp face_recognition_system:/app/face_database.pkl ./backups/face_db_$(date +%Y%m%d).pkl

# Automated backup (cron)
0 2 * * * docker cp face_recognition_system:/app/face_database.pkl /backups/face_db_$(date +\%Y\%m\%d).pkl
```

### Configuration Backup

```bash
# Backup configuration
tar -czf config_backup_$(date +%Y%m%d).tar.gz .env docker-compose.yml config/
```

### Full System Backup

```bash
# Backup everything
tar -czf full_backup_$(date +%Y%m%d).tar.gz \
  .env docker-compose.yml \
  data/ config/ \
  face_database.pkl
```

### Recovery

```bash
# Restore database
docker cp backups/face_db_20231115.pkl face_recognition_system:/app/face_database.pkl

# Restart container
docker-compose restart
```

## Troubleshooting

### Container Won't Start

```bash
# Check logs
docker-compose logs

# Check Docker status
docker info

# Rebuild image
docker-compose build --no-cache
```

### Stream Connection Issues

```bash
# Test RTSP directly
ffmpeg -rtsp_transport tcp -i $RTSP_URL -frames:v 1 test.jpg

# Check network
ping camera-ip
telnet camera-ip 554
```

### High CPU Usage

- Increase `DETECTION_INTERVAL`
- Reduce `DETECTION_SIZE`
- Check for resource limits

### Memory Issues

```bash
# Check memory usage
docker stats

# Increase memory limit in docker-compose.yml
```

### Display Issues (Linux)

```bash
# Enable X11 forwarding
xhost +local:docker

# Set DISPLAY variable
export DISPLAY=:0
```

## Security Considerations

1. **Network Security**
   - Use VPN for remote RTSP access
   - Implement firewall rules
   - Use RTSP over TLS if supported

2. **Container Security**
   - Run as non-root user (update Dockerfile)
   - Use read-only root filesystem
   - Limit capabilities

3. **Data Security**
   - Encrypt face database at rest
   - Secure backup storage
   - Implement access controls

4. **GDPR Compliance**
   - Document data retention policies
   - Implement data deletion mechanisms
   - Maintain audit logs

## Performance Benchmarks

### Test System: i7-13700 (20 threads), 64GB RAM

| Configuration | FPS | CPU Usage | Memory |
|--------------|-----|-----------|---------|
| High Accuracy (interval=1, size=640) | 15-18 | 60-70% | 4-6GB |
| Balanced (interval=2, size=640) | 25-30 | 40-50% | 3-5GB |
| High Speed (interval=3, size=320) | 45-50 | 25-35% | 2-4GB |

### Optimization Tips

- **For Accuracy**: Use interval=1, size=640, threshold=0.5
- **For Speed**: Use interval=3, size=320, threshold=0.6
- **For Balance**: Use interval=2, size=640, threshold=0.5

## Updates and Maintenance

### Updating the System

```bash
# Pull latest code
git pull

# Rebuild image
docker-compose build

# Restart with new image
docker-compose down
docker-compose up -d
```

### Updating Dependencies

```bash
# Edit requirements.txt with new versions
# Rebuild image
docker-compose build --no-cache
```

### Model Updates

```bash
# Download new InsightFace models
docker exec -it face_recognition_system bash
cd /app/models
# Download and configure new models
```

## Support

For issues and questions:
- Check logs: `docker-compose logs -f`
- Review documentation: README.md
- Open GitHub issue with logs and configuration

---

**Version**: 1.0.0
**Last Updated**: 2024
