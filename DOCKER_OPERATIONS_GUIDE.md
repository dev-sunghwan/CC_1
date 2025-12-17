# Docker Operations Guide

Complete guide for managing the Face Recognition System with Docker.

---

## Prerequisites

1. **Docker Desktop** must be installed and running
2. **PowerShell** or Command Prompt
3. Navigate to the project directory:
   ```bash
   cd C:\Users\sungh\Documents\CC_1
   ```

---

## Quick Reference

| Task | Command |
|------|---------|
| Start (background) | `docker-compose up -d` |
| Start (with logs) | `docker-compose up` |
| Stop | `docker-compose down` |
| Restart | `docker-compose restart face-recognition` |
| View logs | `docker-compose logs -f face-recognition` |
| Check status | `docker ps` |
| Build image | `docker-compose build` |

---

## Starting the System

### Option 1: Background Mode (Recommended)

Start the system in detached mode (runs in background):

```bash
docker-compose up -d
```

**What happens:**
- Container starts in background
- You can close PowerShell window
- System keeps running
- Access web viewer at http://localhost:8080

**When to use:**
- Normal daily operation
- When you don't need to see logs
- For long-running sessions

### Option 2: Foreground Mode (With Live Logs)

Start with live log output in terminal:

```bash
docker-compose up
```

**What happens:**
- Shows real-time logs in terminal
- Press `Ctrl+C` to stop the system
- Good for debugging

**When to use:**
- Troubleshooting issues
- Watching real-time activity
- Testing changes

---

## Stopping the System

### Complete Shutdown

Stop and remove all containers:

```bash
docker-compose down
```

**What this does:**
- Stops the running container
- Removes the container
- Removes the network
- **Keeps all data** (face database, logs, etc.)

### Keep Container Running (Stop Logs Only)

If running in foreground mode, press `Ctrl+C` to stop viewing logs while keeping container running.

---

## Restarting the System

### When Web Viewer Freezes

Quick restart without full shutdown:

```bash
docker-compose restart face-recognition
```

**What this does:**
- Stops the container gracefully
- Starts it again immediately
- **Faster than full stop/start**
- Reconnects to RTSP stream

**When to use:**
- Web viewer is frozen
- RTSP stream timeout
- Quick recovery needed

### Full Restart (Rebuild)

If you made code changes or need a clean start:

```bash
docker-compose down
docker-compose build
docker-compose up -d
```

---

## Viewing Logs

### Live Logs (Follow Mode)

View real-time logs:

```bash
docker-compose logs -f face-recognition
```

Press `Ctrl+C` to stop viewing (container keeps running).

### Recent Logs (Last 50 Lines)

```bash
docker-compose logs --tail 50 face-recognition
```

### Search Logs for Errors

```bash
docker-compose logs face-recognition | grep -i error
```

### Search for Stream Issues

```bash
docker-compose logs face-recognition | grep -i "stream\|timeout"
```

---

## Checking System Status

### Container Status

```bash
docker ps
```

**Output example:**
```
NAMES                     STATUS                  PORTS
face_recognition_system   Up 2 hours (healthy)    0.0.0.0:8080->8080/tcp
```

**Status meanings:**
- `Up X hours (healthy)` - System running normally
- `Up X hours (unhealthy)` - System running but health check failing
- Not listed - Container is stopped

### Detailed Status

```bash
docker ps -a
```

Shows all containers including stopped ones.

### System Resources

```bash
docker stats face_recognition_system
```

Shows CPU, memory, and network usage in real-time.

---

## Building the Docker Image

### Normal Build

```bash
docker-compose build
```

### Clean Build (No Cache)

Force rebuild without using cached layers:

```bash
docker-compose build --no-cache
```

**When to use:**
- After major code changes
- When build seems broken
- To ensure fresh installation

---

## Common Operations

### Start Fresh (Clean Slate)

Complete clean restart:

```bash
# Stop everything
docker-compose down

# Rebuild image
docker-compose build

# Start in background
docker-compose up -d
```

### Check Web Viewer

After starting, open in browser:
```
http://localhost:8080
```

### Access Container Shell

Get a terminal inside the running container:

```bash
docker exec -it face_recognition_system bash
```

Type `exit` to leave the shell.

---

## Troubleshooting

### Problem: "buffalo_l" Downloading Every Time

**Fixed!** The system now persists InsightFace models in `./insightface_models/`.

On first run, it will download (~500MB). Subsequent runs will use the cached models.

### Problem: Web Viewer Frozen

**Solution:**
```bash
docker-compose restart face-recognition
```

Then refresh browser (Ctrl+F5).

### Problem: "Docker is not running"

**Solution:**
1. Open Docker Desktop
2. Wait for it to start (whale icon in system tray)
3. Try command again

### Problem: Port 8080 Already in Use

**Check what's using the port:**
```bash
netstat -ano | findstr :8080
```

**Solution:**
- Stop the other application using port 8080
- Or change port in `docker-compose.yml`

### Problem: Container Keeps Restarting

**Check logs:**
```bash
docker-compose logs --tail 100 face-recognition
```

Look for error messages at the end.

### Problem: RTSP Stream Timeout

**Symptoms:**
```
Stream timeout triggered after 30000 ms
WARNING:stream_capture:Failed to read frame
```

**Solutions:**
1. Restart container: `docker-compose restart face-recognition`
2. Check camera is accessible: `ping 192.168.1.100`
3. Verify camera settings (no sleep mode)

---

## Data Persistence

The following data is persisted on your host machine:

| Data Type | Container Path | Host Path |
|-----------|---------------|-----------|
| Face Database | `/app/data/` | `./data/` |
| Logs | `/app/logs/` | `./logs/` |
| Output | `/app/output/` | `./output/` |
| Models | `/app/models/` | `./models/` |
| InsightFace Models | `/root/.insightface/` | `./insightface_models/` |
| Source Code | `/app/src/` | `./src/` |

**What this means:**
- Running `docker-compose down` is safe - your data persists
- Face database survives restarts
- Logs are saved on your computer
- buffalo_l models download only once

---

## Advanced Commands

### View Health Status

```bash
curl http://localhost:8080/health
```

### Save Current Frame

```bash
curl http://localhost:8080/snapshot
```

Saves to `./data/snapshot.jpg`

### Clean Up Everything (DANGER)

**WARNING:** This removes all containers, images, and networks.

```bash
docker system prune -a
```

Only use if you want to start completely fresh and don't mind re-downloading everything.

---

## Automation Scripts

### Windows Batch Script (start.bat)

Create a file `start.bat`:
```batch
@echo off
cd C:\Users\sungh\Documents\CC_1
docker-compose up -d
echo Face Recognition System started!
echo Web viewer: http://localhost:8080
pause
```

### Windows Batch Script (stop.bat)

Create a file `stop.bat`:
```batch
@echo off
cd C:\Users\sungh\Documents\CC_1
docker-compose down
echo Face Recognition System stopped!
pause
```

### Windows Batch Script (restart.bat)

Create a file `restart.bat`:
```batch
@echo off
cd C:\Users\sungh\Documents\CC_1
docker-compose restart face-recognition
echo Face Recognition System restarted!
pause
```

---

## Resource Management

### Current Configuration

```yaml
CPU Limits: 16 cores maximum, 8 cores reserved
Memory Limits: 32GB maximum, 16GB reserved
```

### Adjust Resources

Edit `docker-compose.yml` if needed:

```yaml
deploy:
  resources:
    limits:
      cpus: '8'      # Maximum CPU cores
      memory: 16G    # Maximum memory
    reservations:
      cpus: '4'      # Reserved CPU cores
      memory: 8G     # Reserved memory
```

Then restart:
```bash
docker-compose down
docker-compose up -d
```

---

## Best Practices

1. **Use Background Mode** for normal operation (`docker-compose up -d`)
2. **Check logs regularly** to catch issues early
3. **Don't use `docker-compose down`** unless you want to fully stop
4. **Use `docker-compose restart`** for quick recovery from freezes
5. **Monitor RTSP timeouts** in logs - restart if frequent
6. **Backup face database** periodically: Copy `./data/face_database.pkl`

---

## Quick Troubleshooting Checklist

When something goes wrong:

1. ✓ Check if Docker Desktop is running
2. ✓ Check container status: `docker ps`
3. ✓ Check recent logs: `docker-compose logs --tail 50 face-recognition`
4. ✓ Try restart: `docker-compose restart face-recognition`
5. ✓ If still broken: `docker-compose down && docker-compose up -d`
6. ✓ Hard refresh browser: Ctrl+F5

---

## Getting Help

### Check System Status
```bash
docker ps
docker-compose logs --tail 100 face-recognition
```

### Check Health Endpoint
```bash
curl http://localhost:8080/health
```

### Useful Information to Provide
- Container status (`docker ps`)
- Recent logs (last 50-100 lines)
- Error messages
- What you were doing when it broke
