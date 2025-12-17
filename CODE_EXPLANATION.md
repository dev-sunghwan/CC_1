# Face Recognition System - Code Explanation & Study Guide

**Purpose:** Educational deep-dive into major code components
**Target Audience:** Developers learning from this codebase
**Prerequisites:** Python, OpenCV, basic threading knowledge

---

## Table of Contents

1. [System Architecture Overview](#system-architecture-overview)
2. [Main Application (main.py)](#main-application-mainpy)
3. [Stream Capture (stream_capture.py)](#stream-capture-stream_capturepy)
4. [Face Recognition Pipeline (face_recognition_pipeline.py)](#face-recognition-pipeline)
5. [Tracking System (tracker.py)](#tracking-system-trackerpy)
6. [Web Streaming (web_stream.py)](#web-streaming-web_streampy)
7. [Face Registration (register_face.py)](#face-registration-register_facepy)
8. [Key Algorithms Explained](#key-algorithms-explained)
9. [Threading & Synchronization](#threading--synchronization)
10. [Performance Optimization Techniques](#performance-optimization-techniques)

---

## System Architecture Overview

### The Producer-Consumer Pattern

The entire system is built on the **producer-consumer pattern** with queues:

```python
# Producer Thread 1: Stream Capture
while running:
    frame = camera.read()
    frame_queue.put(frame)  # Produce

# Consumer Thread 2: Detection (also a producer for next stage)
while running:
    frame = frame_queue.get()      # Consume
    faces = detect(frame)
    detection_queue.put(faces)     # Produce

# Consumer Thread 3: Tracking
while running:
    faces = detection_queue.get()  # Consume
    tracks = track(faces)
    result_queue.put(tracks)       # Produce
```

**Why this pattern?**
- **Decoupling:** Each thread works independently
- **Buffering:** Queues handle speed mismatches between stages
- **Fault isolation:** One thread's failure doesn't crash others

---

## Main Application (main.py)

### Class: FaceRecognitionSystem

**Location:** `src/main.py:29-453`

**Purpose:** Orchestrates all system components and threads

### Initialization Method

```python
def __init__(self,
             rtsp_url: str,
             detection_interval: int = 1,
             display: bool = True,
             save_video: bool = False,
             output_path: str = "output.mp4",
             web_streaming: bool = True,
             web_port: int = 8080):
```

**Key Parameters Explained:**

- `rtsp_url`: RTSP stream URL from IP camera
  - Format: `rtsp://username:password@ip:port/path`
  - Example: `rtsp://YOUR_USERNAME:YOUR_PASSWORD@192.168.1.100:554/profile2/media.smp`

- `detection_interval`: Process every Nth frame
  - `1` = every frame (highest accuracy, most CPU)
  - `2` = every other frame (balanced)
  - `5` = every 5th frame (fastest, may miss brief appearances)

**Component Initialization:**

```python
# Lines 110-137
def initialize(self):
    # Step 1: Connect to camera
    self.stream_capture = RTSPStreamCapture(self.rtsp_url, queue_size=10)

    # Step 2: Load AI models (buffalo_l: 500MB, takes ~5 seconds)
    self.face_pipeline = FaceRecognitionPipeline(
        detection_size=(640, 640),  # Larger = more accurate but slower
        det_thresh=0.3,             # 30% confidence threshold
        ctx_id=-1                   # -1 = CPU, 0+ = GPU
    )

    # Step 3: Initialize tracker
    self.tracker = BYTETracker(
        det_thresh=0.3,      # Minimum detection confidence
        track_thresh=0.5,    # Minimum tracking confidence
        match_thresh=0.4     # IoU threshold for association
    )

    # Step 4: Start web server
    if self.web_streaming:
        self.web_streamer = WebStreamer(host='0.0.0.0', port=self.web_port)
        self.web_streamer.set_tracker_reference(self.tracker)  # Share tracker state
        self.web_streamer.face_database = self.face_pipeline.face_database
        self.web_streamer.start()
```

**Why this order?**
1. Stream capture first (fail fast if camera unreachable)
2. Models second (expensive load, want to confirm camera works)
3. Tracker third (lightweight, depends on detection format)
4. Web server last (depends on all previous components)

---

### Thread 1: Capture Worker

**Location:** `src/main.py:139-168`

**Purpose:** Continuously read frames from RTSP stream

```python
def _capture_worker(self):
    logger.info("Capture worker started")
    frame_count = 0

    while self.running:
        try:
            # Get frame with 2-second timeout
            frame = self.stream_capture.get_frame(timeout=2.0)

            if frame is None:
                # CRITICAL CODE: Auto-recovery logic (added in Upgrade #5)
                if not self.stream_capture.is_alive():
                    logger.warning("Stream is not alive, waiting for reconnection...")
                    time.sleep(2.0)  # Wait for stream_capture to reconnect
                continue  # Don't break! Keep trying

            frame_count += 1

            # Add to queue (drop oldest if full to prevent memory buildup)
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()  # Drop oldest frame
                self.frame_queue.put((frame_count, frame), block=False)
            except queue.Full:
                pass  # Queue still full, skip this frame

        except Exception as e:
            logger.error(f"Capture worker error: {e}")

    logger.info("Capture worker stopped")
```

**Key Concepts:**

1. **Timeout Handling:**
   ```python
   frame = self.stream_capture.get_frame(timeout=2.0)
   ```
   - Without timeout: Thread hangs forever if camera disconnects
   - With timeout: Can check `self.running` flag and exit gracefully

2. **Queue Management:**
   ```python
   if self.frame_queue.full():
       self.frame_queue.get_nowait()  # Drop oldest
   ```
   - Prevents memory overflow if detection is slower than capture
   - Ensures we process the most recent frames (better for real-time)

3. **Auto-Recovery (Critical Fix):**
   ```python
   if not self.stream_capture.is_alive():
       time.sleep(2.0)
   continue  # ← This is the key! Don't break
   ```
   - Old code used `break` which killed the thread
   - New code waits and retries, allowing stream_capture to reconnect

---

### Thread 2: Detection Worker

**Location:** `src/main.py:170-204`

**Purpose:** Detect and recognize faces in frames

```python
def _detection_worker(self):
    logger.info("Detection worker started")

    while self.running:
        try:
            # Get frame from queue (blocking with 1s timeout)
            frame_count, frame = self.frame_queue.get(timeout=1.0)

            # Skip frames based on detection_interval
            if frame_count % self.detection_interval != 0:
                self.detection_queue.put((frame_count, frame, []))
                continue

            # Detect and recognize faces (expensive operation!)
            faces = self.face_pipeline.detect_and_extract(frame)
            faces = self.face_pipeline.recognize_faces(faces, threshold=0.4)

            # Update statistics
            self.stats['faces_detected'] += len(faces)

            # Add to next queue
            self.detection_queue.put((frame_count, frame, faces), block=False)

        except queue.Empty:
            continue  # No frames available, keep waiting
```

**Performance Insight:**

```python
# Detection interval explained with example
detection_interval = 5  # Process every 5th frame

Frame 1: Skip (1 % 5 = 1) → Pass empty faces [] to tracker
Frame 2: Skip (2 % 5 = 2) → Pass empty faces [] to tracker
Frame 3: Skip (3 % 5 = 3) → Pass empty faces [] to tracker
Frame 4: Skip (4 % 5 = 4) → Pass empty faces [] to tracker
Frame 5: DETECT! (5 % 5 = 0) → Run face detection, pass results

Why still pass frames with empty faces?
→ Tracker needs to see every frame to update predictions
→ Tracker interpolates positions between detections
```

**Two-Step Detection:**

```python
# Step 1: Detect faces and extract features
faces = self.face_pipeline.detect_and_extract(frame)
# Returns: [{'bbox': [x1,y1,x2,y2], 'embedding': [512 floats], 'score': 0.85}]

# Step 2: Recognize identities by comparing embeddings
faces = self.face_pipeline.recognize_faces(faces, threshold=0.4)
# Adds: {'identity': 'SungHwan', 'identity_confidence': 0.76}
```

---

### Thread 3: Tracking Worker

**Location:** `src/main.py:206-237`

**Purpose:** Maintain persistent IDs across frames

```python
def _tracking_worker(self):
    logger.info("Tracking worker started")

    while self.running:
        try:
            # Get detection results
            frame_count, frame, faces = self.detection_queue.get(timeout=1.0)

            # Update tracker (handles both detections and empty frames)
            if len(faces) > 0:
                tracks = self.tracker.update(faces)
            else:
                tracks = self.tracker.update([])  # Still update! Tracker predicts positions

            # Draw results on frame
            annotated = self._draw_results(frame, tracks)

            # Add to result queue
            self.result_queue.put((frame_count, annotated, tracks), block=False)

        except queue.Empty:
            continue
```

**Why update tracker with empty detections?**

```python
# Frame N: Person detected at position (100, 100)
tracks = tracker.update([{'bbox': [100, 100, 150, 200]}])
# Tracker state: Track ID 1 at (100, 100), velocity = (0, 0)

# Frame N+1: No detection (person briefly occluded)
tracks = tracker.update([])  # Empty detection
# Tracker state: Track ID 1 PREDICTED at (102, 101) using velocity
# Track marked as "tentative" (1 frame without detection)

# Frame N+2: Person detected at position (104, 102)
tracks = tracker.update([{'bbox': [104, 102, 154, 202]}])
# Tracker state: Associates with Track ID 1 (close to prediction)
# Track back to "confirmed" status
# Updates velocity estimate
```

This allows tracking through brief occlusions!

---

### Drawing Results

**Location:** `src/main.py:239-354`

**Purpose:** Visualize detections with color-coded overlays

```python
def _draw_results(self, frame: np.ndarray, tracks: list) -> np.ndarray:
    annotated = frame.copy()  # Never modify original frame
    h, w = annotated.shape[:2]

    for track in tracks:
        bbox = track['bbox']
        x1, y1, x2, y2 = bbox
        identity = track['identity']
        confidence = track.get('identity_confidence', 0.0)

        # Color-coded by confidence (Phase 4 enhancement)
        if identity == 'Unknown':
            color = (0, 0, 255)      # Red
        elif confidence >= 0.7:
            color = (0, 255, 0)      # Green (high confidence)
        elif confidence >= 0.5:
            color = (0, 255, 255)    # Yellow (medium)
        else:
            color = (0, 165, 255)    # Orange (low)
```

**Double-Bordered Box Technique:**

```python
# Draw double border for better visibility
cv2.rectangle(annotated, (x1, y1), (x2, y2), border_color, 4)  # Outer dark
cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)         # Inner bright

# Why? Dark outer border creates contrast against both light and dark backgrounds
# Bright inner border shows the actual detection color
```

**Confidence Bar Visualization:**

```python
# Draw confidence bar below face
if identity != 'Unknown' and confidence > 0:
    bar_y = y2 + 8
    bar_height = 8
    bar_max_width = x2 - x1
    bar_width = int(bar_max_width * confidence)  # Width = confidence percentage

    # Background (gray)
    cv2.rectangle(annotated, (x1, bar_y), (x2, bar_y + bar_height), (100,100,100), -1)

    # Foreground (colored by confidence)
    cv2.rectangle(annotated, (x1, bar_y), (x1 + bar_width, bar_y + bar_height), bar_color, -1)

    # Border
    cv2.rectangle(annotated, (x1, bar_y), (x2, bar_y + bar_height), (255,255,255), 1)
```

**Text with Shadow for Readability:**

```python
# Shadow (offset by 1 pixel down-right)
cv2.putText(annotated, label, (text_x + 1, text_y + 1),
            font, font_scale, (128,128,128), thickness)

# Main text (black)
cv2.putText(annotated, label, (text_x, text_y),
            font, font_scale, (0,0,0), thickness)

# Creates 3D effect and ensures readability on any background
```

---

## Stream Capture (stream_capture.py)

### Class: RTSPStreamCapture

**Location:** `src/stream_capture.py`

**Purpose:** Reliable RTSP stream reading with auto-reconnection

### Key Architecture Decision

**Problem:** OpenCV's `VideoCapture.read()` is blocking and has no timeout
**Solution:** Separate thread with queue to add timeout capability

```python
class RTSPStreamCapture:
    def __init__(self, rtsp_url, queue_size=10):
        self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)

    def get_frame(self, timeout=None):
        """Non-blocking frame retrieval with timeout"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None  # Timeout occurred
```

### The Capture Loop

```python
def _capture_loop(self):
    while self.running:
        # Try to connect/reconnect
        if self.cap is None or not self.cap.isOpened():
            self._connect()
            if self.cap is None:
                time.sleep(1)
                continue

        # Read frame
        ret, frame = self.cap.read()

        if not ret:
            logger.warning("Failed to read frame")
            self._handle_read_failure()
            continue

        # Successfully read frame
        self.consecutive_failures = 0
        self.last_frame_time = time.time()

        # Put in queue (drop oldest if full)
        try:
            if self.frame_queue.full():
                self.frame_queue.get_nowait()
            self.frame_queue.put(frame, block=False)
        except queue.Full:
            pass
```

### Connection Strategy

```python
def _connect(self):
    """Connect to RTSP stream with retry logic"""
    logger.info(f"Opening RTSP stream: {self.rtsp_url}")
    logger.info("Using TCP transport for RTSP")  # Critical for stability!

    # GStreamer pipeline with TCP transport
    gst_pipeline = (
        f"rtspsrc location={self.rtsp_url} latency=0 protocols=tcp ! "
        f"rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
        f"video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
    )

    try:
        # Try GStreamer first (better performance)
        self.cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

        if not self.cap.isOpened():
            # Fallback to OpenCV with TCP
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

    except Exception as e:
        logger.error(f"Failed to open stream: {e}")
        self.cap = None
```

**Why TCP instead of UDP?**

| Protocol | Advantages | Disadvantages |
|----------|------------|---------------|
| UDP (default) | Lower latency, less overhead | Packet loss, stream freezes |
| TCP | Reliable, no packet loss | Slightly higher latency |

For local network: TCP is more reliable (chosen for this project)

### Reconnection Logic

```python
def _handle_read_failure(self):
    self.consecutive_failures += 1

    if self.consecutive_failures >= self.max_retries:
        logger.error(f"Too many failures ({self.max_retries}), reconnecting...")

        if self.cap:
            self.cap.release()
            self.cap = None

        self.consecutive_failures = 0

        # Wait before reconnecting
        time.sleep(2)
```

**Exponential Backoff Alternative (not implemented but recommended):**

```python
# Could improve by using exponential backoff
retry_delays = [1, 2, 4, 8, 16, 30, 30, 30, 30, 30]  # seconds
delay = retry_delays[min(attempt, len(retry_delays)-1)]
time.sleep(delay)
```

---

## Face Recognition Pipeline

### Class: FaceRecognitionPipeline

**Location:** `src/face_recognition_pipeline.py`

**Purpose:** Wrapper around InsightFace for detection & recognition

### Model Loading

```python
def __init__(self, detection_size=(640, 640), det_thresh=0.3, ctx_id=-1):
    self.app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
    self.app.prepare(ctx_id=ctx_id, det_size=detection_size, det_thresh=det_thresh)

    # Load face database
    self.face_database = self.load_database()
```

**InsightFace buffalo_l Model Components:**

1. **Detection Model** (`det_10g.onnx`) - Finds faces in image
   - Input: 640x640 RGB image
   - Output: Bounding boxes [x1, y1, x2, y2] + confidence scores

2. **Recognition Model** (`w600k_r50.onnx`) - Extracts face embeddings
   - Input: Aligned face crop (112x112)
   - Output: 512-dimensional embedding vector

3. **Additional Models:**
   - `genderage.onnx` - Gender and age estimation
   - `1k3d68.onnx` - 68 facial landmarks
   - `2d106det.onnx` - 106 facial landmarks

### Detection Pipeline

```python
def detect_and_extract(self, frame):
    """Detect faces and extract embeddings"""

    # Run InsightFace detection
    faces = self.app.get(frame)

    results = []
    for face in faces:
        results.append({
            'bbox': face.bbox.astype(int),        # [x1, y1, x2, y2]
            'score': float(face.det_score),       # Detection confidence
            'embedding': face.embedding,          # 512-dim vector
            'landmarks': face.landmark_2d_106,    # Facial keypoints
            'age': int(face.age),                 # Estimated age
            'gender': 'M' if face.gender == 1 else 'F'
        })

    return results
```

### Recognition Algorithm

```python
def recognize_faces(self, faces, threshold=0.4):
    """Match face embeddings against database"""

    for face in faces:
        embedding = face['embedding']

        max_similarity = 0.0
        best_match = 'Unknown'

        # Compare against all people in database
        for name, stored_embeddings in self.face_database.items():
            # Each person has 3 stored embeddings
            for stored_embedding in stored_embeddings:
                # Cosine similarity = dot product of normalized vectors
                similarity = np.dot(embedding, stored_embedding) / (
                    np.linalg.norm(embedding) * np.linalg.norm(stored_embedding)
                )

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = name

        # Assign identity if above threshold
        if max_similarity >= threshold:
            face['identity'] = best_match
            face['identity_confidence'] = max_similarity
        else:
            face['identity'] = 'Unknown'
            face['identity_confidence'] = 0.0

    return faces
```

**Cosine Similarity Explained:**

```
Vector A = [a1, a2, ..., a512]  (Current face embedding)
Vector B = [b1, b2, ..., b512]  (Database embedding)

Similarity = (A · B) / (||A|| × ||B||)

Where:
- A · B = a1*b1 + a2*b2 + ... + a512*b512 (dot product)
- ||A|| = sqrt(a1² + a2² + ... + a512²)   (magnitude)

Result: Value between -1 and 1
- 1.0  = Identical faces
- 0.8  = Very similar (likely same person)
- 0.5  = Somewhat similar
- 0.0  = Completely different
- -1.0 = Opposite (rarely happens with face embeddings)

Threshold 0.4 = Require 40% similarity to recognize
```

### Multi-Embedding Strategy

**Why 3 embeddings per person?**

```python
# Single embedding approach (OLD):
database = {
    'SungHwan': [embedding1]  # Only one angle/lighting
}

# Problem: What if person's face is at different angle?
# embedding1 might be frontal view
# But current frame might be side view
# Result: Low similarity → Fails to recognize

# Multi-embedding approach (NEW):
database = {
    'SungHwan': [embedding1, embedding2, embedding3]
}

# embedding1 = Frontal view
# embedding2 = Slight left turn
# embedding3 = Slight right turn

# Recognition compares against ALL 3
# Takes the BEST match
# Result: Much more robust!
```

---

## Tracking System (tracker.py)

### Class: BYTETracker

**Location:** `src/tracker.py`

**Purpose:** Multi-object tracking with identity persistence

### The Tracking Problem

**Input:** Detections from each frame (no IDs)
```
Frame 1: [face at (100,100), face at (200,200)]
Frame 2: [face at (105,103), face at (202,198)]
Frame 3: [face at (110,106), face at (205,201)]
```

**Question:** Which face in Frame 2 corresponds to which in Frame 1?

**Output:** Tracks with persistent IDs
```
Frame 1: [Track ID 1 at (100,100), Track ID 2 at (200,200)]
Frame 2: [Track ID 1 at (105,103), Track ID 2 at (202,198)]
Frame 3: [Track ID 1 at (110,106), Track ID 2 at (205,201)]
```

### BYTE Tracker Algorithm

```python
def update(self, detections):
    """Update tracks with new detections"""

    # Step 1: Predict new positions using Kalman filter
    for track in self.tracks:
        track.predict()  # Use velocity to predict next position

    # Step 2: Separate high and low confidence detections
    high_conf_dets = [d for d in detections if d['score'] > 0.5]
    low_conf_dets = [d for d in detections if d['score'] <= 0.5]

    # Step 3: Match high-confidence detections to tracks
    matches, unmatched_tracks, unmatched_dets = self._associate(
        self.tracks, high_conf_dets
    )

    # Update matched tracks
    for track_idx, det_idx in matches:
        self.tracks[track_idx].update(high_conf_dets[det_idx])

    # Step 4: Try to match unmatched tracks with low-confidence detections
    # (BYTE's key innovation: use low-confidence dets for track continuation)
    second_matches, _, _ = self._associate(
        unmatched_tracks, low_conf_dets
    )

    # Step 5: Mark unmatched tracks as lost
    for track in unmatched_tracks:
        track.mark_lost()

    # Step 6: Create new tracks from unmatched high-conf detections
    for det in unmatched_dets:
        new_track = Track(next_id(), det)
        self.tracks.append(new_track)

    # Step 7: Remove dead tracks (lost for too long)
    self.tracks = [t for t in self.tracks if not t.is_dead()]

    return self.tracks
```

### Association: IoU Matching

```python
def _associate(self, tracks, detections):
    """Match tracks to detections using IoU (Intersection over Union)"""

    # Build cost matrix
    cost_matrix = np.zeros((len(tracks), len(detections)))

    for i, track in enumerate(tracks):
        for j, det in enumerate(detections):
            iou = self._calculate_iou(track.bbox, det['bbox'])
            cost_matrix[i, j] = 1 - iou  # Convert to cost (lower is better)

    # Hungarian algorithm for optimal assignment
    from scipy.optimize import linear_sum_assignment
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    matches = []
    for row, col in zip(row_indices, col_indices):
        if cost_matrix[row, col] < (1 - self.match_thresh):  # IoU > threshold
            matches.append((row, col))

    return matches, unmatched_tracks, unmatched_dets
```

**IoU (Intersection over Union) Explained:**

```
Box A: [x1=100, y1=100, x2=200, y2=200]
Box B: [x1=150, y1=150, x2=250, y2=250]

Intersection:
  left   = max(100, 150) = 150
  top    = max(100, 150) = 150
  right  = min(200, 250) = 200
  bottom = min(200, 250) = 200

  width  = 200 - 150 = 50
  height = 200 - 150 = 50
  area   = 50 × 50 = 2,500

Union:
  area_A = (200-100) × (200-100) = 10,000
  area_B = (250-150) × (250-150) = 10,000
  union  = 10,000 + 10,000 - 2,500 = 17,500

IoU = 2,500 / 17,500 = 0.143 (14.3% overlap)

If IoU > 0.4 (match_thresh) → Boxes belong to same object
If IoU < 0.4 → Different objects
```

### Track State Machine

```python
class Track:
    def __init__(self, track_id, detection):
        self.track_id = track_id
        self.state = 'tentative'  # Start as tentative
        self.age = 0              # Frames since creation
        self.hits = 0             # Successful matches
        self.time_since_update = 0

    def update(self, detection):
        """Matched with a detection"""
        self.hits += 1
        self.time_since_update = 0

        if self.hits >= 3:  # Promote to confirmed after 3 hits
            self.state = 'confirmed'

    def mark_lost(self):
        """No match this frame"""
        self.time_since_update += 1

        if self.time_since_update > 30:  # Lost for 30 frames
            self.state = 'deleted'

    def is_dead(self):
        return self.state == 'deleted'
```

**State Transitions:**

```
NEW DETECTION
    ↓
TENTATIVE (hits < 3)
    ↓ (matched 3 times)
CONFIRMED (active tracking)
    ↓ (no match for 1-30 frames)
LOST (still tracking, using prediction)
    ↓ (no match for 30+ frames)
DELETED (removed from tracks)
```

### Kalman Filter for Prediction

**Purpose:** Predict where object will be in next frame

```python
class KalmanFilter:
    def __init__(self):
        # State: [x, y, width, height, vx, vy, vw, vh]
        # (position + velocity)
        self.state = np.zeros(8)

    def predict(self):
        """Predict next position using velocity"""
        # x_next = x_current + vx
        # y_next = y_current + vy
        self.state[0] += self.state[4]  # x += vx
        self.state[1] += self.state[5]  # y += vy
        self.state[2] += self.state[6]  # w += vw
        self.state[3] += self.state[7]  # h += vh

        return self.state[:4]  # Return [x, y, w, h]

    def update(self, measurement):
        """Update state with new measurement"""
        # Calculate velocity from position change
        self.state[4] = measurement[0] - self.state[0]  # vx
        self.state[5] = measurement[1] - self.state[1]  # vy

        # Update position
        self.state[:4] = measurement
```

**Example:**

```
Frame 1: Person at (100, 100)
         Velocity = (0, 0) [first observation]

Frame 2: Person at (105, 103)
         Velocity = (5, 3) [calculated from movement]
         Prediction for Frame 3 = (105+5, 103+3) = (110, 106)

Frame 3: No detection (person briefly occluded)
         Use prediction: (110, 106)
         Mark as "tentative"

Frame 4: Person detected at (112, 108)
         Close to prediction (110, 106) → Associate with same track
         Update velocity = (7, 5)
         Mark back as "confirmed"
```

---

## Web Streaming (web_stream.py)

### Class: WebStreamer

**Location:** `src/web_stream.py`

**Purpose:** Flask server with MJPEG streaming and RESTful API

### MJPEG Streaming

```python
def video_feed():
    """Generator function for MJPEG stream"""
    return Response(
        generate_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )

def generate_frames():
    """Continuously yield JPEG frames"""
    while True:
        with frame_lock:
            if current_frame is None:
                continue

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', current_frame,
                                      [cv2.IMWRITE_JPEG_QUALITY, 85])
            frame_bytes = buffer.tobytes()

        # Yield in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        time.sleep(0.04)  # ~25 FPS
```

**MJPEG Format Explained:**

```http
HTTP/1.1 200 OK
Content-Type: multipart/x-mixed-replace; boundary=frame

--frame
Content-Type: image/jpeg

[JPEG image data for frame 1]
--frame
Content-Type: image/jpeg

[JPEG image data for frame 2]
--frame
Content-Type: image/jpeg

[JPEG image data for frame 3]
...
```

Browser displays each frame as it arrives, creating video effect.

### Heartbeat Monitoring

```python
@app.route('/heartbeat')
def heartbeat():
    """Health check endpoint"""
    with heartbeat_lock:
        last_update = self.last_frame_update

    current_time = time.time()

    if last_update is None:
        staleness = -1
        status = 'initializing'
    else:
        staleness = current_time - last_update

        # Classify health based on staleness
        if staleness > 5.0:
            status = 'stale'      # Dead stream
        elif staleness > 3.0:
            status = 'degraded'   # Slow stream
        else:
            status = 'healthy'    # Normal

    return jsonify({
        'timestamp': current_time,
        'last_frame_update': last_update,
        'staleness_seconds': round(staleness, 2) if staleness >= 0 else None,
        'frame_count': self.frame_count,
        'status': status,
        'uptime': round(current_time - self.start_time, 2)
    })
```

**Frontend Polling:**

```javascript
// Poll every 2.5 seconds
setInterval(async () => {
    const response = await fetch('/heartbeat');
    const data = await response.json();

    if (data.status === 'stale') {
        // Show reconnecting overlay
        statusElement.textContent = 'RECONNECTING';
        statusElement.className = 'status-error';

        // Force reload video stream
        videoImg.src = `/video_feed?t=${Date.now()}`;
    } else if (data.status === 'healthy') {
        statusElement.textContent = 'LIVE';
        statusElement.className = 'status-healthy';
    }
}, 2500);
```

### Thread-Safe Frame Updates

```python
class WebStreamer:
    def __init__(self):
        self.current_frame = None
        self.frame_lock = threading.Lock()  # Protect current_frame
        self.heartbeat_lock = threading.Lock()  # Protect timestamps

    def update_frame(self, frame):
        """Called by main thread to update displayed frame"""
        with self.frame_lock:
            self.current_frame = frame.copy()  # Always copy!
            self.frame_count += 1

        with self.heartbeat_lock:
            self.last_frame_update = time.time()
```

**Why separate locks?**

```python
# If single lock for everything:
with single_lock:
    current_frame = frame.copy()  # Slow operation (1-2ms)
    last_update = time.time()

# Problem: Heartbeat endpoint must wait for frame copy to finish
# Heartbeat should be fast (<20ms), but now takes 1-2ms minimum

# With separate locks:
with frame_lock:
    current_frame = frame.copy()  # 1-2ms, only blocks video_feed

with heartbeat_lock:
    last_update = time.time()     # 0.001ms, very fast

# Heartbeat can check timestamp while frame is being copied
# Result: Better concurrency, faster responses
```

---

## Face Registration (register_face.py)

### Interactive Registration Process

```python
def register_face(name, camera_url, num_captures=3):
    """Register a face with multiple embeddings"""

    # Initialize pipeline
    pipeline = FaceRecognitionPipeline()

    # Load existing database
    database = pipeline.load_database()

    embeddings = []

    for i in range(num_captures):
        print(f"\nCapture {i+1}/{num_captures}")
        print("Instructions:")
        print("  - Look directly at camera")
        if i == 1:
            print("  - Turn slightly left")
        elif i == 2:
            print("  - Turn slightly right")

        input("Press Enter when ready...")

        # Capture frame
        ret, frame = camera.read()

        # Detect face
        faces = pipeline.detect_and_extract(frame)

        if len(faces) == 0:
            print("ERROR: No face detected! Try again.")
            i -= 1  # Retry this capture
            continue

        if len(faces) > 1:
            print("WARNING: Multiple faces detected. Using largest.")
            faces.sort(key=lambda f: (f['bbox'][2]-f['bbox'][0]) *
                                    (f['bbox'][3]-f['bbox'][1]),
                      reverse=True)

        # Store embedding
        embeddings.append(faces[0]['embedding'])

        # Show preview
        draw_bbox(frame, faces[0]['bbox'])
        cv2.imshow("Captured", frame)
        cv2.waitKey(1000)  # Show for 1 second

    # Save to database
    database[name] = embeddings
    save_database(database)

    print(f"\n✅ Successfully registered {name} with {num_captures} embeddings")
```

**Multi-Angle Capture Strategy:**

```
Embedding 1: Frontal view
- Look directly at camera
- Captures standard facial features
- Works best for head-on recognition

Embedding 2: Left profile
- Turn slightly left (~15-30 degrees)
- Captures left side features
- Helps recognition when person turns left

Embedding 3: Right profile
- Turn slightly right (~15-30 degrees)
- Captures right side features
- Helps recognition when person turns right

Result: Robust recognition from multiple angles
```

---

## Key Algorithms Explained

### 1. Cosine Similarity (Face Matching)

**Mathematical Definition:**

```
cos(θ) = (A · B) / (||A|| × ||B||)

Where:
- A, B are embedding vectors
- A · B is dot product
- ||A|| is magnitude (L2 norm)
- θ is angle between vectors
```

**Implementation:**

```python
def cosine_similarity(embedding1, embedding2):
    """Calculate similarity between two face embeddings"""
    # Normalize vectors to unit length
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    # Dot product of normalized vectors
    similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)

    return similarity

# Efficient vectorized version (used in production):
def cosine_similarity_optimized(embedding1, embedding2):
    return np.dot(embedding1, embedding2) / (
        np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
    )
```

**Why cosine similarity instead of Euclidean distance?**

```python
# Euclidean distance (L2 distance):
def euclidean_distance(A, B):
    return np.sqrt(np.sum((A - B) ** 2))

# Problem: Sensitive to magnitude
# If brightness changes, all values scale by constant
# Distance changes even though face is the same!

# Cosine similarity:
# Only measures angle between vectors
# Invariant to magnitude (brightness changes)
# Perfect for face recognition where lighting varies
```

### 2. Hungarian Algorithm (Track Association)

**Problem:** Match N tracks to M detections optimally

**Example:**

```
3 Tracks:     [T1, T2, T3]
3 Detections: [D1, D2, D3]

Cost Matrix (1 - IoU):
        D1    D2    D3
T1    0.2   0.9   0.7
T2    0.8   0.3   0.9
T3    0.6   0.8   0.1

Goal: Find matching that minimizes total cost

Greedy approach might choose:
T1→D1 (0.2) + T2→D2 (0.3) + T3→D3 (0.1) = 0.6

But optimal solution:
T1→D1 (0.2) + T2→D2 (0.3) + T3→D3 (0.1) = 0.6
(Same in this case, but not always!)

Hungarian algorithm guarantees optimal assignment in O(n³) time
```

**Usage in Code:**

```python
from scipy.optimize import linear_sum_assignment

row_indices, col_indices = linear_sum_assignment(cost_matrix)

# Returns:
# row_indices = [0, 1, 2]  # Track indices
# col_indices = [0, 1, 2]  # Detection indices
# Meaning: T0→D0, T1→D1, T2→D2
```

### 3. Kalman Filter (Motion Prediction)

**Concept:** Estimate future position using past motion

**State Vector:**

```
x = [cx, cy, w, h, vx, vy, vw, vh]

Where:
- cx, cy: Center position
- w, h: Width, height
- vx, vy: Velocity in x, y
- vw, vh: Rate of size change
```

**Prediction Step:**

```python
# State transition matrix
F = [[1, 0, 0, 0, 1, 0, 0, 0],  # cx_next = cx + vx
     [0, 1, 0, 0, 0, 1, 0, 0],  # cy_next = cy + vy
     [0, 0, 1, 0, 0, 0, 1, 0],  # w_next  = w + vw
     [0, 0, 0, 1, 0, 0, 0, 1],  # h_next  = h + vh
     [0, 0, 0, 0, 1, 0, 0, 0],  # vx constant
     [0, 0, 0, 0, 0, 1, 0, 0],  # vy constant
     [0, 0, 0, 0, 0, 0, 1, 0],  # vw constant
     [0, 0, 0, 0, 0, 0, 0, 1]]  # vh constant

x_predicted = F @ x_current
```

**Update Step:**

```python
# New measurement: z = [cx_measured, cy_measured, w_measured, h_measured]

# Kalman gain (how much to trust measurement vs prediction)
K = ... (complex calculation based on uncertainty)

# Update state
x_updated = x_predicted + K @ (z - H @ x_predicted)

Where H extracts position from state:
H = [[1, 0, 0, 0, 0, 0, 0, 0],
     [0, 1, 0, 0, 0, 0, 0, 0],
     [0, 0, 1, 0, 0, 0, 0, 0],
     [0, 0, 0, 1, 0, 0, 0, 0]]
```

---

## Threading & Synchronization

### Producer-Consumer Queues

```python
import queue
import threading

# Create thread-safe queue
frame_queue = queue.Queue(maxsize=10)

# Producer thread
def producer():
    while True:
        frame = capture_frame()
        try:
            frame_queue.put(frame, timeout=1.0)
        except queue.Full:
            # Queue full, drop frame
            pass

# Consumer thread
def consumer():
    while True:
        try:
            frame = frame_queue.get(timeout=1.0)
            process_frame(frame)
            frame_queue.task_done()  # Mark as complete
        except queue.Empty:
            # No frames available
            pass
```

**Why `maxsize=10`?**

```
No limit (maxsize=0):
→ Unlimited memory growth if producer faster than consumer
→ Old frames accumulate (high latency)
→ Potential memory overflow

Small limit (maxsize=2):
→ Producer blocks frequently waiting for consumer
→ Wastes producer time
→ Low throughput

Medium limit (maxsize=10):
→ Buffer absorbs short bursts
→ Producer rarely blocks
→ Limited memory usage
→ Reasonable latency (~0.4s at 25 FPS)
```

### Thread-Safe Locks

```python
import threading

class SharedState:
    def __init__(self):
        self.value = 0
        self.lock = threading.Lock()

    def increment(self):
        # WRONG (race condition):
        # temp = self.value
        # temp = temp + 1
        # self.value = temp

        # CORRECT:
        with self.lock:
            self.value += 1

    def get(self):
        with self.lock:
            return self.value
```

**Race Condition Example:**

```
Thread 1:          Thread 2:
read value (0)
                   read value (0)
increment (1)
                   increment (1)
write (1)
                   write (1)

Result: value = 1 (should be 2!)

With lock:
Thread 1:          Thread 2:
acquire lock
read value (0)
increment (1)
write (1)
release lock
                   acquire lock
                   read value (1)
                   increment (2)
                   write (2)
                   release lock

Result: value = 2 (correct!)
```

### Daemon Threads

```python
# Daemon thread: Dies when main thread exits
daemon_thread = threading.Thread(target=worker, daemon=True)
daemon_thread.start()

# Regular thread: Main thread waits for it to finish
regular_thread = threading.Thread(target=worker, daemon=False)
regular_thread.start()
regular_thread.join()  # Wait for completion
```

**Usage in this project:**

```python
# Web server: Daemon thread
# Why? Should stop immediately when main program exits
web_thread = threading.Thread(target=web_server.run, daemon=True)

# Capture worker: Regular thread
# Why? Should finish current frame before stopping
capture_thread = threading.Thread(target=capture_worker, daemon=False)
```

---

## Performance Optimization Techniques

### 1. Frame Skipping

```python
# Process every Nth frame
if frame_count % detection_interval != 0:
    skip_detection()
else:
    run_detection()  # Expensive operation

# Example: detection_interval = 5
# Reduces CPU by 80% with minimal accuracy loss
# (tracker interpolates between detections)
```

### 2. Queue Dropping

```python
# Drop oldest frames if queue full
if frame_queue.full():
    frame_queue.get_nowait()  # Drop oldest
frame_queue.put(new_frame)

# Prevents: Old frames accumulating
# Ensures: Processing most recent frames
# Result: Lower latency, better real-time performance
```

### 3. JPEG Quality Trade-off

```python
# High quality (95%): ~100 KB/frame, slow encoding
cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

# Medium quality (85%): ~40 KB/frame, fast encoding
cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

# Low quality (50%): ~15 KB/frame, very fast, visible artifacts
cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

# Chosen: 85% (good balance for web streaming)
```

### 4. Copy-on-Write

```python
# WRONG: Share frame reference
def share_frame_wrong(frame):
    self.current_frame = frame  # Aliasing!
    # Problem: If caller modifies frame, self.current_frame changes too

# CORRECT: Always copy
def share_frame_correct(frame):
    self.current_frame = frame.copy()
    # Caller can modify their copy without affecting ours
```

### 5. Lock Granularity

```python
# WRONG: Hold lock for entire operation
with lock:
    data = expensive_operation()  # Slow!
    self.shared_data = data

# CORRECT: Minimize lock duration
data = expensive_operation()  # Outside lock (fast)
with lock:
    self.shared_data = data  # Only lock for write
```

---

## Summary

This codebase demonstrates:

1. **Multi-threaded Architecture** - Efficient parallel processing
2. **Producer-Consumer Pattern** - Decoupled components with queues
3. **State Machine Design** - Track lifecycle management
4. **Robust Error Handling** - Auto-recovery from failures
5. **Thread Safety** - Locks and queues prevent race conditions
6. **Modern Web Stack** - Flask + MJPEG + RESTful API
7. **AI Integration** - InsightFace for detection & recognition
8. **Performance Optimization** - Frame skipping, quality trade-offs

**Key Takeaways for Learning:**

- Always copy shared data between threads
- Use separate locks for different concerns
- Design for failure (timeouts, retries, recovery)
- Monitor system health (heartbeat, metrics)
- Trade quality for performance when needed
- Document architectural decisions

**Recommended Study Path:**

1. Start with `main.py` - Understand overall flow
2. Study `stream_capture.py` - Learn threading patterns
3. Explore `face_recognition_pipeline.py` - AI integration
4. Dive into `tracker.py` - Complex state management
5. Analyze `web_stream.py` - Web streaming techniques
6. Review `register_face.py` - User interaction design

---

**Document Version:** 1.0
**Last Updated:** December 16, 2024
**Author:** Claude Sonnet 4.5
**Purpose:** Educational deep-dive for developers
