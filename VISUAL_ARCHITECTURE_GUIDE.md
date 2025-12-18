# Visual Architecture Guide - Face Recognition System

**Visual diagrams and flowcharts for understanding system architecture**

---

## 1. System Component Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FACE RECOGNITION SYSTEM                         │
│                                                                          │
│  ┌────────────────────┐         ┌────────────────────┐                 │
│  │   RTSP Camera      │         │  Web Browser       │                 │
│  │  (Hanwha)          │         │  (User Interface)  │                 │
│  └─────────┬──────────┘         └─────────┬──────────┘                 │
│            │                               │                             │
│            │ RTSP Stream                   │ HTTP                        │
│            │                               │                             │
│  ┌─────────▼──────────────────────────────▼──────────┐                 │
│  │              DOCKER CONTAINER                      │                 │
│  │  ┌────────────────────────────────────────────┐  │                 │
│  │  │         MAIN PROCESS (main.py)             │  │                 │
│  │  │                                            │  │                 │
│  │  │  ┌──────────────────────────────────────┐ │  │                 │
│  │  │  │  Thread 1: Stream Capture           │ │  │                 │
│  │  │  │  - RTSPStreamCapture                │ │  │                 │
│  │  │  │  - Auto-reconnection                │ │  │                 │
│  │  │  │  - Frame queuing                    │ │  │                 │
│  │  │  └────────────┬─────────────────────────┘ │  │                 │
│  │  │               │ Frame Queue (max 10)      │  │                 │
│  │  │  ┌────────────▼─────────────────────────┐ │  │                 │
│  │  │  │  Thread 2: Detection & Recognition  │ │  │                 │
│  │  │  │  - InsightFace (SCRFD + ArcFace)    │ │  │                 │
│  │  │  │  - Face detection                   │ │  │                 │
│  │  │  │  - Embedding generation             │ │  │                 │
│  │  │  │  - Database matching                │ │  │                 │
│  │  │  └────────────┬─────────────────────────┘ │  │                 │
│  │  │               │ Detection Queue           │  │                 │
│  │  │  ┌────────────▼─────────────────────────┐ │  │                 │
│  │  │  │  Thread 3: Tracking                 │ │  │                 │
│  │  │  │  - BYTETracker                      │ │  │                 │
│  │  │  │  - Kalman filtering                 │ │  │                 │
│  │  │  │  - ID management                    │ │  │                 │
│  │  │  │  - Visualization                    │ │  │                 │
│  │  │  └────────────┬─────────────────────────┘ │  │                 │
│  │  │               │ Result Queue              │  │                 │
│  │  │  ┌────────────▼─────────────────────────┐ │  │                 │
│  │  │  │  Main Thread: Display & Web         │ │  │                 │
│  │  │  │  - OpenCV display (optional)        │ │  │                 │
│  │  │  │  - Web streamer update              │ │  │                 │
│  │  │  │  - Statistics logging               │ │  │                 │
│  │  │  └──────────────────────────────────────┘ │  │                 │
│  │  └────────────────────────────────────────────┘  │                 │
│  │                                                    │                 │
│  │  ┌────────────────────────────────────────────┐  │                 │
│  │  │    WEB STREAMER (Flask - Daemon Thread)    │  │                 │
│  │  │  - MJPEG stream server                     │  │                 │
│  │  │  - REST API endpoints                      │  │                 │
│  │  │  - Track statistics                        │  │                 │
│  │  └────────────────────────────────────────────┘  │                 │
│  │                                                    │                 │
│  │  ┌────────────────────────────────────────────┐  │                 │
│  │  │      PERSISTENT STORAGE                    │  │                 │
│  │  │  /app/data/face_database.pkl               │  │                 │
│  │  │  - Multi-embedding storage                 │  │                 │
│  │  │  - Metadata                                │  │                 │
│  │  │  - Auto-backup                             │  │                 │
│  │  └────────────────────────────────────────────┘  │                 │
│  └────────────────────────────────────────────────────┘                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Data Flow Diagram

```
┌───────────┐
│  Camera   │
│  (RTSP)   │
└─────┬─────┘
      │
      │ Raw H.264 Stream
      ▼
┌─────────────────┐
│ Stream Capture  │  ← TCP transport, auto-reconnect
│  (GStreamer)    │
└─────┬───────────┘
      │
      │ BGR Frame (1920×1080×3)
      ▼
┌─────────────────┐
│  Frame Queue    │  ← Max 10 frames, drop oldest if full
│   (Thread-Safe) │
└─────┬───────────┘
      │
      │ Frame (every Nth frame)
      ▼
┌─────────────────────────────────────┐
│   InsightFace Detection              │
│   ┌──────────────────────────────┐  │
│   │ SCRFD (640×640)              │  │ ← Resize + detect
│   └────────┬─────────────────────┘  │
│            │ Bounding boxes          │
│   ┌────────▼─────────────────────┐  │
│   │ Landmark Detection (5-point) │  │
│   └────────┬─────────────────────┘  │
│            │ Landmarks               │
│   ┌────────▼─────────────────────┐  │
│   │ Face Alignment               │  │ ← Affine transform to canonical
│   └────────┬─────────────────────┘  │
│            │ Aligned 112×112 faces   │
│   ┌────────▼─────────────────────┐  │
│   │ ArcFace Embedding            │  │ ← ResNet-50 inference
│   └────────┬─────────────────────┘  │
│            │ 512-dim embeddings      │
└────────────┼─────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  Face Recognition                   │
│  ┌──────────────────────────────┐  │
│  │ For each embedding:          │  │
│  │   For each person in DB:     │  │
│  │     similarity = dot(emb, db)│  │ ← Cosine similarity
│  │   best_match = argmax(sim)   │  │
│  │   if sim > threshold:        │  │
│  │     identity = best_match    │  │
│  │   else:                      │  │
│  │     identity = "Unknown"     │  │
│  └──────────────────────────────┘  │
└────────────┬────────────────────────┘
             │
             │ Faces with identity + similarity
             ▼
┌─────────────────────────────────────┐
│  Detection Queue                    │
└────────────┬────────────────────────┘
             │
             ▼
┌─────────────────────────────────────┐
│  BYTETracker                        │
│  ┌──────────────────────────────┐  │
│  │ 1. Predict track positions   │  │ ← Kalman filter
│  │ 2. Match detections to tracks│  │ ← IoU + embedding
│  │ 3. Update matched tracks     │  │
│  │ 4. Create new tracks         │  │
│  │ 5. Remove stale tracks       │  │
│  └──────────────────────────────┘  │
└────────────┬────────────────────────┘
             │
             │ Tracks with consistent IDs
             ▼
┌─────────────────────────────────────┐
│  Visualization                      │
│  - Draw bounding boxes              │
│  - Display identity labels          │
│  - Confidence bars                  │
│  - Track statistics                 │
└────────────┬────────────────────────┘
             │
             │ Annotated frame
             ▼
┌─────────────────────────────────────┐
│  Result Queue                       │
└────────────┬────────────────────────┘
             │
             ├─────────────┬───────────────┐
             │             │               │
             ▼             ▼               ▼
    ┌────────────┐  ┌──────────┐  ┌──────────────┐
    │ OpenCV     │  │  Web     │  │ Video Writer │
    │ Display    │  │ Streamer │  │ (optional)   │
    └────────────┘  └──────────┘  └──────────────┘
```

---

## 3. BYTETrack Algorithm Flowchart

```
┌─────────────────────────────────────┐
│  Input: Current Frame Detections    │
│  [bbox, score, embedding, identity] │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Step 1: Predict All Tracks         │
│                                     │
│  For each track:                    │
│    track.kalman.predict()           │
│    predicted_bbox = kalman.state    │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Step 2: Separate Detections        │
│                                     │
│  high_conf = [det for det in dets  │
│               if score >= 0.6]      │
│  low_conf = [det for det in dets   │
│              if 0.3 <= score < 0.6] │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Step 3: First Association          │
│  (High Confidence Detections)       │
│                                     │
│  ┌─────────────────────────────┐   │
│  │ Compute Cost Matrix:        │   │
│  │   cost[i,j] = 0.7*IoU       │   │
│  │             + 0.3*emb_sim   │   │
│  └─────────────┬───────────────┘   │
│                │                    │
│  ┌─────────────▼───────────────┐   │
│  │ Hungarian Algorithm         │   │
│  │ (Optimal Assignment)        │   │
│  └─────────────┬───────────────┘   │
│                │                    │
│  ┌─────────────▼───────────────┐   │
│  │ Filter by Threshold         │   │
│  │ (cost >= 0.2)               │   │
│  └─────────────┬───────────────┘   │
│                │                    │
│  ┌─────────────▼───────────────┐   │
│  │ Update Matched Tracks       │   │
│  │   track.update_bbox(...)    │   │
│  │   track.update_identity(...) │   │
│  └─────────────────────────────┘   │
└──────────────┬──────────────────────┘
               │
               │ unmatched_tracks, unmatched_detections
               │
               ▼
┌─────────────────────────────────────┐
│  Step 4: Second Association         │
│  (Low Confidence Detections)        │
│                                     │
│  Match remaining tracks with        │
│  low-confidence detections          │
│  (Same process as Step 3)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Step 5: Create New Tracks          │
│                                     │
│  For each unmatched detection:      │
│    if score >= 0.6:                 │
│      track = Track(det)             │
│      track.state = 'tentative'      │
│      tracks.append(track)           │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Step 6: Update Track States        │
│                                     │
│  For each track:                    │
│    if state == 'tentative':         │
│      if hits >= 3:                  │
│        state = 'confirmed'          │
│    if time_since_update > 30:       │
│      state = 'deleted'              │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Step 7: Remove Deleted Tracks      │
│                                     │
│  tracks = [t for t in tracks        │
│            if t.state != 'deleted']  │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│  Output: Confirmed Tracks           │
│  [{track_id, bbox, identity, ...}]  │
└─────────────────────────────────────┘
```

---

## 4. Face Recognition Pipeline (Detailed)

```
┌───────────────────────────────────────────────────────────────────┐
│                    FACE RECOGNITION PIPELINE                      │
└───────────────────────────────────────────────────────────────────┘

┌─────────────┐
│ Input Frame │  (1920×1080×3 BGR)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: FACE DETECTION                       │
│                         (SCRFD Model)                           │
│                                                                 │
│  Input Processing:                                              │
│  ┌────────────────────────────────────┐                        │
│  │ 1. Resize: 1920×1080 → 640×640    │                        │
│  │    (Maintain aspect ratio)         │                        │
│  │ 2. Normalize: BGR → RGB            │                        │
│  │    pixel = (pixel - 127.5) / 128   │                        │
│  └────────────────────────────────────┘                        │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────────────────────┐                        │
│  │ SCRFD CNN (10G model)              │                        │
│  │ - Backbone: ResNet-like            │                        │
│  │ - FPN (Feature Pyramid Network)    │                        │
│  │ - Multi-scale detection            │                        │
│  └────────────────────────────────────┘                        │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────────────────────┐                        │
│  │ Post-processing:                   │                        │
│  │ - NMS (Non-Maximum Suppression)    │                        │
│  │ - Confidence thresholding (>0.3)   │                        │
│  │ - Scale boxes back to 1920×1080    │                        │
│  └────────────────────────────────────┘                        │
│           │                                                     │
│           ▼                                                     │
│  Output: [                                                      │
│    {bbox: [x1,y1,x2,y2], score: 0.95},                         │
│    {bbox: [x1,y1,x2,y2], score: 0.87},                         │
│    ...                                                          │
│  ]                                                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 2: LANDMARK DETECTION                     │
│                     (5-Point Model)                             │
│                                                                 │
│  For each detected face:                                        │
│  ┌────────────────────────────────────┐                        │
│  │ Extract face region + margin       │                        │
│  │ Resize to 192×192                  │                        │
│  │ Run landmark detection CNN         │                        │
│  └────────────────────────────────────┘                        │
│           │                                                     │
│           ▼                                                     │
│  Output: 5 keypoints                                            │
│    - Left eye: (x1, y1)                                         │
│    - Right eye: (x2, y2)                                        │
│    - Nose tip: (x3, y3)                                         │
│    - Left mouth: (x4, y4)                                       │
│    - Right mouth: (x5, y5)                                      │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 3: FACE ALIGNMENT                       │
│                  (Affine Transformation)                        │
│                                                                 │
│  Canonical Template (112×112):                                  │
│  ┌────────────────────────────────────┐                        │
│  │  Left eye: (38.2946, 51.6963)     │                        │
│  │  Right eye: (73.5318, 51.5014)    │                        │
│  │  Nose: (56.0252, 71.7366)         │                        │
│  │  Left mouth: (41.5493, 92.3655)   │                        │
│  │  Right mouth: (70.7299, 92.2041)  │                        │
│  └────────────────────────────────────┘                        │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────────────────────┐                        │
│  │ Compute Affine Transform Matrix    │                        │
│  │   M = estimate_transform(          │                        │
│  │         src=detected_landmarks,    │                        │
│  │         dst=template_landmarks)    │                        │
│  └────────────────────────────────────┘                        │
│           │                                                     │
│           ▼                                                     │
│  ┌────────────────────────────────────┐                        │
│  │ Apply Transformation                │                        │
│  │   aligned = cv2.warpAffine(        │                        │
│  │                face, M, (112,112)) │                        │
│  └────────────────────────────────────┘                        │
│           │                                                     │
│           ▼                                                     │
│  Output: Aligned face (112×112×3)                               │
│  - Eyes horizontal                                              │
│  - Face centered                                                │
│  - Consistent pose                                              │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 4: EMBEDDING GENERATION                   │
│                       (ArcFace Model)                           │
│                                                                 │
│  Model Architecture:                                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ Input: 112×112×3                                        │  │
│  │   ↓                                                     │  │
│  │ Conv1: 7×7, stride=2 (112→56)                          │  │
│  │   ↓                                                     │  │
│  │ ResNet-50 Blocks:                                       │  │
│  │   - Stage 1: 56×56, 64 channels                        │  │
│  │   - Stage 2: 28×28, 128 channels                       │  │
│  │   - Stage 3: 14×14, 256 channels                       │  │
│  │   - Stage 4: 7×7, 512 channels                         │  │
│  │   ↓                                                     │  │
│  │ Global Average Pooling (7×7→1×1)                       │  │
│  │   ↓                                                     │  │
│  │ Fully Connected: 512→512                               │  │
│  │   ↓                                                     │  │
│  │ L2 Normalization: ||embedding|| = 1                    │  │
│  │   ↓                                                     │  │
│  │ Output: 512-dimensional embedding                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Output: embedding = [e1, e2, ..., e512]                       │
│    where Σ(ei²) = 1 (L2-normalized)                            │
└─────────────────────────────┬───────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  STEP 5: IDENTITY MATCHING                      │
│                   (Cosine Similarity)                           │
│                                                                 │
│  Database Structure:                                            │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ {                                                       │  │
│  │   "person_001": {                                       │  │
│  │     "embeddings": [emb1, emb2, emb3],                  │  │
│  │     "metadata": {"name": "John"}                        │  │
│  │   },                                                    │  │
│  │   "person_002": {...}                                   │  │
│  │ }                                                       │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Matching Process:                                              │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │ best_match = None                                       │  │
│  │ best_similarity = -1.0                                  │  │
│  │                                                         │  │
│  │ for person_id, data in database.items():               │  │
│  │   for db_embedding in data['embeddings']:              │  │
│  │     similarity = np.dot(query_emb, db_emb)             │  │
│  │                                                         │  │
│  │     if similarity > best_similarity:                   │  │
│  │       best_similarity = similarity                     │  │
│  │       best_match = person_id                           │  │
│  │                                                         │  │
│  │ if best_similarity >= threshold:                       │  │
│  │   identity = best_match                                │  │
│  │ else:                                                   │  │
│  │   identity = "Unknown"                                 │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  Output: {                                                      │
│    bbox: [x1,y1,x2,y2],                                         │
│    embedding: [e1,...,e512],                                    │
│    identity: "person_001",                                      │
│    similarity: 0.76                                             │
│  }                                                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Track State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRACK LIFECYCLE                              │
└─────────────────────────────────────────────────────────────────┘

                    ┌────────────────┐
                    │  New Detection │
                    │  (high conf)   │
                    └───────┬────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   TENTATIVE STATE    │
                 │                      │
                 │  Properties:         │
                 │  - hits = 1          │
                 │  - age = 0           │
                 │  - Not displayed     │
                 └──┬────────────────┬──┘
                    │                │
           Match    │                │  No match for 30 frames
           for 3    │                │
           frames   │                │
                    ▼                ▼
         ┌──────────────────┐   ┌────────────┐
         │  CONFIRMED STATE │   │  DELETED   │
         │                  │   └────────────┘
         │  Properties:     │
         │  - hits >= 3     │
         │  - Displayed     │
         │  - Stable ID     │
         └──┬───────────────┘
            │
            │ Continuous matching
            │ (Kalman prediction + detection)
            │
            ▼
    ┌───────────────────────────────────────┐
    │  TRACKING UPDATES                     │
    │                                       │
    │  Every frame:                         │
    │  1. Predict position (Kalman)         │
    │  2. Try to match with new detection   │
    │                                       │
    │  If matched:                          │
    │    - Update bbox                      │
    │    - Update embedding                 │
    │    - Update identity (if available)   │
    │    - hits++                           │
    │    - time_since_update = 0            │
    │                                       │
    │  If not matched:                      │
    │    - Use predicted position           │
    │    - time_since_update++              │
    └───────────────┬───────────────────────┘
                    │
                    │ time_since_update > 30
                    │
                    ▼
         ┌──────────────────────┐
         │  DELETED STATE       │
         │                      │
         │  - Removed from list │
         │  - ID can be reused  │
         └──────────────────────┘


EXAMPLE TIMELINE:

Frame 10: Face detected (score=0.85)
          → Track #5 created (state=tentative, hits=1)

Frame 11: Matched (IoU=0.7, emb_sim=0.6)
          → Track #5 updated (hits=2)

Frame 12: Matched (IoU=0.8, emb_sim=0.7)
          → Track #5 CONFIRMED (hits=3, state=confirmed)
          → Now visible to user

Frame 13-50: Continuously matched
             → Track #5 maintained with stable ID

Frame 51: Person exits frame, no match
          → Track #5 uses predicted position
          → time_since_update = 1

Frame 52-80: No match
             → time_since_update = 2...30

Frame 81: time_since_update > 30
          → Track #5 DELETED
```

---

## 6. Multi-Embedding Database Structure

```
┌─────────────────────────────────────────────────────────────────┐
│              FACE DATABASE (face_database.pkl)                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Person 1: "john_doe"                                           │
│  ┌───────────────────────────────────────────────────────┐     │
│  │ "embeddings": [                                       │     │
│  │   [0.123, -0.456, 0.789, ..., 0.234],  ← Frontal     │     │
│  │   [0.234, -0.567, 0.890, ..., 0.345],  ← Left side   │     │
│  │   [0.345, -0.678, 0.901, ..., 0.456],  ← Right side  │     │
│  │   [0.456, -0.789, 0.012, ..., 0.567],  ← Dim light   │     │
│  │   [0.567, -0.890, 0.123, ..., 0.678]   ← Bright      │     │
│  │ ],                                                    │     │
│  │ "embedding_count": 5,                                │     │
│  │ "metadata": {                                         │     │
│  │   "name": "John Doe",                                 │     │
│  │   "role": "Employee",                                 │     │
│  │   "department": "Engineering"                         │     │
│  │ }                                                     │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
│  Person 2: "mary_smith"                                         │
│  ┌───────────────────────────────────────────────────────┐     │
│  │ "embeddings": [                                       │     │
│  │   [0.678, -0.345, 0.234, ..., 0.789],                │     │
│  │   [0.789, -0.456, 0.345, ..., 0.890],                │     │
│  │   [0.890, -0.567, 0.456, ..., 0.901]                 │     │
│  │ ],                                                    │     │
│  │ "embedding_count": 3,                                │     │
│  │ "metadata": {...}                                     │     │
│  └───────────────────────────────────────────────────────┘     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

MATCHING STRATEGY:

Query Embedding: [q1, q2, ..., q512]

For john_doe:
  sim1 = dot(query, john_embedding_1) = 0.65
  sim2 = dot(query, john_embedding_2) = 0.72  ← Best match
  sim3 = dot(query, john_embedding_3) = 0.68
  sim4 = dot(query, john_embedding_4) = 0.58
  sim5 = dot(query, john_embedding_5) = 0.70

  person_similarity = max(sim1, sim2, ..., sim5) = 0.72

For mary_smith:
  sim1 = dot(query, mary_embedding_1) = 0.25
  sim2 = dot(query, mary_embedding_2) = 0.30
  sim3 = dot(query, mary_embedding_3) = 0.28

  person_similarity = max(sim1, sim2, sim3) = 0.30

Best Match: john_doe (similarity=0.72 > threshold=0.4)
Identity: "john_doe"

BENEFITS:
✓ Robust to pose variations (frontal, side, profile)
✓ Robust to lighting changes (bright, dim, backlight)
✓ Robust to expression changes (neutral, smile, etc.)
✓ Only best match needed, not all matches
```

---

## 7. Threading & Queue Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      THREADING ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────┘

┌────────────────┐
│  MAIN THREAD   │  (Coordination & Display)
└───────┬────────┘
        │
        │ Spawns 3 worker threads
        │
        ├──────────────────────────────────────────────┐
        │                                              │
        ▼                                              │
┌─────────────────┐                                   │
│  THREAD 1       │                                   │
│  Stream Capture │                                   │
├─────────────────┤                                   │
│ while True:     │                                   │
│   frame = cap() │                                   │
│   queue1.put()  │─────┐                            │
└─────────────────┘     │                            │
                        │                            │
                        ▼                            │
               ┌─────────────────┐                   │
               │  FRAME QUEUE    │                   │
               │  (maxsize=10)   │                   │
               │                 │                   │
               │  Thread-safe:   │                   │
               │  queue.Queue()  │                   │
               └────────┬────────┘                   │
                        │                            │
                        ▼                            │
               ┌─────────────────┐                   │
               │  THREAD 2       │                   │
               │  Detection      │                   │
               ├─────────────────┤                   │
               │ while True:     │                   │
               │   frame=queue1()│                   │
               │   faces=detect()│                   │
               │   queue2.put()  │─────┐            │
               └─────────────────┘     │            │
                                       │            │
                                       ▼            │
                              ┌─────────────────┐   │
                              │ DETECTION QUEUE │   │
                              │  (maxsize=10)   │   │
                              └────────┬────────┘   │
                                       │            │
                                       ▼            │
                              ┌─────────────────┐   │
                              │  THREAD 3       │   │
                              │  Tracking       │   │
                              ├─────────────────┤   │
                              │ while True:     │   │
                              │   det=queue2()  │   │
                              │   tracks=track()│   │
                              │   queue3.put()  │───┤
                              └─────────────────┘   │
                                                    │
                                                    ▼
                                           ┌─────────────────┐
                                           │  RESULT QUEUE   │
                                           │  (maxsize=10)   │
                                           └────────┬────────┘
                                                    │
                                                    │
                        ┌───────────────────────────┘
                        │
                        ▼
                ┌────────────────┐
                │  MAIN THREAD   │
                │  Display Loop  │
                ├────────────────┤
                │ while True:    │
                │   result=q3()  │
                │   show(result) │
                │   web.update() │
                └────────────────┘


QUEUE BEHAVIOR:

Full Queue:
  Producer: queue.put(block=False)  → Drop oldest, add new
  Consumer: queue.get(timeout=1.0)  → Wait up to 1 second

Empty Queue:
  Consumer: queue.get(timeout=1.0)  → Return None after 1s
  Producer: queue.put(block=False)  → Always succeeds

Drop Strategy:
  When queue is full, oldest frame is dropped
  Ensures latest data is always processed
  Prevents memory buildup
```

---

## 8. REST API Endpoints

```
┌─────────────────────────────────────────────────────────────────┐
│                      WEB API ENDPOINTS                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Base URL: http://localhost:8080                                │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ GET /                                                  │    │
│  │ ─────────────────────────────────────────────────────  │    │
│  │ Description: Main web interface (HTML)                │    │
│  │ Returns: HTML page with live video feed               │    │
│  │                                                        │    │
│  │ UI Components:                                         │    │
│  │ - Live MJPEG stream                                    │    │
│  │ - Track statistics panel                               │    │
│  │ - Face database list                                   │    │
│  │ - System metrics                                       │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ GET /video_feed                                        │    │
│  │ ─────────────────────────────────────────────────────  │    │
│  │ Description: MJPEG video stream                        │    │
│  │ Content-Type: multipart/x-mixed-replace; boundary=frame│    │
│  │                                                        │    │
│  │ Response Format:                                       │    │
│  │   --frame                                              │    │
│  │   Content-Type: image/jpeg                             │    │
│  │   <JPEG_DATA>                                          │    │
│  │   --frame                                              │    │
│  │   ...                                                  │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ GET /tracks                                            │    │
│  │ ─────────────────────────────────────────────────────  │    │
│  │ Description: Get active tracks                         │    │
│  │ Returns: JSON                                          │    │
│  │                                                        │    │
│  │ {                                                      │    │
│  │   "timestamp": 1702934567.123,                         │    │
│  │   "tracks": [                                          │    │
│  │     {                                                  │    │
│  │       "track_id": 5,                                   │    │
│  │       "bbox": [100, 150, 250, 350],                    │    │
│  │       "identity": "john_doe",                          │    │
│  │       "similarity": 0.76,                              │    │
│  │       "age": 25,                                       │    │
│  │       "hits": 150                                      │    │
│  │     }                                                  │    │
│  │   ]                                                    │    │
│  │ }                                                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ GET /database                                          │    │
│  │ ─────────────────────────────────────────────────────  │    │
│  │ Description: Get face database info                    │    │
│  │ Returns: JSON                                          │    │
│  │                                                        │    │
│  │ {                                                      │    │
│  │   "total_identities": 3,                               │    │
│  │   "identities": {                                      │    │
│  │     "john_doe": {                                      │    │
│  │       "embedding_count": 5,                            │    │
│  │       "metadata": {...}                                │    │
│  │     },                                                 │    │
│  │     ...                                                │    │
│  │   }                                                    │    │
│  │ }                                                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
│  ┌────────────────────────────────────────────────────────┐    │
│  │ GET /camera_info                                       │    │
│  │ ─────────────────────────────────────────────────────  │    │
│  │ Description: Get camera information                    │    │
│  │ Returns: JSON                                          │    │
│  │                                                        │    │
│  │ {                                                      │    │
│  │   "timestamp": 1702934567.123,                         │    │
│  │   "camera": {                                          │    │
│  │     "rtsp_url": "rtsp://...",                          │    │
│  │     "ip_address": "192.168.1.100",                     │    │
│  │     "username": "admin",                               │    │
│  │     "port": "554",                                     │    │
│  │     "path": "/profile2/media.smp",                     │    │
│  │     "manufacturer": "Hanwha"                           │    │
│  │   }                                                    │    │
│  │ }                                                      │    │
│  └────────────────────────────────────────────────────────┘    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. File Organization

```
face_recognition_system/
│
├── src/                          # Source code
│   ├── main.py                   # Main entry point & orchestration
│   ├── stream_capture.py         # RTSP stream handling
│   ├── face_recognition_pipeline.py  # Detection + recognition
│   ├── tracker.py                # BYTETrack implementation
│   ├── web_stream.py             # Flask web server
│   ├── database_manager.py       # Database operations
│   ├── register_face.py          # Face registration tool
│   └── update_embeddings.py      # Batch embedding update
│
├── data/                         # Persistent storage
│   ├── face_database.pkl         # Main database
│   ├── face_database_backup_*.pkl  # Automatic backups
│   └── known_faces/              # Face images (optional)
│
├── config/                       # Configuration
│   ├── config.yaml               # System config
│   └── config.example.yaml       # Template
│
├── models/                       # Model weights (auto-downloaded)
│   └── .insightface/
│       └── models/
│           └── buffalo_l/
│               ├── det_10g.onnx       # Detection
│               ├── w600k_r50.onnx     # Recognition
│               ├── genderage.onnx     # Demographics
│               └── ...
│
├── docs/                         # Documentation
│   ├── LEARNING_GUIDE.md         # This guide
│   ├── HANDS_ON_EXERCISES.md     # Practical exercises
│   ├── CODE_EXPLANATION.md       # Code walkthrough
│   └── DOCKER_OPERATIONS_GUIDE.md
│
├── docker-compose.yml            # Docker setup
├── Dockerfile                    # Container definition
├── requirements.txt              # Python dependencies
├── .env                          # Environment variables (local)
├── .env.example                  # Template
└── .gitignore                    # Git exclusions
```

---

## Quick Reference: Key Parameters

```
┌─────────────────────────────────────────────────────────────────┐
│                      TUNABLE PARAMETERS                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  DETECTION                                                      │
│  ──────────────────────────────────────────────────────────     │
│  detection_size        (640, 640)     ← Larger = more accurate │
│  det_thresh           0.3              ← Lower = more faces     │
│  detection_interval   1                ← Higher = faster FPS    │
│                                                                 │
│  RECOGNITION                                                    │
│  ──────────────────────────────────────────────────────────     │
│  recognition_threshold 0.4             ← Lower = more matches   │
│                                                                 │
│  TRACKING                                                       │
│  ──────────────────────────────────────────────────────────     │
│  track_thresh         0.5              ← Min score to track     │
│  match_thresh         0.4              ← IoU threshold          │
│  max_time_lost        30               ← Frames before delete   │
│  iou_weight          0.7               ← Spatial importance     │
│  embedding_weight     0.3              ← Appearance importance  │
│                                                                 │
│  PERFORMANCE                                                    │
│  ──────────────────────────────────────────────────────────     │
│  queue_size           10               ← Frame buffer size      │
│  ctx_id              -1                ← -1=CPU, 0=GPU          │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

**This visual guide complements the LEARNING_GUIDE.md - use both together for best understanding!**
