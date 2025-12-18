# Hands-On Exercises - Face Recognition System

**Practical exercises to deepen your understanding of the face recognition pipeline**

---

## Exercise 1: Understanding Face Embeddings

### Objective
Understand how ArcFace embeddings represent faces in 512-dimensional space.

### Tasks

#### Task 1.1: Extract and Visualize Embeddings

```python
# Create script: exercises/embedding_explorer.py
import numpy as np
from face_recognition_pipeline import FaceRecognitionPipeline
import cv2
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Initialize pipeline
pipeline = FaceRecognitionPipeline()

# Load test images of the same person
images = [
    cv2.imread('test_images/person1_frontal.jpg'),
    cv2.imread('test_images/person1_side.jpg'),
    cv2.imread('test_images/person1_smile.jpg'),
]

# Extract embeddings
embeddings = []
for img in images:
    faces = pipeline.detect_and_extract(img)
    if len(faces) > 0:
        embeddings.append(np.array(faces[0]['embedding']))

# Compute pairwise similarities
print("Pairwise Cosine Similarities:")
for i in range(len(embeddings)):
    for j in range(i+1, len(embeddings)):
        similarity = np.dot(embeddings[i], embeddings[j])
        print(f"Image {i} vs Image {j}: {similarity:.4f}")

# Visualize in 2D using PCA
embeddings_array = np.array(embeddings)
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings_array)

plt.figure(figsize=(8, 6))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
for i, (x, y) in enumerate(embeddings_2d):
    plt.annotate(f'Image {i}', (x, y))
plt.title('Face Embeddings Visualization (PCA)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.savefig('embedding_visualization.png')
print("Visualization saved to embedding_visualization.png")
```

**Expected Outcome:**
- Same person: similarity > 0.6
- Different person: similarity < 0.4
- Embeddings of same person cluster together in 2D space

#### Task 1.2: Embedding Statistics

```python
# Analyze embedding properties
embedding = embeddings[0]

print(f"Embedding dimension: {len(embedding)}")
print(f"L2 norm: {np.linalg.norm(embedding):.4f}")  # Should be ~1.0
print(f"Min value: {embedding.min():.4f}")
print(f"Max value: {embedding.max():.4f}")
print(f"Mean: {embedding.mean():.4f}")
print(f"Std: {embedding.std():.4f}")

# Plot distribution
plt.figure(figsize=(10, 4))
plt.hist(embedding, bins=50)
plt.title('Embedding Value Distribution')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('embedding_distribution.png')
```

**Questions to Answer:**
1. Why is the L2 norm approximately 1.0?
2. What does the distribution shape tell you about the embedding?
3. How does the distribution change for different faces?

---

## Exercise 2: Tracking Algorithm Analysis

### Objective
Understand how BYTETrack maintains consistent IDs across frames.

### Tasks

#### Task 2.1: Track ID Consistency Test

```python
# Create script: exercises/tracking_test.py
import cv2
import time
from face_recognition_pipeline import FaceRecognitionPipeline
from tracker import BYTETracker

pipeline = FaceRecognitionPipeline()
tracker = BYTETracker(det_thresh=0.3, track_thresh=0.5, match_thresh=0.4)

# Capture video
cap = cv2.VideoCapture('test_video.mp4')

track_history = {}  # {track_id: [frame_numbers]}

frame_num = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1

    # Detect and track
    faces = pipeline.detect_and_extract(frame)
    faces = pipeline.recognize_faces(faces, threshold=0.4)
    tracks = tracker.update(faces)

    # Record track appearances
    for track in tracks:
        track_id = track['track_id']
        if track_id not in track_history:
            track_history[track_id] = []
        track_history[track_id].append(frame_num)

    # Visualize
    for track in tracks:
        bbox = track['bbox']
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(frame, f"ID: {track['track_id']}", (bbox[0], bbox[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Tracking Test', frame)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# Analyze track consistency
print("\nTrack Analysis:")
for track_id, frames in track_history.items():
    duration = len(frames)
    gaps = [frames[i+1] - frames[i] - 1 for i in range(len(frames)-1) if frames[i+1] - frames[i] > 1]
    print(f"Track {track_id}:")
    print(f"  Duration: {duration} frames")
    print(f"  First appearance: frame {frames[0]}")
    print(f"  Last appearance: frame {frames[-1]}")
    print(f"  Gaps (missed detections): {len(gaps)} times, total {sum(gaps)} frames")
```

**Questions to Answer:**
1. Which tracks have the longest duration?
2. How many gaps (missed detections) occurred?
3. What causes tracks to be lost?

#### Task 2.2: IoU vs Embedding Weighting Experiment

```python
# Test different cost matrix weights
test_configs = [
    {'iou_weight': 1.0, 'emb_weight': 0.0, 'name': 'IoU only'},
    {'iou_weight': 0.7, 'emb_weight': 0.3, 'name': 'Balanced (default)'},
    {'iou_weight': 0.5, 'emb_weight': 0.5, 'name': 'Equal'},
    {'iou_weight': 0.0, 'emb_weight': 1.0, 'name': 'Embedding only'},
]

for config in test_configs:
    # Modify tracker cost matrix weights (requires editing tracker.py:296)
    # Run tracking on test video
    # Count ID switches (when same person gets new track ID)
    print(f"{config['name']}: {id_switches} ID switches")
```

**Expected Result:**
- IoU only: More ID switches when people cross paths
- Embedding only: More ID switches with occlusion
- Balanced (0.7/0.3): Fewest ID switches overall

---

## Exercise 3: Recognition Threshold Tuning

### Objective
Find optimal recognition threshold for your use case.

### Tasks

#### Task 3.1: ROC Curve Generation

```python
# Create script: exercises/threshold_tuning.py
import numpy as np
import matplotlib.pyplot as plt
from face_recognition_pipeline import FaceRecognitionPipeline
from sklearn.metrics import roc_curve, auc

pipeline = FaceRecognitionPipeline()

# Prepare test dataset
# Format: [(image_path, true_identity), ...]
test_dataset = [
    ('test_images/john_1.jpg', 'john'),
    ('test_images/john_2.jpg', 'john'),
    ('test_images/mary_1.jpg', 'mary'),
    ('test_images/mary_2.jpg', 'mary'),
    ('test_images/unknown_1.jpg', 'unknown'),
    # Add more test images...
]

# Collect similarities
true_labels = []  # 1 if should match, 0 if should not match
similarities = []

for img_path, true_identity in test_dataset:
    img = cv2.imread(img_path)
    faces = pipeline.detect_and_extract(img)

    if len(faces) > 0:
        faces = pipeline.recognize_faces(faces, threshold=0.0)  # Get all similarities
        predicted_identity = faces[0]['identity']
        similarity = faces[0]['similarity']

        # Is this a correct match?
        is_correct = (predicted_identity == true_identity)

        true_labels.append(1 if is_correct else 0)
        similarities.append(similarity)

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(true_labels, similarities)
roc_auc = auc(fpr, tpr)

# Plot
plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve for Face Recognition')
plt.legend()
plt.grid(True)
plt.savefig('roc_curve.png')

# Find optimal threshold (max Youden's J statistic)
j_scores = tpr - fpr
optimal_idx = np.argmax(j_scores)
optimal_threshold = thresholds[optimal_idx]

print(f"Optimal threshold: {optimal_threshold:.3f}")
print(f"At this threshold: TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f}")
```

**Questions to Answer:**
1. What is the optimal threshold for your dataset?
2. How does it compare to the default (0.4)?
3. What is the trade-off between false positives and false negatives?

---

## Exercise 4: Multi-Embedding Database

### Objective
Understand how multi-embedding improves recognition robustness.

### Tasks

#### Task 4.1: Single vs Multi-Embedding Comparison

```python
# Create script: exercises/multi_embedding_test.py
import numpy as np
from face_recognition_pipeline import FaceRecognitionPipeline
import cv2

pipeline = FaceRecognitionPipeline()

# Capture 5 embeddings of same person (different angles)
print("Capture 5 faces of the person...")
embeddings = []
cap = cv2.VideoCapture(0)

while len(embeddings) < 5:
    ret, frame = cap.read()
    faces = pipeline.detect_and_extract(frame)

    if len(faces) > 0:
        cv2.imshow('Capture', frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):  # Press 's' to save
            embeddings.append(faces[0]['embedding'])
            print(f"Captured embedding {len(embeddings)}/5")

cap.release()
cv2.destroyAllWindows()

# Test 1: Single embedding (use first one)
pipeline.face_database = {}
pipeline.add_face_to_database('test_person', embeddings[0])

# Test recognition on different angle
test_frame = cv2.imread('test_images/test_angle.jpg')
faces = pipeline.detect_and_extract(test_frame)
faces = pipeline.recognize_faces(faces, threshold=0.4)

single_emb_similarity = faces[0]['similarity'] if len(faces) > 0 else 0.0
print(f"Single embedding similarity: {single_emb_similarity:.3f}")

# Test 2: Multi-embedding (use all 5)
pipeline.face_database = {}
pipeline.add_face_to_database('test_person', embeddings)

faces = pipeline.detect_and_extract(test_frame)
faces = pipeline.recognize_faces(faces, threshold=0.4)

multi_emb_similarity = faces[0]['similarity'] if len(faces) > 0 else 0.0
print(f"Multi-embedding similarity: {multi_emb_similarity:.3f}")

print(f"\nImprovement: {multi_emb_similarity - single_emb_similarity:.3f}")
```

**Expected Outcome:**
- Multi-embedding should have higher similarity
- More robust to pose/lighting variations

---

## Exercise 5: Performance Profiling

### Objective
Identify performance bottlenecks in the pipeline.

### Tasks

#### Task 5.1: Component Timing Analysis

```python
# Create script: exercises/performance_profiling.py
import time
import numpy as np
from face_recognition_pipeline import FaceRecognitionPipeline
from tracker import BYTETracker
import cv2

pipeline = FaceRecognitionPipeline()
tracker = BYTETracker()

cap = cv2.VideoCapture('test_video.mp4')

timings = {
    'detection': [],
    'recognition': [],
    'tracking': [],
    'visualization': []
}

for _ in range(100):  # Profile 100 frames
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Detection
    t0 = time.time()
    faces = pipeline.detect_and_extract(frame)
    t1 = time.time()
    timings['detection'].append(t1 - t0)

    # 2. Recognition
    t0 = time.time()
    faces = pipeline.recognize_faces(faces)
    t1 = time.time()
    timings['recognition'].append(t1 - t0)

    # 3. Tracking
    t0 = time.time()
    tracks = tracker.update(faces)
    t1 = time.time()
    timings['tracking'].append(t1 - t0)

    # 4. Visualization
    t0 = time.time()
    annotated = pipeline.draw_results(frame, faces)
    t1 = time.time()
    timings['visualization'].append(t1 - t0)

cap.release()

# Print statistics
print("Performance Analysis (100 frames):")
print("-" * 50)
for component, times in timings.items():
    avg_time = np.mean(times) * 1000  # Convert to ms
    std_time = np.std(times) * 1000
    print(f"{component:15s}: {avg_time:6.2f} ± {std_time:5.2f} ms")

total_avg = sum(np.mean(times) for times in timings.values()) * 1000
fps = 1000 / total_avg
print("-" * 50)
print(f"{'Total':15s}: {total_avg:6.2f} ms ({fps:.1f} FPS)")
```

**Questions to Answer:**
1. Which component is the bottleneck?
2. How does database size affect recognition time?
3. What FPS can you achieve on your hardware?

---

## Exercise 6: Custom Tracker Modifications

### Objective
Experiment with tracking algorithm parameters.

### Tasks

#### Task 6.1: Implement Track Re-identification

```python
# Modify tracker.py to add re-ID capability
class BYTETracker:
    def __init__(self, ...):
        # Add gallery for recently deleted tracks
        self.deleted_track_gallery = deque(maxlen=50)

    def update(self, detections):
        # ... existing code ...

        # Before creating new track, check if it matches a recent deletion
        for det_idx in unmatched_det:
            det = high_det[det_idx]
            det_emb = np.array(det.get('embedding'))

            # Search gallery
            best_match_id = None
            best_similarity = 0.0

            for deleted_track in self.deleted_track_gallery:
                deleted_emb = deleted_track.get_average_embedding()
                similarity = np.dot(det_emb, deleted_emb)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_id = deleted_track.track_id

            # Re-use old track ID if high similarity
            if best_similarity > 0.7:
                self._create_track_with_id(det, best_match_id)
            else:
                self._create_track(det)

        # When deleting tracks, add to gallery
        for track in tracks_to_delete:
            if track.state == 'confirmed':
                self.deleted_track_gallery.append(track)
```

**Test Scenario:**
1. Person walks into frame → Track ID = 5
2. Person leaves frame for 50 frames → Track deleted
3. Person returns → Should get Track ID = 5 again (not new ID)

---

## Exercise 7: Database Management

### Objective
Learn to manage and analyze the face database.

### Tasks

#### Task 7.1: Database Statistics and Visualization

```python
# Create script: exercises/database_analysis.py
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load database
with open('/app/data/face_database.pkl', 'rb') as f:
    database = pickle.load(f)

print(f"Total identities: {len(database)}")

# Analyze embedding counts
embedding_counts = []
for person_id, data in database.items():
    count = data.get('embedding_count', 1)
    embedding_counts.append(count)
    print(f"{person_id}: {count} embeddings")

# Plot embedding count distribution
plt.figure(figsize=(10, 4))
plt.bar(range(len(embedding_counts)), embedding_counts)
plt.xlabel('Person ID')
plt.ylabel('Number of Embeddings')
plt.title('Embeddings per Person')
plt.savefig('embedding_counts.png')

# Visualize database in 2D using t-SNE
all_embeddings = []
labels = []
for person_id, data in database.items():
    for emb in data['embeddings']:
        all_embeddings.append(emb)
        labels.append(person_id)

embeddings_array = np.array(all_embeddings)
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings_array)

# Plot
plt.figure(figsize=(12, 8))
for person_id in set(labels):
    mask = [label == person_id for label in labels]
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=person_id)

plt.legend()
plt.title('Face Database Visualization (t-SNE)')
plt.savefig('database_tsne.png')
print("Visualizations saved!")
```

**Questions to Answer:**
1. Which identities have the most embeddings?
2. Are embeddings of same person clustered together in t-SNE plot?
3. Are there any outliers (embeddings far from their cluster)?

---

## Exercise 8: Real-World Scenario Testing

### Objective
Test system robustness in challenging conditions.

### Tasks

#### Task 8.1: Lighting Variation Test

```python
# Test recognition under different lighting
test_conditions = [
    ('bright', 'test_images/person_bright.jpg'),
    ('dim', 'test_images/person_dim.jpg'),
    ('backlight', 'test_images/person_backlight.jpg'),
]

for condition, img_path in test_conditions:
    img = cv2.imread(img_path)
    faces = pipeline.detect_and_extract(img)
    faces = pipeline.recognize_faces(faces)

    if len(faces) > 0:
        identity = faces[0]['identity']
        similarity = faces[0]['similarity']
        print(f"{condition:15s}: {identity:10s} (similarity={similarity:.3f})")
```

#### Task 8.2: Occlusion Handling Test

```python
# Test with partial face occlusion
occlusion_tests = [
    ('no_occlusion', 'person_full_face.jpg'),
    ('sunglasses', 'person_sunglasses.jpg'),
    ('mask', 'person_mask.jpg'),
    ('hand', 'person_hand_covering.jpg'),
]

for test_name, img_path in occlusion_tests:
    # Same testing logic as above
    ...
```

**Questions to Answer:**
1. Which lighting condition performs worst?
2. What types of occlusion cause recognition failure?
3. How can you improve robustness?

---

## Challenge Projects

### Challenge 1: Age/Gender Filter
Implement a feature to only track people of specific age range or gender.

**Hints:**
- Modify `tracker.py` to check age/gender before creating tracks
- Add UI controls to set filters

### Challenge 2: Face Cluster Analysis
Group unknown faces by similarity (unsupervised clustering).

**Hints:**
- Use DBSCAN or K-means on embeddings
- Visualize clusters with t-SNE

### Challenge 3: Attention Heatmap
Generate heatmap showing where people spend most time.

**Hints:**
- Track bounding box centers over time
- Use Gaussian blur to create heatmap
- Overlay on frame

### Challenge 4: Multi-Camera Fusion
Track same person across multiple cameras.

**Hints:**
- Share face database across camera streams
- Use embeddings for re-identification
- Implement global track ID mapping

---

## Debugging Exercises

### Debug 1: Lost Track Investigation

**Scenario:** Track ID keeps changing for same person.

**Investigation Steps:**
1. Print IoU scores between detections and tracks
2. Print embedding similarities
3. Check if match_thresh is too high
4. Visualize predicted vs detected bounding boxes

### Debug 2: False Recognition

**Scenario:** Person A recognized as Person B.

**Investigation Steps:**
1. Print similarity scores for both identities
2. Check if embeddings of A and B are too similar
3. Add more diverse embeddings to database
4. Consider increasing recognition threshold

### Debug 3: Memory Leak

**Scenario:** Memory usage grows over time.

**Investigation Steps:**
1. Profile memory with `memory_profiler`
2. Check if frames are being released properly
3. Verify deque maxlen is set for track embeddings
4. Clear old tracks from gallery periodically

---

## Additional Resources

**Datasets for Testing:**
- LFW (Labeled Faces in the Wild): http://vis-www.cs.umass.edu/lfw/
- CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- MS-Celeb-1M: https://www.microsoft.com/en-us/research/project/ms-celeb-1m/

**Tools:**
- TensorBoard: Visualize embeddings in 3D
- Weights & Biases: Track experiments
- FAISS: Fast similarity search for large databases

**Papers to Read:**
1. ArcFace: https://arxiv.org/abs/1801.07698
2. BYTETrack: https://arxiv.org/abs/2110.06864
3. SCRFD: https://arxiv.org/abs/2105.04714

---

**Happy Learning!**

These exercises will give you hands-on experience with every component of the face recognition system. Work through them at your own pace, and don't hesitate to modify the code to experiment with different approaches.
