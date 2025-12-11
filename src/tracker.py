"""
Multi-Person Tracking Module
Implements BYTETrack algorithm for consistent ID tracking across frames
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import deque
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class KalmanFilter:
    """
    Simple Kalman filter for bounding box tracking
    State: [x_center, y_center, area, ratio, vx, vy, va, vr]
    """

    def __init__(self):
        self.dt = 1.0  # Time step

        # State transition matrix (constant velocity model)
        self.F = np.eye(8)
        for i in range(4):
            self.F[i, i+4] = self.dt

        # Measurement matrix
        self.H = np.eye(4, 8)

        # Process noise
        self.Q = np.eye(8)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4) * 10

        # State and covariance
        self.x = np.zeros((8, 1))
        self.P = np.eye(8) * 1000

    def predict(self):
        """Predict next state"""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].flatten()

    def update(self, measurement):
        """Update with measurement"""
        z = measurement.reshape(4, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P

        return self.x[:4].flatten()

    def get_state(self):
        """Get current state"""
        return self.x[:4].flatten()


class Track:
    """
    Single track representing one person across frames
    """

    _id_counter = 0

    def __init__(self, bbox: np.ndarray, score: float, embedding: Optional[np.ndarray] = None):
        """
        Initialize track

        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            score: Detection confidence
            embedding: Face embedding vector
        """
        self.track_id = Track._id_counter
        Track._id_counter += 1

        self.kalman = KalmanFilter()
        self.update_bbox(bbox, score, embedding)

        self.hits = 1
        self.age = 0
        self.time_since_update = 0

        self.state = 'tentative'  # tentative, confirmed, deleted

        # Identity tracking
        self.embeddings = deque(maxlen=30)  # Store recent embeddings
        if embedding is not None:
            self.embeddings.append(embedding)

        self.identity = 'Unknown'
        self.identity_confidence = 0.0

    def update_bbox(self, bbox: np.ndarray, score: float, embedding: Optional[np.ndarray] = None):
        """Update track with new detection"""
        # Convert bbox to [cx, cy, area, ratio]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        area = w * h
        ratio = w / h if h > 0 else 1.0

        measurement = np.array([cx, cy, area, ratio])

        # Initialize or update Kalman filter
        if self.time_since_update > 0:
            self.kalman.update(measurement)
        else:
            self.kalman.x[:4] = measurement.reshape(4, 1)

        self.bbox = bbox
        self.score = score

        # Update embedding
        if embedding is not None:
            self.embeddings.append(embedding)

        # Update counters
        self.hits += 1
        self.time_since_update = 0

        # Promote to confirmed after 3 hits
        if self.state == 'tentative' and self.hits >= 3:
            self.state = 'confirmed'

    def predict(self):
        """Predict next position"""
        self.age += 1
        self.time_since_update += 1

        # Predict using Kalman filter
        cx, cy, area, ratio = self.kalman.predict()

        # Convert back to bbox
        w = np.sqrt(area * ratio)
        h = area / w if w > 0 else np.sqrt(area)

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        self.predicted_bbox = np.array([x1, y1, x2, y2])

        # Mark for deletion if not updated for too long
        if self.time_since_update > 30:
            self.state = 'deleted'

        return self.predicted_bbox

    def get_average_embedding(self) -> Optional[np.ndarray]:
        """Get average embedding from recent detections"""
        if len(self.embeddings) == 0:
            return None
        return np.mean(self.embeddings, axis=0)

    def update_identity(self, identity: str, confidence: float):
        """Update identity information"""
        self.identity = identity
        self.identity_confidence = confidence


class BYTETracker:
    """
    BYTETrack: Multi-Object Tracking by Associating Every Detection Box
    Optimized for face tracking with embedding-based re-identification
    """

    def __init__(self,
                 det_thresh: float = 0.5,
                 track_thresh: float = 0.6,
                 match_thresh: float = 0.8,
                 max_time_lost: int = 30):
        """
        Initialize BYTETracker

        Args:
            det_thresh: Detection confidence threshold
            track_thresh: Track confirmation threshold
            match_thresh: IoU threshold for matching
            max_time_lost: Maximum frames to keep lost tracks
        """
        self.det_thresh = det_thresh
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.max_time_lost = max_time_lost

        self.tracks: List[Track] = []
        self.frame_id = 0

    def update(self, detections: List[Dict]) -> List[Dict]:
        """
        Update tracks with new detections

        Args:
            detections: List of detection dictionaries

        Returns:
            List of active tracks with IDs
        """
        self.frame_id += 1

        # Predict all tracks
        for track in self.tracks:
            track.predict()

        # Separate high and low confidence detections
        high_det = [d for d in detections if d['det_score'] >= self.track_thresh]
        low_det = [d for d in detections if self.det_thresh <= d['det_score'] < self.track_thresh]

        # First association: high confidence detections with confirmed tracks
        confirmed_tracks = [t for t in self.tracks if t.state == 'confirmed']
        unmatched_tracks, unmatched_det = self._match_detections(confirmed_tracks, high_det)

        # Second association: remaining tracks with low confidence detections
        unmatched_tracks_second, unmatched_det_low = self._match_detections(
            [self.tracks[i] for i in unmatched_tracks], low_det
        )

        # Handle unmatched detections: create new tracks
        for det_idx in unmatched_det:
            det = high_det[det_idx]
            self._create_track(det)

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.state != 'deleted']

        # Return active tracks
        active_tracks = []
        for track in self.tracks:
            if track.state == 'confirmed':
                track_dict = {
                    'track_id': track.track_id,
                    'bbox': track.bbox.astype(int).tolist(),
                    'score': track.score,
                    'identity': track.identity,
                    'identity_confidence': track.identity_confidence,
                    'age': track.age,
                    'hits': track.hits
                }
                active_tracks.append(track_dict)

        return active_tracks

    def _match_detections(self, tracks: List[Track], detections: List[Dict]) -> Tuple[List[int], List[int]]:
        """
        Match detections to tracks using IoU and embeddings

        Returns:
            (unmatched_track_indices, unmatched_detection_indices)
        """
        if len(tracks) == 0 or len(detections) == 0:
            return list(range(len(tracks))), list(range(len(detections)))

        # Compute cost matrix (IoU + embedding similarity)
        cost_matrix = np.zeros((len(tracks), len(detections)))

        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                # IoU cost
                iou = self._compute_iou(track.predicted_bbox, np.array(det['bbox']))

                # Embedding similarity cost
                embedding_sim = 0.0
                if 'embedding' in det and len(track.embeddings) > 0:
                    track_emb = track.get_average_embedding()
                    det_emb = np.array(det['embedding'])
                    embedding_sim = np.dot(track_emb, det_emb)

                # Combined cost (higher is better)
                cost_matrix[i, j] = 0.7 * iou + 0.3 * embedding_sim

        # Hungarian matching (greedy for simplicity)
        matched_tracks = []
        matched_dets = []
        unmatched_tracks = list(range(len(tracks)))
        unmatched_dets = list(range(len(detections)))

        # Greedy matching
        while len(unmatched_tracks) > 0 and len(unmatched_dets) > 0:
            # Find best match
            best_score = -1
            best_track = -1
            best_det = -1

            for i in unmatched_tracks:
                for j in unmatched_dets:
                    if cost_matrix[i, j] > best_score:
                        best_score = cost_matrix[i, j]
                        best_track = i
                        best_det = j

            # Check if match is good enough
            if best_score < self.match_thresh * 0.5:  # Adjust threshold
                break

            # Update track
            tracks[best_track].update_bbox(
                np.array(detections[best_det]['bbox']),
                detections[best_det]['det_score'],
                np.array(detections[best_det].get('embedding'))
            )

            # Update identity if available
            if 'identity' in detections[best_det]:
                tracks[best_track].update_identity(
                    detections[best_det]['identity'],
                    detections[best_det].get('similarity', 0.0)
                )

            matched_tracks.append(best_track)
            matched_dets.append(best_det)
            unmatched_tracks.remove(best_track)
            unmatched_dets.remove(best_det)

        return unmatched_tracks, unmatched_dets

    def _create_track(self, detection: Dict):
        """Create new track from detection"""
        bbox = np.array(detection['bbox'])
        score = detection['det_score']
        embedding = np.array(detection.get('embedding')) if 'embedding' in detection else None

        track = Track(bbox, score, embedding)

        # Set identity if available
        if 'identity' in detection:
            track.update_identity(detection['identity'], detection.get('similarity', 0.0))

        self.tracks.append(track)

    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bounding boxes"""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2

        # Intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)

        inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

        # Union
        bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
        bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = bbox1_area + bbox2_area - inter_area

        if union_area == 0:
            return 0.0

        return inter_area / union_area

    def reset(self):
        """Reset tracker"""
        self.tracks = []
        self.frame_id = 0
        Track._id_counter = 0


if __name__ == "__main__":
    # Test the tracker
    print("Testing BYTETracker...")

    tracker = BYTETracker()

    # Simulate detections over multiple frames
    test_detections = [
        # Frame 1
        [
            {'bbox': [100, 100, 200, 250], 'det_score': 0.9, 'embedding': np.random.randn(512)},
            {'bbox': [300, 150, 400, 300], 'det_score': 0.85, 'embedding': np.random.randn(512)},
        ],
        # Frame 2 (moved slightly)
        [
            {'bbox': [105, 105, 205, 255], 'det_score': 0.88, 'embedding': np.random.randn(512)},
            {'bbox': [305, 155, 405, 305], 'det_score': 0.82, 'embedding': np.random.randn(512)},
        ],
        # Frame 3 (one person missing)
        [
            {'bbox': [110, 110, 210, 260], 'det_score': 0.91, 'embedding': np.random.randn(512)},
        ],
    ]

    for frame_idx, dets in enumerate(test_detections):
        print(f"\n--- Frame {frame_idx + 1} ---")
        tracks = tracker.update(dets)

        print(f"Detections: {len(dets)}")
        print(f"Active tracks: {len(tracks)}")

        for track in tracks:
            print(f"  Track {track['track_id']}: bbox={track['bbox']}, age={track['age']}, hits={track['hits']}")

    print("\nTest complete")
