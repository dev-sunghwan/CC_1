"""
InsightFace-based Face Recognition Pipeline
Handles detection, landmark extraction, alignment, and embedding generation
"""

import numpy as np
import cv2
import logging
from typing import List, Tuple, Optional, Dict
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align
import pickle
import json
import shutil
import time
import glob
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceRecognitionPipeline:
    """
    Complete face recognition pipeline using InsightFace
    - Detection: SCRFD
    - Landmarks: 5-point detection
    - Alignment: Face normalization
    - Embedding: ArcFace ONNX model
    """

    def __init__(self,
                 detection_size: Tuple[int, int] = (640, 640),
                 det_thresh: float = 0.5,
                 ctx_id: int = -1):  # -1 for CPU, 0 for GPU
        """
        Initialize face recognition pipeline

        Args:
            detection_size: Input size for detection model
            det_thresh: Detection confidence threshold
            ctx_id: Device ID (-1 for CPU)
        """
        self.detection_size = detection_size
        self.det_thresh = det_thresh

        # Initialize InsightFace app
        logger.info("Initializing InsightFace models...")
        self.app = FaceAnalysis(
            name='buffalo_l',  # High-accuracy model pack
            providers=['CPUExecutionProvider'] if ctx_id == -1 else ['CUDAExecutionProvider']
        )
        self.app.prepare(ctx_id=ctx_id, det_size=detection_size, det_thresh=det_thresh)
        logger.info("InsightFace initialized successfully")

        # Embedding database
        self.face_database = {}
        self.load_database()

    def detect_and_extract(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces and extract features

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of face dictionaries with bbox, landmarks, embedding, etc.
        """
        if frame is None:
            logger.warning("detect_and_extract: frame is None")
            return []

        # Log frame info
        logger.debug(f"Running detection on frame: {frame.shape}")

        # Run face analysis
        faces = self.app.get(frame)

        logger.info(f"InsightFace detected {len(faces)} faces")
        if len(faces) > 0:
            for i, face in enumerate(faces):
                logger.info(f"  Face {i+1}: bbox={face.bbox.astype(int).tolist()}, score={face.det_score:.3f}")

        # Convert to standardized format
        results = []
        for face in faces:
            face_dict = {
                'bbox': face.bbox.astype(int).tolist(),  # [x1, y1, x2, y2]
                'landmarks': face.kps.astype(int).tolist(),  # 5-point landmarks
                'det_score': float(face.det_score),  # Detection confidence
                'embedding': face.normed_embedding.tolist(),  # 512-dim ArcFace embedding
                'gender': face.gender,  # 0: female, 1: male
                'age': int(face.age),
            }
            results.append(face_dict)

        return results

    def recognize_faces(self, faces: List[Dict], threshold: float = 0.4) -> List[Dict]:
        """
        Match detected faces against database

        Args:
            faces: List of detected faces with embeddings
            threshold: Cosine similarity threshold for recognition

        Returns:
            Faces with identity information added
        """
        for face in faces:
            embedding = np.array(face['embedding'])

            best_match = None
            best_similarity = -1.0

            # Compare with all known faces
            for person_id, person_data in self.face_database.items():
                db_embedding = np.array(person_data['embedding'])

                # Cosine similarity
                similarity = np.dot(embedding, db_embedding)

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_id

            # Add recognition result
            if best_similarity >= threshold:
                face['identity'] = best_match
                face['similarity'] = float(best_similarity)
            else:
                face['identity'] = 'Unknown'
                face['similarity'] = float(best_similarity) if best_match else 0.0

        return faces

    def add_face_to_database(self, person_id: str, embedding: np.ndarray,
                            metadata: Optional[Dict] = None):
        """
        Add a face to the recognition database

        Args:
            person_id: Unique identifier for the person
            embedding: Face embedding vector
            metadata: Additional metadata (name, role, etc.)
        """
        self.face_database[person_id] = {
            'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            'metadata': metadata or {}
        }
        logger.info(f"Added {person_id} to face database")

    def save_database(self, filepath: str = "/app/data/face_database.pkl", backup: bool = True):
        """
        Save face database to disk with automatic backup

        Args:
            filepath: Path to save database
            backup: Whether to create backup of existing database
        """
        # Create backup of existing database
        if backup and Path(filepath).exists():
            backup_path = filepath.replace('.pkl', f'_backup_{int(time.time())}.pkl')
            try:
                shutil.copy(filepath, backup_path)
                logger.info(f"Created backup: {backup_path}")

                # Cleanup old backups (keep last 5)
                self._cleanup_old_backups(filepath, max_backups=5)
            except Exception as e:
                logger.warning(f"Failed to create backup: {e}")

        # Save database
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.face_database, f)
            logger.info(f"Face database saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            raise

    def _cleanup_old_backups(self, filepath: str, max_backups: int = 5):
        """
        Cleanup old backup files

        Args:
            filepath: Original database filepath
            max_backups: Maximum number of backups to keep
        """
        backup_pattern = filepath.replace('.pkl', '_backup_*.pkl')
        backups = sorted(glob.glob(backup_pattern))

        # Remove oldest backups
        while len(backups) > max_backups:
            oldest = backups.pop(0)
            try:
                Path(oldest).unlink()
                logger.debug(f"Removed old backup: {oldest}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {oldest}: {e}")

    def load_database(self, filepath: str = "/app/data/face_database.pkl"):
        """
        Load face database from disk

        Args:
            filepath: Path to database file
        """
        if Path(filepath).exists():
            with open(filepath, 'rb') as f:
                self.face_database = pickle.load(f)
            logger.info(f"Loaded {len(self.face_database)} faces from {filepath}")
        else:
            logger.info("No existing database found, starting fresh")
            self.face_database = {}

    def export_database_json(self, filepath: str = "face_database.json"):
        """
        Export database to JSON format

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.face_database, f, indent=2)
        logger.info(f"Database exported to {filepath}")

    def get_aligned_face(self, frame: np.ndarray, landmarks: np.ndarray,
                        output_size: Tuple[int, int] = (112, 112)) -> np.ndarray:
        """
        Extract aligned face crop

        Args:
            frame: Original frame
            landmarks: 5-point facial landmarks
            output_size: Size of aligned face output

        Returns:
            Aligned face image
        """
        aligned_face = face_align.norm_crop(frame, landmarks, image_size=output_size[0])
        return aligned_face

    def draw_results(self, frame: np.ndarray, faces: List[Dict],
                    show_landmarks: bool = True) -> np.ndarray:
        """
        Draw detection and recognition results on frame

        Args:
            frame: Input frame
            faces: List of detected faces
            show_landmarks: Whether to draw landmarks

        Returns:
            Annotated frame
        """
        annotated = frame.copy()

        for face in faces:
            # Draw bounding box
            bbox = face['bbox']
            x1, y1, x2, y2 = bbox

            # Color based on identity
            if face.get('identity') == 'Unknown':
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for known

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Draw landmarks if requested
            if show_landmarks and 'landmarks' in face:
                for lm in face['landmarks']:
                    cv2.circle(annotated, tuple(lm), 2, (255, 255, 0), -1)

            # Draw identity and confidence with improved visibility
            identity = face.get('identity', 'Unknown')
            similarity = face.get('similarity', 0.0)
            det_score = face.get('det_score', 0.0)

            # Main label (name and confidence)
            label = f"{identity} ({similarity:.2f})"

            # Calculate text size for proper background sizing
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw semi-transparent white background for black text
            label_bg_y1 = max(y1 - text_height - 15, 0)
            label_bg_y2 = y1 - 2

            # Create overlay for semi-transparency
            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, label_bg_y1), (x1 + text_width + 10, label_bg_y2), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.8, annotated, 0.2, 0, annotated)

            # Draw main label with outline for extra visibility
            text_x = x1 + 5
            text_y = y1 - 8

            # Draw pure black text
            cv2.putText(annotated, label, (text_x, text_y),
                       font, font_scale, (0, 0, 0), thickness)

            # Draw additional info below bbox with background
            info = f"Age: {face.get('age', 'N/A')} | Gender: {'M' if face.get('gender') == 1 else 'F'}"
            info_font_scale = 0.5
            info_thickness = 1
            (info_width, info_height), info_baseline = cv2.getTextSize(info, font, info_font_scale, info_thickness)

            # Background for info text
            info_y = y2 + 5
            overlay2 = annotated.copy()
            cv2.rectangle(overlay2, (x1, info_y), (x1 + info_width + 10, info_y + info_height + 10), (255, 255, 255), -1)
            cv2.addWeighted(overlay2, 0.8, annotated, 0.2, 0, annotated)

            # Draw info text - pure black
            info_text_y = info_y + info_height + 3
            cv2.putText(annotated, info, (x1 + 5, info_text_y),
                       font, info_font_scale, (0, 0, 0), info_thickness)

        return annotated


if __name__ == "__main__":
    # Test the face recognition pipeline
    print("Testing Face Recognition Pipeline...")

    # Initialize pipeline
    pipeline = FaceRecognitionPipeline(det_thresh=0.5)

    # Test with webcam or image
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Cannot open webcam, trying test image...")
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    else:
        print("Press 'q' to quit, 's' to save current face to database")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect and recognize faces
            faces = pipeline.detect_and_extract(frame)
            faces = pipeline.recognize_faces(faces)

            # Draw results
            annotated = pipeline.draw_results(frame, faces)

            # Show info
            info_text = f"Faces detected: {len(faces)} | Database size: {len(pipeline.face_database)}"
            cv2.putText(annotated, info_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Face Recognition Test", annotated)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and len(faces) > 0:
                # Save first detected face
                person_id = input("Enter person ID: ")
                pipeline.add_face_to_database(person_id, np.array(faces[0]['embedding']))
                pipeline.save_database()

        cap.release()

    cv2.destroyAllWindows()
    print("Test complete")
