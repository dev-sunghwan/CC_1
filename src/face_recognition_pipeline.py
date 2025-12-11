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
            return []

        # Run face analysis
        faces = self.app.get(frame)

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

    def save_database(self, filepath: str = "face_database.pkl"):
        """
        Save face database to disk

        Args:
            filepath: Path to save database
        """
        with open(filepath, 'wb') as f:
            pickle.dump(self.face_database, f)
        logger.info(f"Face database saved to {filepath}")

    def load_database(self, filepath: str = "face_database.pkl"):
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

            # Draw identity and confidence
            identity = face.get('identity', 'Unknown')
            similarity = face.get('similarity', 0.0)
            det_score = face.get('det_score', 0.0)

            label = f"{identity} ({similarity:.2f})"
            label_bg_y1 = max(y1 - 30, 0)
            label_bg_y2 = y1

            cv2.rectangle(annotated, (x1, label_bg_y1), (x2, label_bg_y2), color, -1)
            cv2.putText(annotated, label, (x1 + 5, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw additional info
            info = f"Age: {face.get('age', 'N/A')} | Gender: {'M' if face.get('gender') == 1 else 'F'}"
            cv2.putText(annotated, info, (x1, y2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

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
