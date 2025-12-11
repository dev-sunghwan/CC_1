"""
Face Database Management
Tools for managing the face recognition database
"""

import pickle
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FaceDatabaseManager:
    """
    Manager for face recognition database
    Supports adding, removing, searching, and exporting faces
    """

    def __init__(self, database_path: str = "face_database.pkl"):
        """
        Initialize database manager

        Args:
            database_path: Path to database file
        """
        self.database_path = database_path
        self.database = {}
        self.load()

    def load(self):
        """Load database from disk"""
        if Path(self.database_path).exists():
            try:
                with open(self.database_path, 'rb') as f:
                    self.database = pickle.load(f)
                logger.info(f"Loaded {len(self.database)} identities from {self.database_path}")
            except Exception as e:
                logger.error(f"Failed to load database: {e}")
                self.database = {}
        else:
            logger.info("No existing database found, starting fresh")
            self.database = {}

    def save(self):
        """Save database to disk"""
        try:
            # Create backup
            if Path(self.database_path).exists():
                backup_path = f"{self.database_path}.backup"
                Path(self.database_path).rename(backup_path)

            # Save current database
            with open(self.database_path, 'wb') as f:
                pickle.dump(self.database, f)

            logger.info(f"Saved {len(self.database)} identities to {self.database_path}")
        except Exception as e:
            logger.error(f"Failed to save database: {e}")
            raise

    def add_identity(self,
                     person_id: str,
                     embedding: np.ndarray,
                     metadata: Optional[Dict] = None,
                     overwrite: bool = False):
        """
        Add a new identity to the database

        Args:
            person_id: Unique identifier
            embedding: Face embedding vector
            metadata: Additional metadata (name, role, etc.)
            overwrite: Whether to overwrite existing entry
        """
        if person_id in self.database and not overwrite:
            raise ValueError(f"Identity {person_id} already exists. Use overwrite=True to replace.")

        self.database[person_id] = {
            'embedding': embedding.tolist() if isinstance(embedding, np.ndarray) else embedding,
            'metadata': metadata or {},
            'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()
        }

        logger.info(f"Added identity: {person_id}")

    def remove_identity(self, person_id: str):
        """
        Remove an identity from the database

        Args:
            person_id: Identity to remove
        """
        if person_id not in self.database:
            raise ValueError(f"Identity {person_id} not found in database")

        del self.database[person_id]
        logger.info(f"Removed identity: {person_id}")

    def update_metadata(self, person_id: str, metadata: Dict):
        """
        Update metadata for an identity

        Args:
            person_id: Identity to update
            metadata: New metadata
        """
        if person_id not in self.database:
            raise ValueError(f"Identity {person_id} not found in database")

        self.database[person_id]['metadata'].update(metadata)
        self.database[person_id]['updated_at'] = datetime.now().isoformat()

        logger.info(f"Updated metadata for: {person_id}")

    def search(self,
               embedding: np.ndarray,
               threshold: float = 0.4,
               top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for matching identities

        Args:
            embedding: Query embedding
            threshold: Similarity threshold
            top_k: Number of top matches to return

        Returns:
            List of (person_id, similarity) tuples
        """
        if len(self.database) == 0:
            return []

        # Normalize query embedding
        query_emb = embedding / np.linalg.norm(embedding)

        # Compute similarities
        similarities = []
        for person_id, data in self.database.items():
            db_emb = np.array(data['embedding'])
            db_emb = db_emb / np.linalg.norm(db_emb)

            similarity = np.dot(query_emb, db_emb)

            if similarity >= threshold:
                similarities.append((person_id, float(similarity)))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def get_identity(self, person_id: str) -> Optional[Dict]:
        """
        Get identity data

        Args:
            person_id: Identity to retrieve

        Returns:
            Identity data or None
        """
        return self.database.get(person_id)

    def list_identities(self) -> List[str]:
        """
        List all identities in database

        Returns:
            List of person IDs
        """
        return list(self.database.keys())

    def export_json(self, filepath: str):
        """
        Export database to JSON format

        Args:
            filepath: Path to JSON file
        """
        with open(filepath, 'w') as f:
            json.dump(self.database, f, indent=2)
        logger.info(f"Exported database to {filepath}")

    def import_json(self, filepath: str, merge: bool = False):
        """
        Import database from JSON format

        Args:
            filepath: Path to JSON file
            merge: Whether to merge with existing database
        """
        with open(filepath, 'r') as f:
            imported = json.load(f)

        if merge:
            self.database.update(imported)
        else:
            self.database = imported

        logger.info(f"Imported {len(imported)} identities from {filepath}")

    def get_statistics(self) -> Dict:
        """
        Get database statistics

        Returns:
            Statistics dictionary
        """
        if len(self.database) == 0:
            return {
                'total_identities': 0,
                'embedding_dimension': 0,
                'database_size_kb': 0
            }

        # Get embedding dimension
        first_embedding = np.array(next(iter(self.database.values()))['embedding'])
        embedding_dim = len(first_embedding)

        # Get file size
        db_size = Path(self.database_path).stat().st_size / 1024 if Path(self.database_path).exists() else 0

        stats = {
            'total_identities': len(self.database),
            'embedding_dimension': embedding_dim,
            'database_size_kb': db_size,
            'identities': list(self.database.keys())
        }

        return stats

    def __len__(self):
        """Get number of identities"""
        return len(self.database)

    def __contains__(self, person_id: str):
        """Check if identity exists"""
        return person_id in self.database


def register_face_interactive():
    """Interactive face registration tool"""
    import cv2
    from face_recognition_pipeline import FaceRecognitionPipeline

    print("\n" + "="*60)
    print("Face Registration Tool")
    print("="*60)

    # Initialize
    db_manager = FaceDatabaseManager()
    pipeline = FaceRecognitionPipeline()

    print(f"\nCurrent database: {len(db_manager)} identities")
    print("Identities:", db_manager.list_identities())

    # Open webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    print("\n[Instructions]")
    print("- Position your face in the frame")
    print("- Press 's' to capture and register face")
    print("- Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect faces
        faces = pipeline.detect_and_extract(frame)

        # Draw results
        annotated = pipeline.draw_results(frame, faces, show_landmarks=True)

        # Show info
        info = f"Faces detected: {len(faces)} | Press 's' to save, 'q' to quit"
        cv2.putText(annotated, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow("Face Registration", annotated)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s') and len(faces) > 0:
            # Register first detected face
            face = faces[0]
            embedding = np.array(face['embedding'])

            print("\nFace detected!")
            person_id = input("Enter person ID (unique identifier): ").strip()
            name = input("Enter person name: ").strip()
            role = input("Enter role (optional): ").strip()

            if person_id:
                metadata = {
                    'name': name,
                    'role': role,
                    'registration_date': datetime.now().isoformat()
                }

                try:
                    db_manager.add_identity(person_id, embedding, metadata)
                    db_manager.save()
                    print(f"✓ Registered: {person_id}")
                except ValueError as e:
                    print(f"✗ Error: {e}")
                    overwrite = input("Overwrite existing? (y/n): ").strip().lower()
                    if overwrite == 'y':
                        db_manager.add_identity(person_id, embedding, metadata, overwrite=True)
                        db_manager.save()
                        print(f"✓ Updated: {person_id}")

    cap.release()
    cv2.destroyAllWindows()

    # Final statistics
    print("\n" + "="*60)
    stats = db_manager.get_statistics()
    print("Final Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*60)


if __name__ == "__main__":
    register_face_interactive()
