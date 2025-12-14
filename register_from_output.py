#!/usr/bin/env python3
"""
Register face from a saved image file
"""

import cv2
import sys
import argparse
import numpy as np
sys.path.insert(0, 'src')

from face_recognition_pipeline import FaceRecognitionPipeline

def register_from_image(image_path, person_name):
    """
    Register face from an image file

    Args:
        image_path: Path to image file
        person_name: Name/ID for the person
    """
    print(f"Registering face for: {person_name}")
    print(f"Loading image from: {image_path}")

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"ERROR: Could not load image from {image_path}")
        return False

    print(f"Image loaded: {frame.shape}")

    # Initialize face pipeline
    print("Initializing face detection...")
    pipeline = FaceRecognitionPipeline(
        detection_size=(640, 640),
        det_thresh=0.3,
        ctx_id=-1
    )

    # Detect faces
    print("Detecting faces...")
    faces = pipeline.detect_and_extract(frame)

    if len(faces) == 0:
        print("ERROR: No faces detected in the image!")
        return False

    if len(faces) > 1:
        print(f"WARNING: Detected {len(faces)} faces. Using the face with highest confidence.")

    # Use face with highest confidence
    best_face = max(faces, key=lambda f: f['det_score'])
    print(f"Selected face with confidence: {best_face['det_score']:.3f}")
    print(f"  BBox: {best_face['bbox']}")
    print(f"  Age: {best_face.get('age', 'Unknown')}")
    print(f"  Gender: {'Male' if best_face.get('gender') == 1 else 'Female'}")

    # Add to database
    print(f"\nAdding {person_name} to database...")
    embedding = np.array(best_face['embedding'])

    metadata = {
        'name': person_name,
        'age': best_face.get('age', 'Unknown'),
        'gender': 'Male' if best_face.get('gender') == 1 else 'Female'
    }

    pipeline.add_face_to_database(person_name, embedding, metadata)

    # Save database
    print("Saving database...")
    pipeline.save_database("/app/data/face_database.pkl")
    pipeline.export_database_json("/app/data/face_database.json")

    print(f"\n{'='*50}")
    print(f"SUCCESS! {person_name} has been registered!")
    print(f"{'='*50}")
    print(f"Restart the system to see your name:")
    print(f"  docker-compose restart")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--name', type=str, required=True)
    args = parser.parse_args()

    success = register_from_image(args.image, args.name)
    sys.exit(0 if success else 1)
