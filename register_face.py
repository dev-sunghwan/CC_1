#!/usr/bin/env python3
"""
Face Registration Tool
Captures a face from the RTSP stream and adds it to the database
"""

import cv2
import sys
import argparse
sys.path.insert(0, 'src')

from face_recognition_pipeline import FaceRecognitionPipeline

def register_face(rtsp_url, person_name):
    """
    Capture face from RTSP stream and register with given name

    Args:
        rtsp_url: RTSP stream URL
        person_name: Name/ID for the person
    """
    print(f"Registering face for: {person_name}")
    print(f"Connecting to RTSP stream: {rtsp_url}")

    # Connect to stream
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("ERROR: Could not open RTSP stream")
        return False

    # Skip a few frames to let stream stabilize
    for _ in range(5):
        cap.read()

    # Capture frame
    print("Capturing frame...")
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print("ERROR: Could not capture frame")
        return False

    print(f"Frame captured: {frame.shape}")

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
        print("ERROR: No faces detected!")
        print("Please ensure:")
        print("  - You are visible in the camera")
        print("  - Good lighting")
        print("  - Face is clearly visible (not at extreme angle)")
        return False

    if len(faces) > 1:
        print(f"WARNING: Detected {len(faces)} faces. Using the face with highest confidence.")

    # Use face with highest confidence
    best_face = max(faces, key=lambda f: f['det_score'])
    print(f"Selected face with confidence: {best_face['det_score']:.3f}")

    # Add to database
    print(f"Adding {person_name} to database...")
    import numpy as np
    embedding = np.array(best_face['embedding'])

    metadata = {
        'name': person_name,
        'age': best_face.get('age', 'Unknown'),
        'gender': 'Male' if best_face.get('gender') == 1 else 'Female'
    }

    pipeline.add_face_to_database(person_name, embedding, metadata)

    # Save database
    print("Saving database...")
    pipeline.save_database("data/face_database.pkl")

    # Also save as JSON for easier viewing
    pipeline.export_database_json("data/face_database.json")

    print(f"\n{'='*50}")
    print(f"SUCCESS! {person_name} has been registered!")
    print(f"{'='*50}")
    print(f"Database location: data/face_database.pkl")
    print(f"JSON export: data/face_database.json")
    print(f"\nThe system will now recognize you as '{person_name}'")
    print(f"Restart the system to apply changes:")
    print(f"  docker-compose restart")

    return True


def main():
    parser = argparse.ArgumentParser(description="Register a face in the database")
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Name/ID for the person (e.g., "John Smith")'
    )
    parser.add_argument(
        '--rtsp-url',
        type=str,
        default='rtsp://admin:Sunap1!!@45.92.235.163:554/profile2/media.smp',
        help='RTSP stream URL'
    )

    args = parser.parse_args()

    success = register_face(args.rtsp_url, args.name)

    if success:
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
