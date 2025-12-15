"""
Face Registration Script
Captures a face from the RTSP stream and adds it to the database
Improved version: Filters out known faces and allows selection
"""

import cv2
import argparse
import sys
import time
import numpy as np
from pathlib import Path

from stream_capture import RTSPStreamCapture
from face_recognition_pipeline import FaceRecognitionPipeline


def filter_unknown_faces(faces, pipeline, threshold=0.4):
    """
    Filter out faces that are already in the database

    Args:
        faces: List of detected faces
        pipeline: FaceRecognitionPipeline instance with loaded database
        threshold: Similarity threshold for matching

    Returns:
        List of (face_index, face, matched_name, similarity) tuples
    """
    results = []

    for idx, face in enumerate(faces):
        embedding = np.array(face['embedding'])

        # Find best match in database
        best_match = None
        best_similarity = -1.0

        for person_id, person_data in pipeline.face_database.items():
            # Support both old (single embedding) and new (multi-embedding) formats
            if 'embeddings' in person_data:
                # New format: multiple embeddings per person
                db_embeddings = [np.array(emb) for emb in person_data['embeddings']]

                # Compare against ALL embeddings, use BEST match
                person_best_similarity = max(
                    np.dot(embedding, db_emb) for db_emb in db_embeddings
                )
            else:
                # Old format: single embedding (backward compatible)
                db_embedding = np.array(person_data['embedding'])
                person_best_similarity = np.dot(embedding, db_embedding)

            if person_best_similarity > best_similarity:
                best_similarity = person_best_similarity
                best_match = person_id

        # Mark if this is a known face
        if best_similarity >= threshold:
            status = f"KNOWN ({best_match}, similarity: {best_similarity:.3f})"
        else:
            status = f"UNKNOWN (similarity: {best_similarity:.3f})"

        results.append({
            'index': idx,
            'face': face,
            'matched_name': best_match if best_similarity >= threshold else None,
            'similarity': best_similarity,
            'is_known': best_similarity >= threshold
        })

    return results


def register_face(name: str, rtsp_url: str = "rtsp://admin:Sunap1!!@45.92.235.163:554/profile2/media.smp"):
    """
    Register a new face by capturing from RTSP stream

    Args:
        name: Name of the person to register
        rtsp_url: RTSP stream URL
    """
    print(f"\n{'='*60}")
    print(f"Face Registration for: {name}")
    print(f"{'='*60}\n")

    # Initialize components
    print("Initializing camera stream...")
    stream = RTSPStreamCapture(rtsp_url)
    stream.start()
    time.sleep(2)  # Wait for stream to stabilize

    print("Initializing face recognition...")
    pipeline = FaceRecognitionPipeline(
        detection_size=(640, 640),
        det_thresh=0.5,
        ctx_id=-1
    )

    print(f"\nLooking for faces in the camera...")
    print("Capturing best frame with all faces...")

    # Capture frame with faces
    attempts = 0
    max_attempts = 50
    best_frame_faces = None
    best_avg_score = 0.0

    while attempts < max_attempts:
        frame = stream.get_frame(timeout=2.0)
        if frame is None:
            continue

        # Detect faces
        faces = pipeline.detect_and_extract(frame)

        if len(faces) > 0:
            avg_score = sum(f['det_score'] for f in faces) / len(faces)
            print(f"Attempt {attempts + 1}: Detected {len(faces)} face(s), avg score: {avg_score:.3f}", end='\r')

            if avg_score > best_avg_score:
                best_avg_score = avg_score
                best_frame_faces = faces

            # If we have good detections, stop early
            if avg_score >= 0.80 and len(faces) >= 1:
                print(f"\n\nGood detections found!")
                break

        attempts += 1
        time.sleep(0.1)

    # Stop stream
    stream.stop()

    if best_frame_faces is None or len(best_frame_faces) == 0:
        print(f"\n\nFailed to detect any faces after {max_attempts} attempts.")
        print("Please ensure people are clearly visible in the camera.")
        return False

    print(f"\n\nDetected {len(best_frame_faces)} face(s)")
    print(f"Checking against database (current size: {len(pipeline.face_database)})...\n")

    # Filter out known faces
    face_analysis = filter_unknown_faces(best_frame_faces, pipeline)

    # Display all detected faces
    print("Detected faces:")
    for result in face_analysis:
        face = result['face']
        idx = result['index']
        status = "KNOWN" if result['is_known'] else "UNKNOWN"
        matched = f" - Matches: {result['matched_name']}" if result['is_known'] else ""

        print(f"  [{idx + 1}] {status}: "
              f"Age {face['age']}, "
              f"{'Male' if face['gender'] == 1 else 'Female'}, "
              f"Score {face['det_score']:.3f}, "
              f"Similarity {result['similarity']:.3f}"
              f"{matched}")

    # Filter unknown faces only
    unknown_faces = [r for r in face_analysis if not r['is_known']]

    if len(unknown_faces) == 0:
        print(f"\nAll detected faces are already in the database!")
        print(f"Please ensure {name} is in the frame, or only {name} is visible.")
        return False

    # Select which face to register
    if len(unknown_faces) == 1:
        selected = unknown_faces[0]
        print(f"\nFound 1 unknown face - registering as {name}")
    else:
        print(f"\nFound {len(unknown_faces)} unknown faces.")
        print(f"Which face belongs to {name}?")

        # Show unknown faces with their indices
        for i, result in enumerate(unknown_faces):
            face = result['face']
            print(f"  [{i + 1}] Age {face['age']}, "
                  f"{'Male' if face['gender'] == 1 else 'Female'}, "
                  f"Score {face['det_score']:.3f}")

        # Ask user to select
        while True:
            try:
                choice = input(f"\nEnter number (1-{len(unknown_faces)}), or 0 to cancel: ")
                choice_num = int(choice)

                if choice_num == 0:
                    print("Registration cancelled.")
                    return False

                if 1 <= choice_num <= len(unknown_faces):
                    selected = unknown_faces[choice_num - 1]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(unknown_faces)}")
            except (ValueError, EOFError):
                # If running in non-interactive mode, just pick the first unknown face
                print("\nNon-interactive mode: Selecting first unknown face")
                selected = unknown_faces[0]
                break

    # Register the selected face
    face = selected['face']
    print(f"\nRegistering face:")
    print(f"  Age: {face['age']}")
    print(f"  Gender: {'Male' if face['gender'] == 1 else 'Female'}")
    print(f"  Detection Score: {face['det_score']:.3f}")

    embedding = face['embedding']
    metadata = {
        'age': face['age'],
        'gender': 'Male' if face['gender'] == 1 else 'Female',
        'registration_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'detection_score': face['det_score']
    }

    pipeline.add_face_to_database(name, embedding, metadata)
    pipeline.save_database()

    print(f"\n{'='*60}")
    print(f"Successfully registered {name}!")
    print(f"Total faces in database: {len(pipeline.face_database)}")
    print(f"{'='*60}\n")

    return True


def main():
    parser = argparse.ArgumentParser(description="Register a new face")
    parser.add_argument('--name', type=str, required=True, help='Name of the person to register')
    parser.add_argument('--rtsp-url', type=str,
                       default='rtsp://admin:Sunap1!!@45.92.235.163:554/profile2/media.smp',
                       help='RTSP stream URL')

    args = parser.parse_args()

    success = register_face(args.name, args.rtsp_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
