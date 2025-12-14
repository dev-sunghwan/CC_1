#!/usr/bin/env python3
"""
Update Embeddings Script
Add additional embeddings to an existing person for improved recognition accuracy
"""

import cv2
import argparse
import sys
import time
import numpy as np
from pathlib import Path

from stream_capture import RTSPStreamCapture
from face_recognition_pipeline import FaceRecognitionPipeline


def update_person_embeddings(
    name: str,
    num_samples: int = 2,
    rtsp_url: str = "rtsp://admin:Sunap1!!@45.92.235.163:554/profile2/media.smp"
):
    """
    Add additional embeddings to an existing person's database entry

    Args:
        name: Name of the person to update
        num_samples: Number of additional embeddings to capture (default: 2)
        rtsp_url: RTSP stream URL
    """
    print(f"\n{'='*70}")
    print(f"UPDATE EMBEDDINGS FOR: {name}")
    print(f"{'='*70}\n")

    # Initialize components
    print("Initializing face recognition pipeline...")
    pipeline = FaceRecognitionPipeline(
        detection_size=(640, 640),
        det_thresh=0.5,
        ctx_id=-1
    )

    # Check if person exists in database
    if name not in pipeline.face_database:
        print(f"\nERROR: '{name}' not found in database!")
        print("\nCurrent database contains:")
        for person_name in pipeline.face_database.keys():
            print(f"  - {person_name}")
        return False

    # Get current embeddings
    person_data = pipeline.face_database[name]

    # Handle both old (single embedding) and new (multi-embedding) formats
    if 'embeddings' in person_data:
        current_embeddings = person_data['embeddings']
        print(f"\nCurrent status: {name} has {len(current_embeddings)} embedding(s)")
    elif 'embedding' in person_data:
        current_embeddings = [person_data['embedding']]
        print(f"\nCurrent status: {name} has 1 embedding (old format)")
    else:
        print(f"\nERROR: No embeddings found for {name}")
        return False

    print(f"Will capture {num_samples} additional sample(s)\n")

    # Initialize camera stream
    print("Connecting to camera stream...")
    stream = RTSPStreamCapture(rtsp_url)
    stream.start()
    time.sleep(2)  # Wait for stream to stabilize

    # Capture additional embeddings with guided poses
    new_embeddings = []
    pose_instructions = [
        "Face the camera directly and stay still...",
        "Turn your head SLIGHTLY to the LEFT and hold...",
        "Turn your head SLIGHTLY to the RIGHT and hold...",
        "Look at the camera with a slight smile...",
        "Tilt your head slightly and look at camera..."
    ]

    for sample_idx in range(num_samples):
        print(f"\n{'='*70}")
        print(f"SAMPLE {sample_idx + 1} of {num_samples}")
        print(f"{'='*70}")

        # Show pose instruction
        if sample_idx < len(pose_instructions):
            print(f"\n{pose_instructions[sample_idx]}")
        else:
            print(f"\nStay in position...")

        print("\nCapturing in 3 seconds...")
        for countdown in [3, 2, 1]:
            print(f"  {countdown}...")
            time.sleep(1)

        print("\nCapturing frames... (hold position for 3 seconds)")

        # Capture best frame for this pose
        attempts = 0
        max_attempts = 30
        best_face = None
        best_score = 0.0

        capture_start = time.time()
        while time.time() - capture_start < 3.0 and attempts < max_attempts:
            frame = stream.get_frame(timeout=2.0)
            if frame is None:
                attempts += 1
                continue

            # Detect faces
            faces = pipeline.detect_and_extract(frame)

            # Find the face that best matches the person
            for face in faces:
                face_embedding = np.array(face['embedding'])

                # Compare with existing embeddings to ensure it's the same person
                similarities = []
                for existing_emb in current_embeddings:
                    emb_array = np.array(existing_emb)
                    similarity = np.dot(face_embedding, emb_array)
                    similarities.append(similarity)

                avg_similarity = np.mean(similarities)

                # Must match existing embeddings with high confidence
                if avg_similarity > 0.4 and face['det_score'] > best_score:
                    best_score = face['det_score']
                    best_face = face
                    print(f"  Found matching face: score={face['det_score']:.3f}, "
                          f"similarity={avg_similarity:.3f}", end='\r')

            attempts += 1

        if best_face is None:
            print(f"\n\nFailed to capture face for sample {sample_idx + 1}")
            print("Please ensure:")
            print(f"  - {name} is clearly visible in the camera")
            print("  - Lighting is adequate")
            print("  - You're holding the pose steady")

            retry = input("\nRetry this sample? (y/n): ").strip().lower()
            if retry == 'y':
                sample_idx -= 1  # Retry this sample
                continue
            else:
                print("Skipping this sample...")
                continue

        print(f"\n\nCapture successful! Score: {best_face['det_score']:.3f}")
        new_embeddings.append(best_face['embedding'])

    # Stop stream
    stream.stop()

    if len(new_embeddings) == 0:
        print("\n\nNo new embeddings captured. Update cancelled.")
        return False

    print(f"\n\n{'='*70}")
    print(f"UPDATING DATABASE")
    print(f"{'='*70}\n")

    # Combine old and new embeddings
    all_embeddings = current_embeddings + new_embeddings

    print(f"Total embeddings for {name}: {len(all_embeddings)}")
    print(f"  - Original: {len(current_embeddings)}")
    print(f"  - New: {len(new_embeddings)}")

    # Update database with combined embeddings
    metadata = person_data.get('metadata', {})
    metadata['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')

    pipeline.add_face_to_database(name, all_embeddings, metadata)
    pipeline.save_database()

    print(f"\n{'='*70}")
    print(f"SUCCESS!")
    print(f"{'='*70}")
    print(f"\n{name} now has {len(all_embeddings)} embeddings for improved accuracy!")
    print("\nRestart the face recognition system to use the updated database.\n")

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Update embeddings for existing person in database"
    )
    parser.add_argument(
        '--name',
        type=str,
        required=True,
        help='Name of the person to update'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=2,
        help='Number of additional samples to capture (default: 2)'
    )
    parser.add_argument(
        '--rtsp-url',
        type=str,
        default='rtsp://admin:Sunap1!!@45.92.235.163:554/profile2/media.smp',
        help='RTSP stream URL'
    )

    args = parser.parse_args()

    success = update_person_embeddings(args.name, args.samples, args.rtsp_url)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
