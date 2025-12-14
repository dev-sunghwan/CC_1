#!/usr/bin/env python3
"""
Test face detection on a single frame
"""
import cv2
import sys
sys.path.insert(0, '/app/src')

from face_recognition_pipeline import FaceRecognitionPipeline

# RTSP URL
rtsp_url = "rtsp://admin:Sunap1!!@45.92.235.163:554/profile2/media.smp"

print("Connecting to RTSP stream...")
cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

if not cap.isOpened():
    print("ERROR: Could not open stream")
    sys.exit(1)

print("Reading frame...")
ret, frame = cap.read()
cap.release()

if not ret or frame is None:
    print("ERROR: Could not read frame")
    sys.exit(1)

print(f"Frame captured: {frame.shape}")
cv2.imwrite("/app/output/test_frame_raw.jpg", frame)
print("Raw frame saved to /app/output/test_frame_raw.jpg")

# Test face detection
print("\nInitializing face detection...")
pipeline = FaceRecognitionPipeline(
    detection_size=(640, 640),
    det_thresh=0.3,
    ctx_id=-1
)

print("Running face detection...")
faces = pipeline.detect_and_extract(frame)

print(f"\n{'='*50}")
print(f"DETECTION RESULTS:")
print(f"  Faces detected: {len(faces)}")
print(f"{'='*50}")

if len(faces) > 0:
    print("\nFace details:")
    for i, face in enumerate(faces):
        bbox = face['bbox']
        score = face['det_score']
        print(f"  Face {i+1}:")
        print(f"    Bounding box: {bbox}")
        print(f"    Confidence: {score:.3f}")

        # Draw bounding box
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"Face {i+1}: {score:.2f}", (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    cv2.imwrite("/app/output/test_frame_detected.jpg", frame)
    print(f"\nAnnotated frame saved to /app/output/test_frame_detected.jpg")
else:
    print("\nNo faces detected in this frame.")
    print("Possible reasons:")
    print("  1. No people visible in camera view")
    print("  2. Faces too small/far away")
    print("  3. Poor lighting")
    print("  4. Faces at extreme angles")

print("\nTest complete!")
