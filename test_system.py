"""
System Test Script
Tests all components of the face recognition system
"""

import sys
import logging
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test if all required modules can be imported"""
    logger.info("Testing imports...")

    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst, GLib
        logger.info("✓ GStreamer bindings OK")
    except Exception as e:
        logger.error(f"✗ GStreamer import failed: {e}")
        return False

    try:
        import insightface
        logger.info("✓ InsightFace OK")
    except Exception as e:
        logger.error(f"✗ InsightFace import failed: {e}")
        return False

    try:
        import onnxruntime as ort
        logger.info("✓ ONNXRuntime OK")
    except Exception as e:
        logger.error(f"✗ ONNXRuntime import failed: {e}")
        return False

    try:
        import cv2
        logger.info(f"✓ OpenCV {cv2.__version__} OK")
    except Exception as e:
        logger.error(f"✗ OpenCV import failed: {e}")
        return False

    return True


def test_gstreamer():
    """Test GStreamer initialization"""
    logger.info("\nTesting GStreamer...")

    try:
        import gi
        gi.require_version('Gst', '1.0')
        from gi.repository import Gst

        Gst.init(None)
        version = Gst.version()
        logger.info(f"✓ GStreamer {version[0]}.{version[1]}.{version[2]} initialized")

        # Test pipeline parsing
        pipeline_str = "videotestsrc ! videoconvert ! appsink"
        pipeline = Gst.parse_launch(pipeline_str)
        logger.info("✓ GStreamer pipeline parsing OK")

        return True
    except Exception as e:
        logger.error(f"✗ GStreamer test failed: {e}")
        return False


def test_insightface():
    """Test InsightFace model loading"""
    logger.info("\nTesting InsightFace...")

    try:
        from insightface.app import FaceAnalysis

        # This will download models if not present
        logger.info("Loading InsightFace models (may take a while on first run)...")
        app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
        app.prepare(ctx_id=-1, det_size=(640, 640))

        logger.info("✓ InsightFace models loaded successfully")

        # Test with dummy image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        faces = app.get(test_image)
        logger.info(f"✓ InsightFace inference OK (detected {len(faces)} faces in random image)")

        return True
    except Exception as e:
        logger.error(f"✗ InsightFace test failed: {e}")
        return False


def test_modules():
    """Test project modules"""
    logger.info("\nTesting project modules...")

    try:
        sys.path.insert(0, 'src')

        from stream_capture import RTSPStreamCapture
        logger.info("✓ stream_capture module OK")

        from face_recognition_pipeline import FaceRecognitionPipeline
        logger.info("✓ face_recognition_pipeline module OK")

        from tracker import BYTETracker
        logger.info("✓ tracker module OK")

        from database_manager import FaceDatabaseManager
        logger.info("✓ database_manager module OK")

        from config import Config
        logger.info("✓ config module OK")

        return True
    except Exception as e:
        logger.error(f"✗ Module import failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_face_pipeline():
    """Test face recognition pipeline with dummy image"""
    logger.info("\nTesting face recognition pipeline...")

    try:
        sys.path.insert(0, 'src')
        from face_recognition_pipeline import FaceRecognitionPipeline

        pipeline = FaceRecognitionPipeline()

        # Create test image with face (or random if no face detection)
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

        # Test detection
        faces = pipeline.detect_and_extract(test_image)
        logger.info(f"✓ Face detection OK (found {len(faces)} faces)")

        # Test recognition (should be empty with random image)
        faces = pipeline.recognize_faces(faces)
        logger.info("✓ Face recognition OK")

        return True
    except Exception as e:
        logger.error(f"✗ Face pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tracker():
    """Test tracking module"""
    logger.info("\nTesting tracker...")

    try:
        sys.path.insert(0, 'src')
        from tracker import BYTETracker

        tracker = BYTETracker()

        # Simulate detections
        test_detections = [
            {'bbox': [100, 100, 200, 250], 'det_score': 0.9, 'embedding': np.random.randn(512)},
            {'bbox': [300, 150, 400, 300], 'det_score': 0.85, 'embedding': np.random.randn(512)},
        ]

        tracks = tracker.update(test_detections)
        logger.info(f"✓ Tracker OK (created {len(tracks)} tracks)")

        return True
    except Exception as e:
        logger.error(f"✗ Tracker test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_database():
    """Test database manager"""
    logger.info("\nTesting database manager...")

    try:
        sys.path.insert(0, 'src')
        from database_manager import FaceDatabaseManager
        import os

        # Use temporary database
        test_db_path = "test_face_database.pkl"

        db = FaceDatabaseManager(test_db_path)

        # Test adding identity
        test_embedding = np.random.randn(512)
        db.add_identity("test_person", test_embedding, {"name": "Test Person"})
        logger.info("✓ Add identity OK")

        # Test search
        results = db.search(test_embedding, threshold=0.1)
        logger.info(f"✓ Search OK (found {len(results)} matches)")

        # Test save/load
        db.save()
        db2 = FaceDatabaseManager(test_db_path)
        logger.info(f"✓ Save/Load OK ({len(db2)} identities)")

        # Cleanup
        if os.path.exists(test_db_path):
            os.remove(test_db_path)

        return True
    except Exception as e:
        logger.error(f"✗ Database test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests"""
    print("="*70)
    print("Face Recognition System - Component Tests")
    print("="*70)

    tests = [
        ("Imports", test_imports),
        ("GStreamer", test_gstreamer),
        ("InsightFace", test_insightface),
        ("Project Modules", test_modules),
        ("Face Pipeline", test_face_pipeline),
        ("Tracker", test_tracker),
        ("Database", test_database),
    ]

    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            logger.error(f"Test '{name}' crashed: {e}")
            results[name] = False

    # Summary
    print("\n" + "="*70)
    print("Test Summary")
    print("="*70)

    passed = sum(results.values())
    total = len(results)

    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{name:20s} {status}")

    print("="*70)
    print(f"Total: {passed}/{total} tests passed")
    print("="*70)

    if passed == total:
        print("\n🎉 All tests passed! System is ready.")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
