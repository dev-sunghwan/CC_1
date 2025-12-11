"""
Face Recognition System
Real-time face detection, recognition, and tracking using InsightFace and GStreamer
"""

__version__ = "1.0.0"
__author__ = "Face Recognition System Team"

from .stream_capture import RTSPStreamCapture
from .face_recognition_pipeline import FaceRecognitionPipeline
from .tracker import BYTETracker, Track
from .database_manager import FaceDatabaseManager
from .config import Config, get_config

__all__ = [
    'RTSPStreamCapture',
    'FaceRecognitionPipeline',
    'BYTETracker',
    'Track',
    'FaceDatabaseManager',
    'Config',
    'get_config',
]
