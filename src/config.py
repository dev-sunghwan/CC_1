"""
Centralized Configuration for Face Recognition System
All tunable parameters in one place
"""

import os
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DetectionConfig:
    """Face detection configuration"""
    # Detection model settings
    detection_size: Tuple[int, int] = (640, 640)
    det_thresh: float = 0.3  # Detection confidence threshold (30%)
    
    # Recognition settings
    recognition_thresh: float = 0.4  # Face matching threshold (40%)
    
    # Processing
    detection_interval: int = 2  # Process every Nth frame
    ctx_id: int = -1  # -1 for CPU, 0+ for GPU


@dataclass
class TrackerConfig:
    """Multi-person tracking configuration"""
    det_thresh: float = 0.3  # Minimum detection score
    track_thresh: float = 0.5  # Track confirmation threshold
    match_thresh: float = 0.4  # IoU threshold for matching (lower = more lenient)
    max_time_lost: int = 30  # Max frames before deleting lost track
    min_hits_confirm: int = 3  # Hits required to confirm track


@dataclass
class StreamConfig:
    """RTSP stream configuration"""
    # RTSP settings
    rtsp_url: str = os.getenv('RTSP_URL', 'rtsp://localhost:554/stream')
    queue_size: int = 10
    buffer_size: int = 1  # OpenCV buffer size (1 = minimal latency)
    
    # Reconnection
    max_reconnect_attempts: int = 10
    reconnect_delay: float = 2.0  # seconds
    
    # Timeouts
    stream_timeout: float = 5.0  # seconds
    frame_timeout: float = 2.0  # seconds


@dataclass
class WebStreamConfig:
    """Web streaming configuration"""
    host: str = '0.0.0.0'
    port: int = 8080
    jpeg_quality: int = 85  # JPEG compression quality (0-100)


@dataclass
class DatabaseConfig:
    """Face database configuration"""
    database_path: str = "/app/data/face_database.pkl"
    database_json_path: str = "/app/data/face_database.json"
    backup_enabled: bool = True
    max_backups: int = 5  # Keep last N backups


@dataclass
class SystemConfig:
    """System-wide configuration"""
    # Threading
    worker_thread_count: int = 3  # capture, detection, tracking
    
    # Queues
    frame_queue_size: int = 10
    detection_queue_size: int = 10
    result_queue_size: int = 10
    
    # Logging
    log_level: str = os.getenv('LOG_LEVEL', 'INFO')
    log_interval: int = 100  # Log stats every N frames
    
    # Video output
    output_video: bool = False
    output_path: str = "output/output.mp4"
    output_fps: float = 20.0
    
    # Display
    display_window: bool = False  # Set to True for local debugging


@dataclass
class Config:
    """Master configuration object"""
    detection: DetectionConfig = DetectionConfig()
    tracker: TrackerConfig = TrackerConfig()
    stream: StreamConfig = StreamConfig()
    web_stream: WebStreamConfig = WebStreamConfig()
    database: DatabaseConfig = DatabaseConfig()
    system: SystemConfig = SystemConfig()


# Global config instance
config = Config()


def load_config_from_env():
    """Load configuration from environment variables"""
    # Stream settings
    if rtsp_url := os.getenv('RTSP_URL'):
        config.stream.rtsp_url = rtsp_url
    
    # Detection settings
    if det_interval := os.getenv('DETECTION_INTERVAL'):
        config.detection.detection_interval = int(det_interval)
    
    if det_thresh := os.getenv('DETECTION_THRESHOLD'):
        config.detection.det_thresh = float(det_thresh)
    
    # System settings
    if log_level := os.getenv('LOG_LEVEL'):
        config.system.log_level = log_level
    
    # Web stream settings
    if web_port := os.getenv('WEB_PORT'):
        config.web_stream.port = int(web_port)


# Load environment variables on import
load_config_from_env()
