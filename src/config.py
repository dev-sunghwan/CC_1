"""
Configuration Management
Loads settings from environment variables and config files
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)


class Config:
    """Configuration manager for face recognition system"""

    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration

        Args:
            config_file: Path to YAML config file (optional)
        """
        # Load environment variables
        load_dotenv()

        # Default configuration
        self.config = self._get_default_config()

        # Load from file if provided
        if config_file and Path(config_file).exists():
            self._load_from_file(config_file)

        # Override with environment variables
        self._load_from_env()

        logger.info("Configuration loaded successfully")

    @staticmethod
    def _get_default_config() -> Dict[str, Any]:
        """Get default configuration"""
        return {
            # RTSP Stream
            'rtsp': {
                'url': 'rtsp://45.92.235.163:554/profile2/media.smp',
                'reconnect_attempts': 10,
                'timeout': 5,
                'queue_size': 10
            },
            # Detection
            'detection': {
                'interval': 1,
                'threshold': 0.5,
                'size': (640, 640),
                'model': 'buffalo_l'
            },
            # Recognition
            'recognition': {
                'threshold': 0.4,
                'database_path': 'face_database.pkl'
            },
            # Tracking
            'tracking': {
                'track_threshold': 0.6,
                'match_threshold': 0.8,
                'max_time_lost': 30
            },
            # Display
            'display': {
                'enabled': True,
                'show_fps': True,
                'show_landmarks': False,
                'window_name': 'Face Recognition System'
            },
            # Video
            'video': {
                'save': False,
                'output_path': './output/output.mp4',
                'snapshot_dir': './output/snapshots',
                'fps': 20
            },
            # Logging
            'logging': {
                'level': 'INFO',
                'file': './logs/face_recognition.log',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
            # Performance
            'performance': {
                'num_threads': 8,
                'use_gpu': False,
                'gpu_id': 0
            }
        }

    def _load_from_file(self, config_file: str):
        """Load configuration from YAML file"""
        try:
            with open(config_file, 'r') as f:
                file_config = yaml.safe_load(f)

            # Deep merge with default config
            self._deep_merge(self.config, file_config)
            logger.info(f"Loaded configuration from {config_file}")
        except Exception as e:
            logger.warning(f"Failed to load config file {config_file}: {e}")

    def _load_from_env(self):
        """Load configuration from environment variables"""
        # RTSP
        if os.getenv('RTSP_URL'):
            self.config['rtsp']['url'] = os.getenv('RTSP_URL')
        if os.getenv('RTSP_RECONNECT_ATTEMPTS'):
            self.config['rtsp']['reconnect_attempts'] = int(os.getenv('RTSP_RECONNECT_ATTEMPTS'))

        # Detection
        if os.getenv('DETECTION_INTERVAL'):
            self.config['detection']['interval'] = int(os.getenv('DETECTION_INTERVAL'))
        if os.getenv('DETECTION_THRESHOLD'):
            self.config['detection']['threshold'] = float(os.getenv('DETECTION_THRESHOLD'))
        if os.getenv('DETECTION_SIZE'):
            size = int(os.getenv('DETECTION_SIZE'))
            self.config['detection']['size'] = (size, size)

        # Recognition
        if os.getenv('RECOGNITION_THRESHOLD'):
            self.config['recognition']['threshold'] = float(os.getenv('RECOGNITION_THRESHOLD'))

        # Display
        if os.getenv('DISPLAY_ENABLED'):
            self.config['display']['enabled'] = os.getenv('DISPLAY_ENABLED').lower() == 'true'

        # Video
        if os.getenv('SAVE_VIDEO'):
            self.config['video']['save'] = os.getenv('SAVE_VIDEO').lower() == 'true'
        if os.getenv('OUTPUT_PATH'):
            self.config['video']['output_path'] = os.getenv('OUTPUT_PATH')

        # Logging
        if os.getenv('LOG_LEVEL'):
            self.config['logging']['level'] = os.getenv('LOG_LEVEL')
        if os.getenv('LOG_FILE'):
            self.config['logging']['file'] = os.getenv('LOG_FILE')

        # Performance
        if os.getenv('NUM_THREADS'):
            self.config['performance']['num_threads'] = int(os.getenv('NUM_THREADS'))

    @staticmethod
    def _deep_merge(base: Dict, update: Dict):
        """Deep merge two dictionaries"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                Config._deep_merge(base[key], value)
            else:
                base[key] = value

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value by dot-separated key path

        Args:
            key_path: Dot-separated key path (e.g., 'rtsp.url')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def set(self, key_path: str, value: Any):
        """
        Set configuration value by dot-separated key path

        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def save(self, filepath: str):
        """
        Save configuration to YAML file

        Args:
            filepath: Path to save configuration
        """
        with open(filepath, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {filepath}")

    def __getitem__(self, key: str) -> Any:
        """Dict-like access to config"""
        return self.config[key]

    def __setitem__(self, key: str, value: Any):
        """Dict-like setting of config"""
        self.config[key] = value


# Global config instance
_config = None


def get_config(config_file: Optional[str] = None) -> Config:
    """
    Get global configuration instance

    Args:
        config_file: Optional path to config file

    Returns:
        Config instance
    """
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config


if __name__ == "__main__":
    # Test configuration
    config = Config()

    print("Configuration test:")
    print(f"RTSP URL: {config.get('rtsp.url')}")
    print(f"Detection threshold: {config.get('detection.threshold')}")
    print(f"Display enabled: {config.get('display.enabled')}")

    # Test setting
    config.set('test.value', 123)
    print(f"Test value: {config.get('test.value')}")

    # Save test
    # config.save('config_test.yaml')
    print("\nConfiguration loaded successfully")
