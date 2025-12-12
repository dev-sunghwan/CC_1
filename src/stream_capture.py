"""
OpenCV-based RTSP Stream Capture Module
Handles real-time RTSP stream ingestion with automatic reconnection
Optimized for NAT/port-forwarded environments using TCP transport
"""

import cv2
import numpy as np
import threading
import queue
import logging
import time
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RTSPStreamCapture:
    """
    OpenCV-based RTSP stream capture with automatic reconnection
    Optimized for NAT/port-forwarded environments using TCP transport
    """

    def __init__(self, rtsp_url: str, queue_size: int = 10):
        """
        Initialize RTSP stream capture

        Args:
            rtsp_url: RTSP stream URL (should include credentials if needed)
            queue_size: Maximum frame queue size
        """
        self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.cap = None
        self.running = False
        self.thread = None
        self.last_frame_time = time.time()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.frame_count = 0

    def _open_stream(self) -> bool:
        """
        Open RTSP stream with OpenCV

        Returns:
            True if successful, False otherwise
        """
        try:
            # Set environment variable for RTSP transport (must be before VideoCapture)
            import os
            os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS'] = 'rtsp_transport;tcp'

            logger.info(f"Opening RTSP stream: {self.rtsp_url}")
            logger.info("Using TCP transport for RTSP")

            # Create VideoCapture with FFmpeg backend
            # Use raw URL without ?tcp parameter
            self.cap = cv2.VideoCapture(self.rtsp_url, cv2.CAP_FFMPEG)

            if not self.cap.isOpened():
                logger.error("Failed to open RTSP stream")
                return False

            # Configure for low latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
            self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)  # 10 second timeout

            # Try to read first frame to verify connection
            ret, frame = self.cap.read()
            if not ret or frame is None:
                logger.error("Failed to read first frame")
                self.cap.release()
                return False

            # Get stream properties
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = self.cap.get(cv2.CAP_PROP_FPS)

            logger.info(f"Stream opened successfully: {width}x{height} @ {fps:.2f} FPS")

            # Put first frame in queue
            try:
                if self.frame_queue.full():
                    self.frame_queue.get_nowait()
                self.frame_queue.put(frame, block=False)
            except queue.Full:
                pass

            return True

        except Exception as e:
            logger.error(f"Error opening stream: {e}")
            if self.cap:
                self.cap.release()
            return False

    def _capture_loop(self):
        """Main capture loop running in separate thread"""
        logger.info("Capture loop started")

        while self.running:
            try:
                # Check if capture is open
                if self.cap is None or not self.cap.isOpened():
                    logger.warning("Stream not open, attempting reconnection...")
                    if not self._reconnect():
                        time.sleep(2)
                        continue

                # Read frame
                ret, frame = self.cap.read()

                if not ret or frame is None:
                    logger.warning("Failed to read frame")
                    if not self._reconnect():
                        time.sleep(1)
                    continue

                # Frame read successfully
                self.frame_count += 1
                self.last_frame_time = time.time()
                self.reconnect_attempts = 0  # Reset on successful read

                # Add to queue (drop oldest if full)
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put(frame, block=False)
                except queue.Full:
                    pass

                # Log progress periodically
                if self.frame_count % 100 == 0:
                    logger.debug(f"Captured {self.frame_count} frames")

            except Exception as e:
                logger.error(f"Capture loop error: {e}")
                time.sleep(1)

        logger.info("Capture loop stopped")

    def _reconnect(self) -> bool:
        """
        Attempt to reconnect to stream

        Returns:
            True if successful, False otherwise
        """
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False

        self.reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")

        # Close existing capture
        if self.cap:
            self.cap.release()
            self.cap = None

        # Wait before reconnecting
        time.sleep(2)

        # Try to reopen
        return self._open_stream()

    def start(self) -> bool:
        """
        Start stream capture

        Returns:
            True if successful, False otherwise
        """
        if self.running:
            logger.warning("Stream capture already running")
            return True

        # Open stream
        if not self._open_stream():
            logger.error("Failed to open stream")
            return False

        # Start capture thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

        logger.info("Stream capture started")
        return True

    def stop(self):
        """Stop stream capture"""
        if not self.running:
            return

        logger.info("Stopping stream capture...")
        self.running = False

        # Wait for thread
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        # Release capture
        if self.cap:
            self.cap.release()
            self.cap = None

        logger.info("Stream capture stopped")

    def get_frame(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """
        Get next frame from queue

        Args:
            timeout: Maximum wait time in seconds

        Returns:
            Frame as numpy array or None if timeout
        """
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return frame
        except queue.Empty:
            return None

    def is_alive(self) -> bool:
        """
        Check if stream is alive

        Returns:
            True if receiving frames
        """
        time_since_last_frame = time.time() - self.last_frame_time
        return self.running and time_since_last_frame < 5.0

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


if __name__ == "__main__":
    # Test the stream capture
    RTSP_URL = "rtsp://admin:Sunap1!!@45.92.235.163:554/profile2/media.smp"

    print("Testing RTSP stream capture with OpenCV...")
    print(f"URL: {RTSP_URL}")

    with RTSPStreamCapture(RTSP_URL) as capture:
        frame_count = 0
        start_time = time.time()

        while frame_count < 100:  # Capture 100 frames for test
            frame = capture.get_frame(timeout=2.0)

            if frame is not None:
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                # Display frame with info
                info_text = f"Frame: {frame_count} | FPS: {fps:.2f} | Shape: {frame.shape}"
                cv2.putText(frame, info_text, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                cv2.imshow("RTSP Stream Test", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                if frame_count % 10 == 0:
                    logger.info(info_text)
            else:
                logger.warning("No frame received (timeout)")
                if not capture.is_alive():
                    logger.error("Stream is not alive!")
                    break

    cv2.destroyAllWindows()
    print(f"\nTest complete. Captured {frame_count} frames")
