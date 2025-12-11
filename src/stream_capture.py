"""
GStreamer-based RTSP Stream Capture Module
Handles real-time RTSP stream ingestion with automatic reconnection
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import numpy as np
import cv2
import threading
import queue
import logging
from typing import Optional, Callable
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RTSPStreamCapture:
    """
    GStreamer-based RTSP stream capture with automatic reconnection
    Optimized for NAT/port-forwarded environments
    """

    def __init__(self, rtsp_url: str, queue_size: int = 10):
        """
        Initialize RTSP stream capture

        Args:
            rtsp_url: RTSP stream URL
            queue_size: Maximum frame queue size
        """
        Gst.init(None)

        self.rtsp_url = rtsp_url
        self.frame_queue = queue.Queue(maxsize=queue_size)
        self.pipeline = None
        self.loop = None
        self.running = False
        self.thread = None
        self.last_frame_time = time.time()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10

    def _build_pipeline(self) -> str:
        """
        Build GStreamer pipeline string optimized for RTSP over NAT

        Returns:
            Pipeline string
        """
        pipeline_str = (
            f"rtspsrc location={self.rtsp_url} "
            "protocols=tcp "  # Force TCP for NAT traversal
            "latency=200 "    # Low latency buffer
            "drop-on-latency=true "
            "! rtph264depay "  # Handle FU-A NAL units
            "! h264parse "
            "! avdec_h264 "    # CPU decoder (sufficient for i7-13700)
            "! videoconvert "
            "! video/x-raw,format=BGR "  # OpenCV-compatible format
            "! appsink name=sink emit-signals=true max-buffers=1 drop=true"
        )
        return pipeline_str

    def _on_new_sample(self, sink) -> Gst.FlowReturn:
        """
        Callback for new frame from GStreamer

        Args:
            sink: GStreamer appsink element

        Returns:
            Gst.FlowReturn status
        """
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buffer = sample.get_buffer()
        caps = sample.get_caps()

        # Extract frame dimensions
        structure = caps.get_structure(0)
        width = structure.get_value('width')
        height = structure.get_value('height')

        # Convert to numpy array
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        frame = np.ndarray(
            shape=(height, width, 3),
            dtype=np.uint8,
            buffer=map_info.data
        )

        # Deep copy to prevent memory issues
        frame = frame.copy()
        buffer.unmap(map_info)

        # Add to queue (drop oldest if full)
        try:
            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self.frame_queue.put(frame, block=False)
            self.last_frame_time = time.time()
            self.reconnect_attempts = 0  # Reset on successful frame
        except queue.Full:
            logger.warning("Frame queue full, dropping frame")

        return Gst.FlowReturn.OK

    def _on_bus_message(self, bus, message):
        """
        Handle GStreamer bus messages

        Args:
            bus: GStreamer bus
            message: Bus message
        """
        t = message.type

        if t == Gst.MessageType.EOS:
            logger.warning("End of stream, attempting reconnect...")
            self._handle_reconnect()
        elif t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"GStreamer error: {err}, {debug}")
            self._handle_reconnect()
        elif t == Gst.MessageType.WARNING:
            warn, debug = message.parse_warning()
            logger.warning(f"GStreamer warning: {warn}")
        elif t == Gst.MessageType.STATE_CHANGED:
            if message.src == self.pipeline:
                old_state, new_state, pending = message.parse_state_changed()
                logger.info(f"Pipeline state: {old_state.value_nick} -> {new_state.value_nick}")

    def _handle_reconnect(self):
        """Handle reconnection logic"""
        if self.reconnect_attempts >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached, stopping...")
            self.stop()
            return

        self.reconnect_attempts += 1
        logger.info(f"Reconnection attempt {self.reconnect_attempts}/{self.max_reconnect_attempts}")

        # Stop current pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        # Wait before reconnecting
        time.sleep(2)

        # Restart pipeline
        if self.running:
            self._start_pipeline()

    def _start_pipeline(self):
        """Start GStreamer pipeline"""
        pipeline_str = self._build_pipeline()
        logger.info(f"Starting pipeline: {pipeline_str}")

        self.pipeline = Gst.parse_launch(pipeline_str)

        # Get appsink and set callback
        sink = self.pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_new_sample)

        # Setup bus
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        # Start pipeline
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            logger.error("Failed to start pipeline")
            return False

        return True

    def _run_loop(self):
        """Run GStreamer main loop in separate thread"""
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        except Exception as e:
            logger.error(f"Main loop error: {e}")

    def start(self):
        """Start stream capture"""
        if self.running:
            logger.warning("Stream capture already running")
            return

        self.running = True

        # Start pipeline
        if not self._start_pipeline():
            self.running = False
            return False

        # Start GLib main loop in separate thread
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        logger.info("Stream capture started")
        return True

    def stop(self):
        """Stop stream capture"""
        if not self.running:
            return

        self.running = False

        # Stop pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)

        # Stop main loop
        if self.loop and self.loop.is_running():
            self.loop.quit()

        # Wait for thread
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

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
    RTSP_URL = "rtsp://45.92.235.163:554/profile2/media.smp"

    print("Testing RTSP stream capture...")
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
