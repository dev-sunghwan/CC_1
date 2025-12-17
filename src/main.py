"""
Main Face Recognition System
Multi-threaded real-time face detection, recognition, and tracking
"""

import cv2
import numpy as np
import logging
import threading
import queue
import time
import signal
import sys
from typing import Optional
from pathlib import Path

from stream_capture import RTSPStreamCapture
from face_recognition_pipeline import FaceRecognitionPipeline
from tracker import BYTETracker
from web_stream import WebStreamer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FaceRecognitionSystem:
    """
    Complete face recognition system with multi-threading
    Architecture:
        Thread 1: Stream capture (GStreamer)
        Thread 2: Face detection & recognition (InsightFace)
        Thread 3: Tracking & visualization
        Main thread: Coordination and display
    """

    def __init__(self,
                 rtsp_url: str,
                 detection_interval: int = 1,
                 display: bool = True,
                 save_video: bool = False,
                 output_path: str = "output.mp4",
                 web_streaming: bool = True,
                 web_port: int = 8080):
        """
        Initialize face recognition system

        Args:
            rtsp_url: RTSP stream URL
            detection_interval: Process every Nth frame for detection
            display: Whether to display output
            save_video: Whether to save output video
            output_path: Path for output video
            web_streaming: Whether to enable web streaming
            web_port: Port for web streaming server
        """
        self.rtsp_url = rtsp_url
        self.detection_interval = detection_interval
        self.display = display
        self.save_video = save_video
        self.output_path = output_path
        self.web_streaming = web_streaming
        self.web_port = web_port

        # Components
        self.stream_capture = None
        self.face_pipeline = None
        self.tracker = None
        self.web_streamer = None

        # Threading
        self.running = False
        self.frame_queue = queue.Queue(maxsize=10)
        self.detection_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)

        # Threads
        self.capture_thread = None
        self.detection_thread = None
        self.tracking_thread = None

        # Statistics
        self.stats = {
            'frames_processed': 0,
            'faces_detected': 0,
            'fps': 0.0,
            'start_time': None
        }

        # Video writer
        self.video_writer = None

        # Shutdown handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info("Shutdown signal received, stopping system...")
        self.stop()
        sys.exit(0)

    def initialize(self):
        """Initialize all components"""
        logger.info("Initializing Face Recognition System...")

        # Initialize stream capture
        logger.info(f"Connecting to RTSP stream: {self.rtsp_url}")
        self.stream_capture = RTSPStreamCapture(self.rtsp_url, queue_size=10)

        # Initialize face recognition pipeline
        logger.info("Loading face recognition models...")
        self.face_pipeline = FaceRecognitionPipeline(
            detection_size=(640, 640),
            det_thresh=0.3,  # 30% confidence threshold (filters out false positives)
            ctx_id=-1  # CPU
        )

        # Initialize tracker
        logger.info("Initializing multi-person tracker...")
        self.tracker = BYTETracker(
            det_thresh=0.3,
            track_thresh=0.5,
            match_thresh=0.4  # Lower threshold = more lenient matching (better for movement)
        )

        # Initialize web streamer
        if self.web_streaming:
            logger.info(f"Initializing web streamer on port {self.web_port}...")
            self.web_streamer = WebStreamer(host='0.0.0.0', port=self.web_port)
            self.web_streamer.set_tracker_reference(self.tracker)
            self.web_streamer.set_camera_info(self.rtsp_url, manufacturer="Hanwha")
            self.web_streamer.face_database = self.face_pipeline.face_database
            self.web_streamer.start()

        logger.info("System initialized successfully")

    def _capture_worker(self):
        """Worker thread for frame capture"""
        logger.info("Capture worker started")
        frame_count = 0

        while self.running:
            try:
                # Get frame from stream
                frame = self.stream_capture.get_frame(timeout=2.0)

                if frame is None:
                    if not self.stream_capture.is_alive():
                        logger.warning("Stream is not alive, waiting for reconnection...")
                        time.sleep(2.0)  # Wait for stream_capture to reconnect
                    continue

                frame_count += 1

                # Add to queue (drop oldest if full)
                try:
                    if self.frame_queue.full():
                        self.frame_queue.get_nowait()
                    self.frame_queue.put((frame_count, frame), block=False)
                except queue.Full:
                    pass

            except Exception as e:
                logger.error(f"Capture worker error: {e}")

        logger.info("Capture worker stopped")

    def _detection_worker(self):
        """Worker thread for face detection and recognition"""
        logger.info("Detection worker started")

        while self.running:
            try:
                # Get frame from queue
                frame_count, frame = self.frame_queue.get(timeout=1.0)

                # Process every Nth frame
                if frame_count % self.detection_interval != 0:
                    self.detection_queue.put((frame_count, frame, []))
                    continue

                # Detect and recognize faces
                faces = self.face_pipeline.detect_and_extract(frame)
                faces = self.face_pipeline.recognize_faces(faces, threshold=0.4)

                # Update statistics
                self.stats['faces_detected'] += len(faces)

                # Add to detection queue
                try:
                    if self.detection_queue.full():
                        self.detection_queue.get_nowait()
                    self.detection_queue.put((frame_count, frame, faces), block=False)
                except queue.Full:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Detection worker error: {e}")

        logger.info("Detection worker stopped")

    def _tracking_worker(self):
        """Worker thread for tracking and visualization"""
        logger.info("Tracking worker started")

        while self.running:
            try:
                # Get detection results
                frame_count, frame, faces = self.detection_queue.get(timeout=1.0)

                # Update tracker
                if len(faces) > 0:
                    tracks = self.tracker.update(faces)
                else:
                    tracks = self.tracker.update([])

                # Draw results
                annotated = self._draw_results(frame, tracks)

                # Add to result queue
                try:
                    if self.result_queue.full():
                        self.result_queue.get_nowait()
                    self.result_queue.put((frame_count, annotated, tracks), block=False)
                except queue.Full:
                    pass

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Tracking worker error: {e}")

        logger.info("Tracking worker stopped")

    def _draw_results(self, frame: np.ndarray, tracks: list) -> np.ndarray:
        """
        Draw tracking results on frame with enhanced visual overlays

        Args:
            frame: Input frame
            tracks: List of active tracks

        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Draw each track
        for track in tracks:
            bbox = track['bbox']
            x1, y1, x2, y2 = bbox

            # Get confidence
            identity = track['identity']
            confidence = track.get('identity_confidence', 0.0)

            # Color-coded by confidence (Phase 4: Enhanced Visual Overlays)
            if identity == 'Unknown':
                color = (0, 0, 255)  # Red
                border_color = (0, 0, 200)  # Darker red for outer border
            elif confidence >= 0.7:
                color = (0, 255, 0)  # Green (high confidence)
                border_color = (0, 200, 0)  # Darker green
            elif confidence >= 0.5:
                color = (0, 255, 255)  # Yellow (medium confidence)
                border_color = (0, 200, 200)  # Darker yellow
            else:
                color = (0, 165, 255)  # Orange (low confidence)
                border_color = (0, 130, 200)  # Darker orange

            # Draw double-bordered bounding box for better visibility
            cv2.rectangle(annotated, (x1, y1), (x2, y2), border_color, 4)  # Outer dark border
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)  # Inner bright border

            # Draw track ID and identity with improved visibility
            track_id = track['track_id']
            label = f"ID:{track_id} | {identity}"
            if identity != 'Unknown':
                label += f" ({confidence:.2f})"

            # Larger font for better readability
            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.9  # Increased from 0.7
            thickness = 2

            # Calculate text size
            (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

            # Draw semi-transparent white background for black text
            label_bg_y1 = max(y1 - text_height - 15, 0)
            label_bg_y2 = y1 - 2

            overlay = annotated.copy()
            cv2.rectangle(overlay, (x1, label_bg_y1), (x1 + text_width + 10, label_bg_y2), (255, 255, 255), -1)
            cv2.addWeighted(overlay, 0.85, annotated, 0.15, 0, annotated)  # More opaque background

            # Draw label - pure black text with shadow for readability
            text_x = x1 + 5
            text_y = y1 - 8
            # Text shadow
            cv2.putText(annotated, label, (text_x + 1, text_y + 1), font, font_scale, (128, 128, 128), thickness)
            # Main text
            cv2.putText(annotated, label, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

            # Draw confidence bar for known identities
            if identity != 'Unknown' and confidence > 0:
                bar_y = y2 + 8
                bar_height = 8
                bar_max_width = x2 - x1
                bar_width = int(bar_max_width * confidence)

                # Background bar (gray)
                cv2.rectangle(annotated, (x1, bar_y), (x2, bar_y + bar_height), (100, 100, 100), -1)

                # Confidence bar (color-coded)
                if confidence >= 0.7:
                    bar_color = (0, 255, 0)  # Green
                elif confidence >= 0.5:
                    bar_color = (0, 255, 255)  # Yellow
                else:
                    bar_color = (0, 165, 255)  # Orange

                cv2.rectangle(annotated, (x1, bar_y), (x1 + bar_width, bar_y + bar_height), bar_color, -1)

                # Bar border
                cv2.rectangle(annotated, (x1, bar_y), (x2, bar_y + bar_height), (255, 255, 255), 1)

                # Update info position to be below confidence bar
                info_y = bar_y + bar_height + 5
            else:
                info_y = y2 + 5

            # Draw track info with background
            info = f"Age:{track['age']} Hits:{track['hits']}"
            info_font_scale = 0.6  # Slightly larger
            info_thickness = 1
            (info_width, info_height), _ = cv2.getTextSize(info, font, info_font_scale, info_thickness)

            overlay2 = annotated.copy()
            cv2.rectangle(overlay2, (x1, info_y), (x1 + info_width + 10, info_y + info_height + 10), (255, 255, 255), -1)
            cv2.addWeighted(overlay2, 0.85, annotated, 0.15, 0, annotated)

            info_text_y = info_y + info_height + 3
            # Info shadow
            cv2.putText(annotated, info, (x1 + 6, info_text_y + 1), font, info_font_scale, (128, 128, 128), info_thickness)
            # Main info
            cv2.putText(annotated, info, (x1 + 5, info_text_y), font, info_font_scale, (0, 0, 0), info_thickness)
        # System info now displayed in web UI dashboard - no overlay needed
        return annotated

    def start(self):
        """Start the face recognition system"""
        if self.running:
            logger.warning("System already running")
            return

        logger.info("Starting Face Recognition System...")
        self.running = True
        self.stats['start_time'] = time.time()

        # Start stream capture
        self.stream_capture.start()
        time.sleep(2)  # Wait for stream to stabilize

        # Start worker threads
        self.capture_thread = threading.Thread(target=self._capture_worker, daemon=True)
        self.detection_thread = threading.Thread(target=self._detection_worker, daemon=True)
        self.tracking_thread = threading.Thread(target=self._tracking_worker, daemon=True)

        self.capture_thread.start()
        self.detection_thread.start()
        self.tracking_thread.start()

        logger.info("All threads started successfully")

        # Main loop: display and save results
        self._main_loop()

    def _main_loop(self):
        """Main loop for display and video saving"""
        logger.info("Entering main display loop...")

        fps_start_time = time.time()
        fps_frame_count = 0

        while self.running:
            try:
                # Get result
                frame_count, annotated, tracks = self.result_queue.get(timeout=1.0)

                # Update statistics
                self.stats['frames_processed'] += 1
                fps_frame_count += 1

                # Calculate FPS every second
                elapsed = time.time() - fps_start_time
                if elapsed >= 1.0:
                    self.stats['fps'] = fps_frame_count / elapsed
                    fps_frame_count = 0
                    fps_start_time = time.time()

                # Display
                if self.display:
                    cv2.imshow("Face Recognition System", annotated)
                    key = cv2.waitKey(1) & 0xFF

                    if key == ord('q'):
                        logger.info("User requested quit")
                        break
                    elif key == ord('s'):
                        # Save current frame
                        filename = f"snapshot_{int(time.time())}.jpg"
                        cv2.imwrite(filename, annotated)
                        logger.info(f"Saved snapshot: {filename}")
                    elif key == ord('r'):
                        # Reset tracker
                        self.tracker.reset()
                        logger.info("Tracker reset")

                # Update web stream
                if self.web_streaming and self.web_streamer:
                    self.web_streamer.update_frame(annotated)

                # Save video
                if self.save_video:
                    if self.video_writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        self.video_writer = cv2.VideoWriter(
                            self.output_path, fourcc, 20.0, (w, h)
                        )
                    self.video_writer.write(annotated)

                # Log statistics periodically
                if self.stats['frames_processed'] % 100 == 0:
                    logger.info(
                        f"Processed {self.stats['frames_processed']} frames | "
                        f"FPS: {self.stats['fps']:.2f} | "
                        f"Active tracks: {len(tracks)}"
                    )

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Main loop error: {e}")

        self.stop()

    def stop(self):
        """Stop the face recognition system"""
        if not self.running:
            return

        logger.info("Stopping Face Recognition System...")
        self.running = False

        # Stop stream capture
        if self.stream_capture:
            self.stream_capture.stop()

        # Wait for threads
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=5)
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=5)
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=5)

        # Close video writer
        if self.video_writer:
            self.video_writer.release()

        # Stop web streamer
        if self.web_streamer:
            self.web_streamer.stop()

        # Close windows
        cv2.destroyAllWindows()

        # Print final statistics
        elapsed = time.time() - self.stats['start_time']
        logger.info(f"\n{'='*50}")
        logger.info("Final Statistics:")
        logger.info(f"  Total Runtime: {elapsed:.2f} seconds")
        logger.info(f"  Frames Processed: {self.stats['frames_processed']}")
        logger.info(f"  Average FPS: {self.stats['frames_processed'] / elapsed:.2f}")
        logger.info(f"  Total Faces Detected: {self.stats['faces_detected']}")
        logger.info(f"{'='*50}\n")

        logger.info("System stopped successfully")


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(description="Face Recognition System")
    parser.add_argument(
        '--rtsp-url',
        type=str,
        default='rtsp://admin:Sunap1!!@45.92.235.163:554/profile2/media.smp',
        help='RTSP stream URL'
    )
    parser.add_argument(
        '--detection-interval',
        type=int,
        default=1,
        help='Process every Nth frame (1 = every frame)'
    )
    parser.add_argument(
        '--no-display',
        action='store_true',
        help='Disable display window'
    )
    parser.add_argument(
        '--save-video',
        action='store_true',
        help='Save output video'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output.mp4',
        help='Output video path'
    )

    args = parser.parse_args()

    # Create system
    system = FaceRecognitionSystem(
        rtsp_url=args.rtsp_url,
        detection_interval=args.detection_interval,
        display=not args.no_display,
        save_video=args.save_video,
        output_path=args.output
    )

    # Initialize and start
    try:
        system.initialize()
        system.start()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"System error: {e}", exc_info=True)
    finally:
        system.stop()


if __name__ == "__main__":
    main()
