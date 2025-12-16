"""
Web Streaming Module
Provides Flask-based MJPEG streaming for viewing video in browser
"""

import cv2
import threading
import logging
import time
from flask import Flask, Response, render_template_string, jsonify
from typing import Optional
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebStreamer:
    """
    Flask-based MJPEG streaming server
    Streams processed video frames to web browser
    """

    def __init__(self, host: str = '0.0.0.0', port: int = 8080):
        """
        Initialize web streamer

        Args:
            host: Host to bind to (0.0.0.0 for all interfaces)
            port: Port to listen on
        """
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.server_thread = None
        self.running = False
        self.start_time = None
        self.frame_count = 0

        # Heartbeat monitoring state
        self.last_frame_update = None
        self.heartbeat_lock = threading.Lock()

        # Setup routes
        self._setup_routes()

    def _setup_routes(self):
        """Setup Flask routes"""

        @self.app.route('/')
        def index():
            """Main page with video stream"""
            html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Face Recognition System - Live Stream</title>
                <style>
                    body {
                        margin: 0;
                        padding: 20px;
                        background-color: #1a1a1a;
                        color: #ffffff;
                        font-family: Arial, sans-serif;
                        display: flex;
                        flex-direction: column;
                        align-items: center;
                    }
                    h1 {
                        color: #4CAF50;
                        margin-bottom: 20px;
                    }
                    .stream-container {
                        max-width: 1280px;
                        width: 100%;
                        background-color: #2a2a2a;
                        padding: 20px;
                        border-radius: 10px;
                        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                        position: relative;
                    }
                    img {
                        width: 100%;
                        height: auto;
                        border-radius: 5px;
                    }
                    .info {
                        margin-top: 20px;
                        padding: 15px;
                        background-color: #333;
                        border-radius: 5px;
                        color: #aaa;
                    }
                    .status {
                        display: inline-block;
                        padding: 5px 10px;
                        background-color: #4CAF50;
                        color: white;
                        border-radius: 3px;
                        margin-top: 10px;
                        transition: background-color 0.3s;
                    }
                    .status.reconnecting {
                        background-color: #ff9800;
                        animation: pulse 1.5s ease-in-out infinite;
                    }
                    .status.error {
                        background-color: #f44336;
                    }
                    @keyframes pulse {
                        0%, 100% { opacity: 1; }
                        50% { opacity: 0.5; }
                    }
                    .overlay {
                        position: absolute;
                        top: 20px;
                        left: 20px;
                        right: 20px;
                        background-color: rgba(255, 152, 0, 0.9);
                        color: white;
                        padding: 15px;
                        border-radius: 5px;
                        display: none;
                        text-align: center;
                        font-weight: bold;
                    }
                    .overlay.show {
                        display: block;
                    }
                </style>
            </head>
            <body>
                <h1>Face Recognition System - Live Stream</h1>
                <div class="stream-container">
                    <div id="reconnectOverlay" class="overlay">
                        ⚠️ Stream frozen - Auto-reconnecting in <span id="countdown">3</span>s...
                    </div>
                    <img id="streamImage" src="{{ url_for('video_feed') }}" alt="Live Stream">
                    <div class="info">
                        <p><strong>Stream URL:</strong> <code>http://localhost:8080/video_feed</code></p>
                        <p><strong>Status:</strong> <span id="statusIndicator" class="status">LIVE</span></p>
                        <p>Streaming from RTSP camera with face detection and tracking</p>
                        <p><small>Last updated: <span id="lastUpdate">Just now</span></small></p>
                    </div>
                </div>

                <script>
                    // Heartbeat monitoring state
                    let heartbeatInterval = null;
                    let reconnectTimer = null;
                    let countdownTimer = null;
                    let heartbeatFailures = 0;
                    const HEARTBEAT_INTERVAL = 2500; // 2.5 seconds
                    const MAX_HEARTBEAT_FAILURES = 2; // Reconnect after 2 failures
                    const RECONNECT_DELAY = 3000; // 3 second delay before reconnect

                    // DOM elements
                    const streamImage = document.getElementById('streamImage');
                    const statusIndicator = document.getElementById('statusIndicator');
                    const overlay = document.getElementById('reconnectOverlay');
                    const countdownSpan = document.getElementById('countdown');
                    const lastUpdateSpan = document.getElementById('lastUpdate');

                    // Image load success handler
                    streamImage.addEventListener('load', function() {
                        resetReconnectTimer();
                        updateStatus('live');
                    });

                    // Image error handler
                    streamImage.addEventListener('error', function() {
                        console.error('Stream image error');
                        updateStatus('error');
                        scheduleReconnect();
                    });

                    // Start heartbeat monitoring
                    function startHeartbeatMonitoring() {
                        if (heartbeatInterval) return;

                        heartbeatInterval = setInterval(async () => {
                            try {
                                const controller = new AbortController();
                                const timeoutId = setTimeout(() => controller.abort(), 3000);

                                const response = await fetch('/heartbeat', {
                                    signal: controller.signal
                                });
                                clearTimeout(timeoutId);

                                if (!response.ok) {
                                    throw new Error('Heartbeat request failed');
                                }

                                const data = await response.json();
                                heartbeatFailures = 0;

                                // Update last update display
                                if (data.staleness_seconds !== null) {
                                    if (data.staleness_seconds < 1) {
                                        lastUpdateSpan.textContent = 'Just now';
                                    } else if (data.staleness_seconds < 60) {
                                        lastUpdateSpan.textContent = Math.round(data.staleness_seconds) + 's ago';
                                    } else {
                                        lastUpdateSpan.textContent = Math.floor(data.staleness_seconds / 60) + 'm ago';
                                    }
                                } else {
                                    lastUpdateSpan.textContent = 'Initializing...';
                                }

                                // Check for stale data
                                if (data.status === 'stale') {
                                    console.warn('Stream data is stale - triggering reconnect');
                                    updateStatus('reconnecting');
                                    scheduleReconnect();
                                } else if (data.status === 'degraded') {
                                    console.warn('Stream degraded - staleness: ' + data.staleness_seconds + 's');
                                    // Still live but show warning in console
                                } else if (data.status === 'healthy') {
                                    updateStatus('live');
                                }

                            } catch (error) {
                                console.error('Heartbeat failed:', error.name, error.message);
                                heartbeatFailures++;

                                if (heartbeatFailures >= MAX_HEARTBEAT_FAILURES) {
                                    console.error('Multiple heartbeat failures - reconnecting');
                                    updateStatus('error');
                                    scheduleReconnect();
                                }
                            }
                        }, HEARTBEAT_INTERVAL);
                    }

                    // Initialize heartbeat monitoring on load
                    startHeartbeatMonitoring();

                    function updateStatus(status) {
                        statusIndicator.className = 'status';

                        if (status === 'live') {
                            statusIndicator.textContent = 'LIVE';
                            statusIndicator.classList.add('live');
                            overlay.classList.remove('show');
                        } else if (status === 'reconnecting') {
                            statusIndicator.textContent = 'RECONNECTING...';
                            statusIndicator.classList.add('reconnecting');
                        } else if (status === 'error') {
                            statusIndicator.textContent = 'ERROR';
                            statusIndicator.classList.add('error');
                        }
                    }

                    function scheduleReconnect() {
                        if (reconnectTimer) return; // Already scheduled

                        overlay.classList.add('show');
                        let countdownValue = Math.floor(RECONNECT_DELAY / 1000);
                        countdownSpan.textContent = countdownValue;

                        // Countdown display
                        countdownTimer = setInterval(function() {
                            countdownValue--;
                            countdownSpan.textContent = countdownValue;
                            if (countdownValue <= 0) {
                                clearInterval(countdownTimer);
                            }
                        }, 1000);

                        // Actual reconnect
                        reconnectTimer = setTimeout(function() {
                            console.log('Reconnecting stream...');
                            reconnectStream();
                        }, RECONNECT_DELAY);
                    }

                    function resetReconnectTimer() {
                        if (reconnectTimer) {
                            clearTimeout(reconnectTimer);
                            reconnectTimer = null;
                        }
                        if (countdownTimer) {
                            clearInterval(countdownTimer);
                            countdownTimer = null;
                        }
                        overlay.classList.remove('show');
                    }

                    function reconnectStream() {
                        // Force reload by adding timestamp
                        const timestamp = new Date().getTime();
                        const baseUrl = "{{ url_for('video_feed') }}";
                        streamImage.src = baseUrl + '?t=' + timestamp;

                        reconnectTimer = null;
                        lastImageUpdate = Date.now();
                    }

                    // Manual refresh with Ctrl+R or F5
                    document.addEventListener('keydown', function(e) {
                        if (e.key === 'F5' || (e.ctrlKey && e.key === 'r')) {
                            e.preventDefault();
                            reconnectStream();
                        }
                    });

                    console.log('Heartbeat monitoring active - checking stream health every ' + (HEARTBEAT_INTERVAL/1000) + 's');
                </script>
            </body>
            </html>
            """
            return render_template_string(html)

        @self.app.route('/video_feed')
        def video_feed():
            """Video streaming route"""
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )

        @self.app.route('/snapshot')
        def snapshot():
            """Save current frame as snapshot"""
            with self.frame_lock:
                if self.current_frame is None:
                    return "No frame available", 404

                # Save snapshot to data directory
                snapshot_path = '/app/data/snapshot.jpg'
                cv2.imwrite(snapshot_path, self.current_frame)
                logger.info(f"Snapshot saved to {snapshot_path}")
                return f"Snapshot saved to {snapshot_path}", 200

        @self.app.route('/health')
        def health():
            """Health check endpoint for monitoring"""
            uptime = time.time() - self.start_time if self.start_time else 0

            health_data = {
                'status': 'healthy' if self.running else 'stopped',
                'stream_active': self.current_frame is not None,
                'uptime_seconds': round(uptime, 2),
                'uptime_formatted': self._format_uptime(uptime),
                'frame_count': self.frame_count,
                'server': {
                    'host': self.host,
                    'port': self.port
                }
            }

            return jsonify(health_data)

        @self.app.route('/heartbeat')
        def heartbeat():
            """Lightweight heartbeat endpoint for freeze detection"""
            with self.heartbeat_lock:
                last_update = self.last_frame_update

            current_time = time.time()

            if last_update is None:
                staleness = -1
                status = 'initializing'
            else:
                staleness = current_time - last_update
                if staleness > 5.0:
                    status = 'stale'
                elif staleness > 3.0:
                    status = 'degraded'
                else:
                    status = 'healthy'

            return jsonify({
                'timestamp': current_time,
                'last_frame_update': last_update,
                'staleness_seconds': round(staleness, 2) if staleness >= 0 else None,
                'frame_count': self.frame_count,
                'status': status,
                'uptime': round(current_time - self.start_time, 2) if self.start_time else 0
            })

    def _format_uptime(self, seconds: float) -> str:
        """Format uptime in human-readable format"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"

    def _generate_frames(self):
        """Generate frames for MJPEG stream"""
        while self.running:
            with self.frame_lock:
                if self.current_frame is None:
                    continue

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if not ret:
                    continue

                frame_bytes = buffer.tobytes()

            # Yield frame in MJPEG format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    def update_frame(self, frame: np.ndarray):
        """
        Update the current frame to be streamed

        Args:
            frame: New frame to stream
        """
        with self.frame_lock:
            self.current_frame = frame.copy()
            self.frame_count += 1

        # Update heartbeat timestamp (separate lock for performance)
        with self.heartbeat_lock:
            self.last_frame_update = time.time()

    def start(self):
        """Start the web server in a separate thread"""
        if self.running:
            logger.warning("Web streamer already running")
            return

        self.running = True
        self.start_time = time.time()
        self.frame_count = 0

        def run_server():
            logger.info(f"Starting web streamer on http://{self.host}:{self.port}")
            # Disable Flask's default logger for cleaner output
            import logging as flask_logging
            log = flask_logging.getLogger('werkzeug')
            log.setLevel(flask_logging.ERROR)

            self.app.run(
                host=self.host,
                port=self.port,
                threaded=True,
                debug=False
            )

        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        logger.info(f"Web streamer started - View at http://localhost:{self.port}")

    def stop(self):
        """Stop the web server"""
        self.running = False
        logger.info("Web streamer stopped")


if __name__ == "__main__":
    # Test the web streamer
    import time

    streamer = WebStreamer(port=8080)
    streamer.start()

    # Generate test frames (colored rectangles)
    try:
        frame_count = 0
        while True:
            # Create a test frame
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            color = (
                (frame_count % 255),
                ((frame_count * 2) % 255),
                ((frame_count * 3) % 255)
            )
            cv2.rectangle(frame, (100, 100), (1180, 620), color, -1)
            cv2.putText(frame, f"Frame {frame_count}", (400, 360),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

            streamer.update_frame(frame)
            frame_count += 1
            time.sleep(0.033)  # ~30 FPS

    except KeyboardInterrupt:
        streamer.stop()
        print("\nTest complete")
