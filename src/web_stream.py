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

        # Tracker and database references for advanced endpoints
        self.tracker_ref = None
        self.face_database = {}

        # Camera information
        self.camera_info = {
            'rtsp_url': None,
            'ip_address': None,
            'username': None,
            'port': None,
            'path': None,
            'manufacturer': 'Unknown'
        }

        # Pause/resume state
        self.paused = False
        self.pause_lock = threading.Lock()

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
    <title>Face Recognition System - Enhanced UI</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            background-color: #0f0f0f;
            color: #ffffff;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif;
            overflow: hidden;
        }

        /* Header */
        .header {
            background: linear-gradient(135deg, #1a1a1a 0%, #2a2a2a 100%);
            padding: 20px;
            border-bottom: 2px solid #4CAF50;
        }

        h1 {
            font-size: 24px;
            margin-bottom: 15px;
            color: #4CAF50;
        }

        /* Statistics Dashboard */
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }

        .stat-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(76, 175, 80, 0.3);
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s;
        }

        .stat-card:hover {
            border-color: #4CAF50;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.2);
        }

        .stat-label {
            font-size: 11px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }

        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #4CAF50;
        }

        .stat-unit {
            font-size: 14px;
            color: #aaa;
            margin-left: 4px;
        }

        /* Main Content Area */
        .main-container {
            display: grid;
            grid-template-columns: 1fr 320px;
            gap: 20px;
            padding: 20px;
            height: calc(100vh - 220px);
        }

        /* Stream Container */
        .stream-section {
            display: flex;
            flex-direction: column;
            gap: 15px;
        }

        .stream-wrapper {
            position: relative;
            background: #1a1a1a;
            border-radius: 10px;
            overflow: hidden;
            flex: 1;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        #streamImage {
            max-width: 100%;
            max-height: 100%;
            object-fit: contain;
            border-radius: 10px;
        }

        .stream-status {
            position: absolute;
            top: 15px;
            right: 15px;
            background: rgba(0, 0, 0, 0.8);
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: bold;
            border: 2px solid #4CAF50;
        }

        .stream-status.degraded {
            border-color: #ff9800;
            color: #ff9800;
        }

        .stream-status.error {
            border-color: #f44336;
            color: #f44336;
        }

        /* Control Bar */
        .control-bar {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .btn {
            background: rgba(76, 175, 80, 0.2);
            border: 1px solid #4CAF50;
            color: #4CAF50;
            padding: 12px 24px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.3s;
            flex: 1;
            min-width: 120px;
        }

        .btn:hover {
            background: #4CAF50;
            color: #000;
            transform: translateY(-2px);
            box-shadow: 0 4px 12px rgba(76, 175, 80, 0.4);
        }

        .btn:active {
            transform: translateY(0);
        }

        .btn.paused {
            background: rgba(255, 152, 0, 0.2);
            border-color: #ff9800;
            color: #ff9800;
        }

        /* Face List Sidebar */
        .face-sidebar {
            background: #1a1a1a;
            border-radius: 10px;
            padding: 20px;
            overflow-y: auto;
            border: 1px solid rgba(76, 175, 80, 0.3);
        }

        .sidebar-header {
            font-size: 18px;
            font-weight: bold;
            margin-bottom: 15px;
            color: #4CAF50;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .face-count {
            background: rgba(76, 175, 80, 0.2);
            padding: 4px 12px;
            border-radius: 12px;
            font-size: 14px;
        }

        .face-list {
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .face-card {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(76, 175, 80, 0.2);
            border-radius: 8px;
            padding: 15px;
            transition: all 0.3s;
        }

        .face-card:hover {
            border-color: #4CAF50;
            background: rgba(255, 255, 255, 0.08);
        }

        .face-name {
            font-size: 16px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #4CAF50;
        }

        .face-detail {
            font-size: 12px;
            color: #aaa;
            margin: 4px 0;
        }

        .confidence-bar {
            width: 100%;
            height: 6px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 3px;
            overflow: hidden;
            margin-top: 8px;
        }

        .confidence-fill {
            height: 100%;
            background: linear-gradient(90deg, #4CAF50, #8BC34A);
            transition: width 0.3s;
        }

        .empty-state {
            text-align: center;
            padding: 40px 20px;
            color: #666;
        }

        .empty-state-icon {
            font-size: 48px;
            margin-bottom: 15px;
        }

        /* Reconnect Overlay */
        .reconnect-overlay {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            background: rgba(255, 152, 0, 0.95);
            color: white;
            padding: 30px 50px;
            border-radius: 10px;
            display: none;
            z-index: 1000;
            text-align: center;
            font-weight: bold;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
        }

        .reconnect-overlay.show {
            display: block;
        }

        /* Responsive Design */
        @media (max-width: 1024px) {
            .main-container {
                grid-template-columns: 1fr;
                height: auto;
            }

            .face-sidebar {
                max-height: 400px;
            }

            .stats-grid {
                grid-template-columns: repeat(3, 1fr);
            }
        }

        @media (max-width: 640px) {
            .stats-grid {
                grid-template-columns: repeat(2, 1fr);
            }

            .control-bar {
                flex-direction: column;
            }

            .btn {
                width: 100%;
            }
        }

        /* Loading Animation */
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }

        .loading {
            animation: pulse 1.5s ease-in-out infinite;
        }
    </style>
</head>
<body>
    <!-- Header -->
    <div class="header">
        <h1>Face Recognition System - Live Monitoring</h1>

        <!-- Statistics Dashboard -->
        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">FPS</div>
                <div class="stat-value"><span id="statFps">--</span><span class="stat-unit">fps</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Active Faces</div>
                <div class="stat-value"><span id="statActiveFaces">--</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Database</div>
                <div class="stat-value"><span id="statDatabase">--</span><span class="stat-unit">faces</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Uptime</div>
                <div class="stat-value"><span id="statUptime">--</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Frames</div>
                <div class="stat-value"><span id="statFrames">--</span></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Health</div>
                <div class="stat-value"><span id="statHealth">--</span></div>
            </div>
        </div>
    </div>

    <!-- Main Content -->
    <div class="main-container">
        <!-- Stream Section -->
        <div class="stream-section">
            <div class="stream-wrapper">
                <div class="stream-status" id="streamStatus">LIVE</div>
                <img id="streamImage" src="{{ url_for('video_feed') }}" alt="Live Stream">
            </div>

            <!-- Control Bar -->
            <div class="control-bar">
                <button class="btn" onclick="refreshStream()">Refresh</button>
                <button class="btn" id="pauseBtn" onclick="togglePause()">Pause</button>
                <button class="btn" onclick="downloadSnapshot()">Snapshot</button>
            </div>
        </div>

        <!-- Face List Sidebar -->
        <div class="face-sidebar">
            <div class="sidebar-header">
                <span>Detected Faces</span>
                <span class="face-count" id="faceCount">0</span>
            </div>
            <div class="face-list" id="faceList">
                <div class="empty-state">
                    <div class="empty-state-icon">👤</div>
                    <div>No faces detected</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Reconnect Overlay -->
    <div class="reconnect-overlay" id="reconnectOverlay">
        ⚠️ Stream frozen - Auto-reconnecting in <span id="countdown">3</span>s...
    </div>

    <script>
        // State
        let isPaused = false;
        let heartbeatInterval = null;
        let statsInterval = null;
        let facesInterval = null;
        let reconnectTimer = null;
        let countdownTimer = null;
        let heartbeatFailures = 0;

        const HEARTBEAT_INTERVAL = 2500;
        const STATS_INTERVAL = 2000;
        const FACES_INTERVAL = 1500;
        const MAX_HEARTBEAT_FAILURES = 2;
        const RECONNECT_DELAY = 3000;

        // DOM Elements
        const streamImage = document.getElementById('streamImage');
        const streamStatus = document.getElementById('streamStatus');
        const reconnectOverlay = document.getElementById('reconnectOverlay');
        const countdownSpan = document.getElementById('countdown');
        const pauseBtn = document.getElementById('pauseBtn');
        const faceList = document.getElementById('faceList');
        const faceCount = document.getElementById('faceCount');

        // Initialize
        window.onload = function() {
            startHeartbeatMonitoring();
            startStatsPolling();
            startFacesPolling();
        };

        // Heartbeat Monitoring
        function startHeartbeatMonitoring() {
            if (heartbeatInterval) return;

            heartbeatInterval = setInterval(async () => {
                if (isPaused) return;

                try {
                    const controller = new AbortController();
                    const timeoutId = setTimeout(() => controller.abort(), 3000);

                    const response = await fetch('/heartbeat', { signal: controller.signal });
                    clearTimeout(timeoutId);

                    if (!response.ok) throw new Error('Heartbeat failed');

                    const data = await response.json();
                    heartbeatFailures = 0;

                    // Update status
                    if (data.status === 'stale') {
                        updateStreamStatus('error', 'STALE');
                        scheduleReconnect();
                    } else if (data.status === 'degraded') {
                        updateStreamStatus('degraded', 'SLOW');
                    } else {
                        updateStreamStatus('live', 'LIVE');
                    }
                } catch (error) {
                    heartbeatFailures++;
                    if (heartbeatFailures >= MAX_HEARTBEAT_FAILURES) {
                        updateStreamStatus('error', 'ERROR');
                        scheduleReconnect();
                    }
                }
            }, HEARTBEAT_INTERVAL);
        }

        // Stats Polling
        function startStatsPolling() {
            if (statsInterval) return;

            statsInterval = setInterval(async () => {
                if (isPaused) return;

                try {
                    const response = await fetch('/health');
                    if (!response.ok) return;

                    const data = await response.json();
                    document.getElementById('statFps').textContent = data.fps.average.toFixed(1);
                    document.getElementById('statActiveFaces').textContent = data.active_tracks;
                    document.getElementById('statDatabase').textContent = data.database_size;
                    document.getElementById('statUptime').textContent = data.uptime_formatted;
                    document.getElementById('statFrames').textContent = data.frame_count;
                    document.getElementById('statHealth').textContent = data.status === 'healthy' ? 'OK' : 'ERR';
                } catch (error) {
                    console.error('Stats polling error:', error);
                }
            }, STATS_INTERVAL);
        }

        // Faces Polling
        function startFacesPolling() {
            if (facesInterval) return;

            facesInterval = setInterval(async () => {
                if (isPaused) return;

                try {
                    const response = await fetch('/faces');
                    if (!response.ok) return;

                    const data = await response.json();
                    updateFaceList(data.faces);
                    faceCount.textContent = data.count;
                } catch (error) {
                    console.error('Faces polling error:', error);
                }
            }, FACES_INTERVAL);
        }

        // Update Face List
        function updateFaceList(faces) {
            if (faces.length === 0) {
                faceList.innerHTML = `
                    <div class="empty-state">
                        <div class="empty-state-icon">👤</div>
                        <div>No faces detected</div>
                    </div>
                `;
                return;
            }

            faceList.innerHTML = faces.map(face => `
                <div class="face-card">
                    <div class="face-name">${face.identity}</div>
                    <div class="face-detail">Track ID: #${face.track_id}</div>
                    <div class="face-detail">Confidence: ${(face.confidence * 100).toFixed(1)}%</div>
                    <div class="face-detail">Time in view: ${face.time_in_view_seconds}s</div>
                    <div class="face-detail">Detection: ${(face.det_score * 100).toFixed(1)}%</div>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: ${face.confidence * 100}%"></div>
                    </div>
                </div>
            `).join('');
        }

        // Stream Status
        function updateStreamStatus(status, text) {
            streamStatus.className = 'stream-status ' + status;
            streamStatus.textContent = text;
        }

        // Reconnect Logic
        function scheduleReconnect() {
            if (reconnectTimer) return;

            reconnectOverlay.classList.add('show');
            let countdownValue = Math.floor(RECONNECT_DELAY / 1000);
            countdownSpan.textContent = countdownValue;

            countdownTimer = setInterval(() => {
                countdownValue--;
                countdownSpan.textContent = countdownValue;
                if (countdownValue <= 0) clearInterval(countdownTimer);
            }, 1000);

            reconnectTimer = setTimeout(() => {
                refreshStream();
                reconnectOverlay.classList.remove('show');
                reconnectTimer = null;
            }, RECONNECT_DELAY);
        }

        // Control Functions
        function refreshStream() {
            const timestamp = new Date().getTime();
            streamImage.src = "{{ url_for('video_feed') }}?t=" + timestamp;
            heartbeatFailures = 0;
        }

        async function togglePause() {
            isPaused = !isPaused;

            try {
                const endpoint = isPaused ? '/pause' : '/resume';
                await fetch(endpoint, { method: 'POST' });

                pauseBtn.textContent = isPaused ? 'Resume' : 'Pause';
                pauseBtn.classList.toggle('paused', isPaused);
            } catch (error) {
                console.error('Pause/Resume error:', error);
            }
        }

        async function downloadSnapshot() {
            try {
                window.location.href = '/snapshot/download';
            } catch (error) {
                console.error('Snapshot download error:', error);
            }
        }

        // Keyboard Shortcuts
        document.addEventListener('keydown', (e) => {
            if (e.key === 'F5' || (e.ctrlKey && e.key === 'r')) {
                e.preventDefault();
                refreshStream();
            } else if (e.key === ' ') {
                e.preventDefault();
                togglePause();
            }
        });

        console.log('Enhanced UI loaded - Monitoring active');
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
            """Enhanced health check endpoint with comprehensive statistics"""
            uptime = time.time() - self.start_time if self.start_time else 0

            # Calculate active tracks
            active_tracks = 0
            if self.tracker_ref:
                active_tracks = len([t for t in self.tracker_ref.tracks if t.state == 'confirmed'])

            health_data = {
                'status': 'healthy' if self.running else 'stopped',
                'stream_active': self.current_frame is not None,
                'uptime_seconds': round(uptime, 2),
                'uptime_formatted': self._format_uptime(uptime),
                'frame_count': self.frame_count,
                'fps': {
                    'average': round(self.frame_count / uptime, 2) if uptime > 0 else 0.0
                },
                'database_size': len(self.face_database),
                'active_tracks': active_tracks,
                'server': {
                    'host': self.host,
                    'port': self.port
                }
            }

            return jsonify(health_data)

        @self.app.route('/camera_info')
        def camera_info():
            """Get current camera information"""
            return jsonify({
                'timestamp': time.time(),
                'camera': self.camera_info
            })

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

        @self.app.route('/faces')
        def faces():
            """Get currently tracked faces"""
            if self.tracker_ref is None:
                return jsonify({'error': 'Tracker not available', 'faces': []})

            faces_data = []
            for track in self.tracker_ref.tracks:
                if track.state == 'confirmed':
                    faces_data.append({
                        'track_id': track.track_id,
                        'identity': track.identity,
                        'confidence': round(track.identity_confidence, 3),
                        'bbox': track.bbox.astype(int).tolist() if hasattr(track.bbox, 'astype') else track.bbox,
                        'age': track.age,
                        'hits': track.hits,
                        'time_since_update': track.time_since_update,
                        'time_in_view_seconds': round(track.age / 30.0, 1),
                        'state': track.state,
                        'det_score': round(track.score, 3)
                    })

            faces_data.sort(key=lambda x: x['track_id'])

            return jsonify({
                'timestamp': time.time(),
                'count': len(faces_data),
                'faces': faces_data
            })

        @self.app.route('/pause', methods=['POST'])
        def pause_stream():
            """Pause stream polling (for frontend use)"""
            with self.pause_lock:
                self.paused = True
            logger.info("Stream polling paused")
            return jsonify({
                'status': 'paused',
                'timestamp': time.time()
            })

        @self.app.route('/resume', methods=['POST'])
        def resume_stream():
            """Resume stream polling (for frontend use)"""
            with self.pause_lock:
                self.paused = False
            logger.info("Stream polling resumed")
            return jsonify({
                'status': 'resumed',
                'timestamp': time.time()
            })

        @self.app.route('/snapshot/download')
        def download_snapshot():
            """Download current frame as JPEG"""
            from flask import send_file
            import io

            with self.frame_lock:
                if self.current_frame is None:
                    return "No frame available", 404

                # Encode frame as JPEG
                ret, buffer = cv2.imencode('.jpg', self.current_frame,
                                          [cv2.IMWRITE_JPEG_QUALITY, 95])
                if not ret:
                    return "Failed to encode frame", 500

            # Convert to BytesIO for sending
            img_io = io.BytesIO(buffer.tobytes())
            img_io.seek(0)
            filename = f"snapshot_{int(time.time())}.jpg"

            return send_file(img_io, mimetype='image/jpeg',
                           as_attachment=True, download_name=filename)

    def set_tracker_reference(self, tracker):
        """
        Set reference to tracker for live face data

        Args:
            tracker: BYTETracker instance with active face tracking data
        """
        self.tracker_ref = tracker
        logger.info("Tracker reference set for web streamer")

    def set_camera_info(self, rtsp_url: str, manufacturer: str = "Hanwha"):
        """
        Set camera information from RTSP URL

        Args:
            rtsp_url: Full RTSP URL (e.g., rtsp://user:pass@192.168.1.100:554/path)
            manufacturer: Camera manufacturer (default: Hanwha)
        """
        import re

        # Parse RTSP URL: rtsp://username:password@ip:port/path
        pattern = r'rtsp://(?:([^:]+):([^@]+)@)?([^:/]+)(?::(\d+))?(/.*)?'
        match = re.match(pattern, rtsp_url)

        if match:
            username, password, ip_address, port, path = match.groups()
            self.camera_info = {
                'rtsp_url': rtsp_url,
                'ip_address': ip_address or 'Unknown',
                'username': username or 'Unknown',
                'port': port or '554',
                'path': path or '/profile2/media.smp',
                'manufacturer': manufacturer
            }
            logger.info(f"Camera info set: {ip_address} ({manufacturer})")
        else:
            logger.warning(f"Could not parse RTSP URL: {rtsp_url}")

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
