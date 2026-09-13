#!/usr/bin/env python3
"""
Minimind Web UI Server - Debug Version
Every step is logged for clarity
"""

import socket
import json
import time
import threading
import base64
import os
import glob
import re
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

# The UI is intentionally self-contained: this server serves the page and the
# browser uses the same process's command/status endpoints below.
# ============================================================
# HTML TEMPLATE (minimal for testing)
# ============================================================
INDEX_HTML = '''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Minimind Debug</title>
    <style>
        body { background: #0d0d0d; color: #e0e0e0; font-family: monospace; padding: 20px; }
        /* CHANGED: stat label and value share the same size (24px) */
        .stat { margin: 10px 0; font-size: 24px; }
        .value { font-size: inherit; font-weight: bold; }
        .green { color: #2ecc71; }
        .red { color: #e74c3c; }
        .yellow { color: #f1c40f; }
        .orange { color: #e67e22; }
        /* CHANGED: buttons +30% */
        button { padding: 13px 26px; margin: 5px; cursor: pointer; font-size: 21px; }
        .btn-start { background: #2ecc71; color: #fff; border: none; }
        .btn-stop { background: #e74c3c; color: #fff; border: none; }
        .btn-quit { background: #7f1d1d; color: #fff; border: none; }
        .btn-calib { background: #0e7490; color: #fff; border: none; position: relative; }
        /* CHANGED: debug log text 14 -> 18px */
        .debug-log { background: #1a1a1a; padding: 12px; margin-top: 20px; max-height: 300px; overflow-y: auto; font-size: 18px; }
        .debug-entry { border-bottom: 1px solid #2a2a2a; padding: 4px 0; }
        .timestamp { color: #666; }

        .layout { display: flex; gap: 24px; align-items: flex-start; }
        .left-col { flex: 1 1 auto; min-width: 0; }
        .right-col { flex: 0 0 380px; }

        /* CHANGED: distinct background for right-side panel vs debug log */
        .panel { background: #151515; border: 1px solid #2a2a2a; padding: 15px; border-radius: 4px; margin-bottom: 16px; }
        .panel h3 { margin-top: 0; }

        .calib-wrap { position: relative; display: inline-block; }
        .calib-ring { position: absolute; inset: 0; border-radius: 4px; pointer-events: none; }
        .calib-progress-text { font-size: 14px; color: #94a3b8; margin-left: 8px; }
        .calib-status { font-size: 15px; margin-top: 8px; color: #94a3b8; }
        .calib-status.done { color: #2ecc71; }
        .calib-status.failed { color: #e74c3c; }
        .calib-status.running { color: #f1c40f; }
    </style>
</head>
<body>
    <!-- CHANGED: heading +20% (28 -> 34px) -->
    <h1 style="font-size: 34px; margin-bottom: 24px;">🔍 Minimind UI</h1>

    <div class="stat">
        <div>Status: <span id="statusText" class="value yellow">WAITING</span></div>
    </div>
    <div class="stat">
        <div>FPS: <span id="fpsVal" class="value green">0.0</span></div>
    </div>
    <div class="stat">
        <div>Frames: <span id="framesVal" class="value">0</span></div>
    </div>
    <div class="stat">
        <div>Defects: <span id="defectsVal" class="value red">0</span></div>
    </div>
    <div class="stat">
        <div>Last Defect: <span id="lastDefectVal" class="value">None</span></div>
    </div>
    <div class="stat">
        <div>CPU Temp: <span id="cpuTempVal" class="value yellow">0.0</span>°C</div>
    </div>
    <div class="stat">
        <div>GPU Temp: <span id="gpuTempVal" class="value orange">0.0</span>°C</div>
    </div>
    <div class="stat">
        <div>Memory: <span id="memVal" class="value">0</span> MB</div>
    </div>

    <div class="layout">
      <div class="left-col">
        <div style="margin: 20px 0;">
            <button class="btn-start" onclick="sendCommand('START')">▶ START</button>
            <button class="btn-stop" onclick="sendCommand('STOP')">⏹ STOP</button>
            <button class="btn-quit" onclick="sendCommand('QUIT')">✕ QUIT</button>
            <button onclick="exportAndDownload()" style="background:#3498db;color:#fff;border:none;">📊 EXPORT CSV</button>
            <button onclick="clearLog()">🗑 Clear Log</button>
        </div>

        <div style="margin: 20px 0;">
            <button onclick="toggleDefectPreview()" id="defectPreviewBtn" style="background:#9b59b6;color:#fff;border:none;padding:13px 26px;cursor:pointer;font-size:21px;">
                👁 Show Defects
            </button>
        </div>

        <div id="defectPreview" style="display:none; margin: 20px 0; background:#1a1a1a; padding:15px; border-radius:4px;">
            <h3 style="margin-top:0;color:#e74c3c;">Detected Defects</h3>
            <div id="defectList" style="max-height:300px; overflow-y:auto;">
                <div style="color:#444;padding:10px;">No defects detected yet</div>
            </div>
        </div>
      </div> <!-- CHANGED: /.left-col added (was missing) -->

      <div class="right-col">
        <div class="panel">
          <h3 style="margin-top:0;color:#0e7490;">Calibration</h3>

          <div class="calib-wrap">
            <button class="btn-calib" id="calibBtn" onclick="sendCommand('RECALIBRATE')">
              🎯 Recalibrate
            </button>
            <span class="calib-progress-text" id="calibProgressText"></span>
          </div>

          <div class="calib-status" id="calibStatus">idle</div>

          <div style="margin-top: 14px;">
            <button id="calibPreviewBtn" onclick="toggleCalibPreview()"
                    style="background:#0e7490;color:#fff;border:none;padding:10px 20px;cursor:pointer;font-size:18px;">
              👁 Show Calibration Source
            </button>
          </div>

          <div id="calibPreview" style="display:none; margin-top: 12px;">
            <img id="calibImage" style="max-width:100%; border-radius:4px;" src="" />
            <div id="calibCaption" style="color:#aaa; margin-top:8px; font-size:15px;"></div>
          </div>
        </div>
      </div> <!-- /.right-col -->
    </div> <!-- /.layout -->

    <!-- Image Modal -->
    <div id="imageModal" style="display:none; position:fixed; top:0; left:0; right:0; bottom:0; background:rgba(0,0,0,0.9); z-index:1000; justify-content:center; align-items:center; flex-direction:column;">
        <span onclick="closeModal()" style="position:absolute; top:20px; right:30px; font-size:30px; color:#fff; cursor:pointer;">&times;</span>
        <img id="modalImage" style="max-width:90vw; max-height:80vh; border-radius:8px;" src="" />
        <div id="modalCaption" style="color:#aaa; margin-top:12px; font-size:14px;"></div>
    </div>

    <div class="debug-log" id="debugLog">
        <div class="debug-entry">⏳ Ready...</div>
    </div>

    <script>
        let logCount = 0;
        let defects = [];
        let calibPreviewVisible = false;
        let lastCalibState = 'idle';
        let lastOvershootState = false;

        function addLog(msg, color = '#888') {
            logCount++;
            const log = document.getElementById('debugLog');
            const entry = document.createElement('div');
            entry.className = 'debug-entry';
            const time = new Date().toLocaleTimeString();
            entry.innerHTML = `<span class="timestamp">[${time}]</span> <span style="color:${color}">${msg}</span>`;
            log.appendChild(entry);
            log.scrollTop = log.scrollHeight;
            if (logCount > 100) {
                log.removeChild(log.firstChild);
            }
        }

        function clearLog() {
            document.getElementById('debugLog').innerHTML = '';
            logCount = 0;
            addLog('🗑 Log cleared');
        }

        function exportAndDownload() {
            addLog('📊 Export + download requested...', '#3498db');
            sendCommand('EXPORT');
            setTimeout(() => {
                addLog('⬇ Downloading CSV...', '#16a085');
                window.location.href = '/export_csv?t=' + Date.now();
            }, 600);
        }

        function sendCommand(cmd) {
            addLog(`📤 Sending command: ${cmd}`, '#3498db');
            fetch('/command', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ command: cmd })
            })
            .then(r => r.json())
            .then(data => addLog(`✅ Command response: ${JSON.stringify(data)}`, '#2ecc71'))
            .catch(err => addLog(`❌ Command error: ${err}`, '#e74c3c'));
        }

        function updateUI(data) {
            lastData = data;
            document.getElementById('statusText').textContent = data.status || 'UNKNOWN';
            document.getElementById('fpsVal').textContent = (data.fps || 0).toFixed(2);
                        // ---- Overshoot alert (transition-driven) ----
            if (data.roi_overshoot !== undefined) {
                const nowOvershoot = !!data.roi_overshoot;
                if (nowOvershoot && !lastOvershootState) {
                    addLog(
                        `⚠️ ROI overshoot on frame ${data.roi_overshoot_frame_id} — recalibrating...`,
                        '#e67e22'
                    );
                } else if (!nowOvershoot && lastOvershootState) {
                    addLog('✅ Recalibration done — awaiting START', '#2ecc71');
                }
                lastOvershootState = nowOvershoot;
            }
            document.getElementById('framesVal').textContent = data.frames_processed || 0;
            document.getElementById('cpuTempVal').textContent = (data.cpu_temp || 0).toFixed(1);
            document.getElementById('gpuTempVal').textContent = (data.gpu_temp || 0).toFixed(1);
            document.getElementById('memVal').textContent = (data.memory_usage_mb || 0).toFixed(0);

            if (data.recent_defects && Array.isArray(data.recent_defects)) {
                defects = data.recent_defects;

                // Unbounded counter, distinct from the capped preview list
                const totalCount = (data.total_defects !== undefined)
                    ? data.total_defects
                    : defects.length;
                document.getElementById('defectsVal').textContent = totalCount;

                if (defects.length > 0) {
                    const last = defects[defects.length - 1];
                    document.getElementById('lastDefectVal').textContent = 
                        `${last.predicted_class} (${(last.confidence*100).toFixed(1)}%)`;
                }
                // Refresh defect list if visible
                if (defectPreviewVisible) {
                    renderDefectList();
                }
            }

            if (data.calibration_state !== undefined) {
                lastCalibState = data.calibration_state;
                const statusEl = document.getElementById('calibStatus');
                const progText = document.getElementById('calibProgressText');
                const btn = document.getElementById('calibBtn');

                statusEl.textContent = data.calibration_state;
                statusEl.className = 'calib-status ' + data.calibration_state;

                if (data.calibration_state === 'running') {
                    const pct = Math.round(data.calibration_progress || 0);
                    progText.textContent = pct + '%';
                    btn.disabled = true;
                    btn.style.opacity = 0.7;
                } else {
                    progText.textContent = '';
                    btn.disabled = false;
                    btn.style.opacity = 1;
                }

                if (data.calibration_state === 'done' &&
                    data.calibration_source_path &&
                    calibPreviewVisible) {
                    refreshCalibImage(data.calibration_source_path);
                }
            }

            const statusEl = document.getElementById('statusText');
            if (data.status === 'RUNNING') {
                statusEl.style.color = '#2ecc71';
            } else if (data.status === 'RECALIBRATING') {
                statusEl.style.color = '#3498db';
            } else if (data.status === 'CALIBRATION_FAILED') {
                statusEl.style.color = '#e74c3c';
            } else if (data.status === 'PAUSED') {
                statusEl.style.color = '#f39c12';
            } else {
                statusEl.style.color = '#e74c3c';
            }
        }

        addLog('🚀 Starting polling every 1000ms...', '#3498db');
        setInterval(() => {
            fetch('/data?t=' + Date.now())
                .then(r => r.json())
                .then(data => {
                    if (data.frames_processed > 0 || data.status === 'RUNNING') {
                        updateUI(data);
                    }
                })
                .catch(err => addLog(`⚠️ Poll error: ${err}`, '#f39c12'));
        }, 1000);

        let defectPreviewVisible = false;

        function toggleDefectPreview() {
            defectPreviewVisible = !defectPreviewVisible;
            const panel = document.getElementById('defectPreview');
            const btn = document.getElementById('defectPreviewBtn');

            if (defectPreviewVisible) {
                panel.style.display = 'block';
                btn.textContent = '✕ Hide Defects';
                renderDefectList();
            } else {
                panel.style.display = 'none';
                btn.textContent = '👁 Show Defects';
            }
        }

        function renderDefectList() {
            const list = document.getElementById('defectList');

            if (!defects || defects.length === 0) {
                list.innerHTML = '<div style="color:#444;padding:10px;">No defects detected yet</div>';
                return;
            }

            let html = '';
            for (let i = defects.length - 1; i >= 0; i--) {
                const d = defects[i];
                const conf = ((d.confidence || 0) * 100).toFixed(1);
                html += `
                    <div onclick="showDefectImage(${d.frame_id})" style="
                        padding: 8px 12px;
                        margin: 4px 0;
                        background: #2a2a2a;
                        border-radius: 4px;
                        cursor: pointer;
                        display: flex;
                        justify-content: space-between;
                        border: 1px solid transparent;
                        transition: all 0.2s;
                    " onmouseover="this.style.borderColor='#9b59b6'" onmouseout="this.style.borderColor='transparent'">
                        <span>
                            <span style="color:#666;">#${d.frame_id}</span>
                            ROI ${d.roi_id}: <span style="color:#e74c3c;font-weight:bold;">${d.predicted_class || 'unknown'}</span>
                        </span>
                        <span style="color:#2ecc71;">${conf}%</span>
                    </div>
                `;
            }
            list.innerHTML = html;
        }

        function showDefectImage(frameId) {
            console.log('Fetching frame image:', frameId);
            fetch('/frame_image?frame_id=' + frameId)
                .then(r => r.json())
                .then(data => {
                    if (data.image) {
                        document.getElementById('modalImage').src = 'data:image/jpeg;base64,' + data.image;
                        document.getElementById('modalCaption').textContent = 'Frame ' + frameId;
                        document.getElementById('imageModal').style.display = 'flex';
                    } else {
                        console.warn('No image for frame', frameId);
                    }
                })
                .catch(err => console.error('Image fetch error:', err));
        }

        function closeModal() {
            document.getElementById('imageModal').style.display = 'none';
        }

        function toggleCalibPreview() {
            calibPreviewVisible = !calibPreviewVisible;
            const panel = document.getElementById('calibPreview');
            const btn = document.getElementById('calibPreviewBtn');
            if (calibPreviewVisible) {
                panel.style.display = 'block';
                btn.textContent = '✕ Hide Calibration Source';
                const src = lastData && lastData.calibration_source_path;
                if (src) refreshCalibImage(src);
            } else {
                panel.style.display = 'none';
                btn.textContent = '👁 Show Calibration Source';
            }
        }

        function refreshCalibImage(path) {
            fetch('/calibration_image?t=' + Date.now())
                .then(r => r.json())
                .then(data => {
                    if (data.image) {
                        document.getElementById('calibImage').src =
                            'data:image/jpeg;base64,' + data.image;
                        document.getElementById('calibCaption').textContent =
                            'Source: ' + (path || '');
                    }
                })
                .catch(() => {});
        }

        let lastData = {};
        fetch('/data?t=' + Date.now()).then(r => r.json()).then(d => { lastData = d; updateUI(d); }).catch(() => {});

    </script>
</body>
</html>
'''

# ============================================================
# DEBUG DATA FETCHER
# ============================================================
class DataFetcher:
    def __init__(self, host='localhost', port=9999):
        self.host = host
        self.port = port
        self.latest_data = {}
        self.running = True
        self.connected = False
        self.lock = threading.Lock()
        self.total_messages = 0
        self.reconnect_count = 0
        self.last_received_time = 0

    def update_data(self, data):
        with self.lock:
            self.latest_data = data
            self.total_messages += 1
            self.last_received_time = time.time()

    def get_data(self):
        with self.lock:
            return self.latest_data.copy()

    def fetch_loop(self):
        """Connect to DataSender and read JSON messages"""
        print("[FETCHER] Thread started")
        
        while self.running:
            self.reconnect_count += 1
            
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(5.0)  # 5 second timeout
                sock.connect((self.host, self.port))
                self.connected = True
                print(f"[FETCHER] ✅ CONNECTED to DataSender")
                print(f"[FETCHER] Socket timeout: {sock.gettimeout()}s")

                buffer = ""
                connection_cycles = 0
                
                while self.running:
                    try:
                        chunk = sock.recv(8192).decode('utf-8')
                        
                        if not chunk:
                            print("[FETCHER] Server closed connection")
                            break
                        
                        buffer += chunk
                        
                        while '\n' in buffer:
                            line, buffer = buffer.split('\n', 1)
                            line = line.strip()
                            
                            if not line:
                                continue
                            
                            try:
                                data = json.loads(line)
                                self.update_data(data)
                            except json.JSONDecodeError as e:
                                print(f"[FETCHER] JSON parse error: {e}")
                                
                    except socket.timeout:
                        print("[FETCHER] ⏱️ Socket timeout (no data received)")
                        # Check if we've been disconnected
                        continue
                        
                    except ConnectionError as e:
                        print(f"[FETCHER] ❌ Connection error: {e}")
                        break
                        
                    except Exception as e:
                        print(f"[FETCHER] ❌ Unexpected error: {e}")
                        break

                sock.close()
                self.connected = False
                print("[FETCHER] 🔌 Disconnected")
                
            except ConnectionRefusedError:
                print(f"[FETCHER] ❌ Connection refused - DataSender not running?")
                time.sleep(1)
                
            except Exception as e:
                print(f"[FETCHER] ❌ Connection error: {e}")
                time.sleep(1)
            
            # Wait before reconnecting
            if self.running:
                print("[FETCHER] ⏳ Waiting 1 second before reconnect...")
                time.sleep(1)


# ============================================================
# HTTP REQUEST HANDLER
# ============================================================
class RequestHandler(SimpleHTTPRequestHandler):
    fetcher = None
    request_count = 0

    def do_GET(self):
        self.request_count += 1
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(INDEX_HTML.encode('utf-8'))
            return

        if path == '/data':
            try:
                data = {}
                if RequestHandler.fetcher:
                    data = RequestHandler.fetcher.get_data()
                    data['_connected'] = RequestHandler.fetcher.connected
                    data['_total_messages'] = RequestHandler.fetcher.total_messages
                    data['_last_update'] = RequestHandler.fetcher.last_received_time

                response = json.dumps(data)
                body = response.encode('utf-8')

                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.send_header('Cache-Control', 'no-cache')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                return

            except Exception as e:
                print(f"[HTTP] /data error: {e}")
                return

        if path == '/debug':
            debug_info = {
                'fetcher_connected': RequestHandler.fetcher.connected if RequestHandler.fetcher else False,
                'fetcher_running': RequestHandler.fetcher.running if RequestHandler.fetcher else False,
                'total_messages': RequestHandler.fetcher.total_messages if RequestHandler.fetcher else 0,
                'reconnect_count': RequestHandler.fetcher.reconnect_count if RequestHandler.fetcher else 0,
                'last_update': RequestHandler.fetcher.last_received_time if RequestHandler.fetcher else 0,
                'latest_data': RequestHandler.fetcher.latest_data if RequestHandler.fetcher else {},
                'request_count': self.request_count
            }
            
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(debug_info, indent=2).encode('utf-8'))
            return

        if path == '/frame_image':
            qs = parse_qs(parsed.query)
            frame_id = qs.get('frame_id', [None])[0]
            
            if not frame_id:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Missing frame_id"}')
                return
            
            try:
                frame_int = int(frame_id)
                # Look for the saved frame image
                pattern = f"output/frames/frame_{frame_int:06d}_full.jpg"
                
                if os.path.exists(pattern):
                    with open(pattern, 'rb') as f:
                        img_data = base64.b64encode(f.read()).decode('utf-8')
                    
                    response = {
                        'image': img_data,
                        'frame_id': frame_int
                    }
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.end_headers()
                    self.wfile.write(json.dumps(response).encode('utf-8'))
                    return
                else:
                    self.send_response(404)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(b'{"error": "Frame image not found"}')
                    return
                    
            except ValueError:
                self.send_response(400)
                self.end_headers()
                self.wfile.write(b'{"error": "Invalid frame_id"}')
                return

        if path == '/calibration_image':
            img_path = "output/calibration/calibration_source.jpg"
            if os.path.exists(img_path):
                with open(img_path, 'rb') as f:
                    img_data = base64.b64encode(f.read()).decode('utf-8')
                response = {'image': img_data}
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Access-Control-Allow-Origin', '*')
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))
                return
            else:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error": "No calibration image"}')
                return

        if path == '/export_csv':
            import glob
            candidates = glob.glob("output/defect_report_*.csv")
            if not candidates:
                self.send_response(404)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(b'{"error": "No CSV available."}')
                return

            newest = max(candidates, key=os.path.getmtime)
            with open(newest, 'rb') as f:
                data = f.read()

            self.send_response(200)
            self.send_header('Content-Type', 'text/csv')
            self.send_header('Content-Disposition',
                             f'attachment; filename="{os.path.basename(newest)}"')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if path == '/command':
            # POST only
            self.send_response(405)
            self.end_headers()
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b'Not found')

    def do_POST(self):
        if self.path == '/command':
            
            try:
                content_length = int(self.headers.get('Content-Length', 0))
                body = self.rfile.read(content_length)
                data = json.loads(body)
                command = data.get('command', '')
                
                if command:
                    print(f"[COMMAND] Received: {command}")
                    
                    # Write to command file
                    with open('/tmp/minimind_command.txt', 'w') as f:
                        f.write(command)
                    
                    print(f"[COMMAND] ✅ Written to /tmp/minimind_command.txt")
                    
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({'status': 'ok', 'command': command}).encode('utf-8'))
                    return
                    
            except Exception as e:
                print(f"[HTTP] ❌ POST error: {e}")

        self.send_response(400)
        self.end_headers()
        self.wfile.write(b'{"error": "Bad request"}')

    def log_message(self, format, *args):
        # Already logging everything
        pass


# ============================================================
# MAIN
# ============================================================
def main():
    print("\n" + "="*70)
    print("  🚀 MINIMIND WEB UI - DEBUG VERSION")
    print("="*70)
    print()
    
    # Create fetcher
    fetcher = DataFetcher(host='localhost', port=9999)
    RequestHandler.fetcher = fetcher
    
    # Start fetcher thread
    fetcher_thread = threading.Thread(target=fetcher.fetch_loop, daemon=True)
    fetcher_thread.start()
    print("[MAIN] DataFetcher thread started")
    
    # Start HTTP server
    port = 8081
    server = HTTPServer(('0.0.0.0', port), RequestHandler)
    print(f"[MAIN] HTTP Server running on http://localhost:{port}")
    print("[MAIN] Press Ctrl+C to stop")
    print("="*70)
    print()
    print("📊 DEBUG ENDPOINTS:")
    print("  - http://localhost:8081/        → UI Dashboard")
#    print("  - http://localhost:8081/data    → Current data")
#    print("  - http://localhost:8081/debug   → Fetcher status")
    print()
    print("📝 Check the terminal for detailed logs")
    print("="*70)
    print()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")
        fetcher.running = False
        server.shutdown()
        print("[MAIN] Done")


if __name__ == "__main__":
    main()
