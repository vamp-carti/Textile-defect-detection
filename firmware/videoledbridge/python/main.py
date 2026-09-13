# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
# SPDX-License-Identifier: MPL-2.0

"""
VideoLEDBridge: Merged App
- WebSocket camera ingestion (frames saved to /app/frames/)
- LED matrix state watcher (reads /app/led_state, calls Bridge)
"""

import secrets
import string
import time
import threading
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

from arduino.app_utils import App, Bridge
from arduino.app_bricks.web_ui import WebUI
from arduino.app_peripherals.camera import WebSocketCamera

# ========== CONFIGURATION ==========
FRAME_SAVE_DIR = "/app/frames"
LED_STATE_FILE = "/app/led_state"
CAPTURE_INTERVAL_SECONDS = 5
MAX_FRAMES = 1000
RESOLUTION = (1920, 1080)
WEBSOCKET_PORT = 8080
BLUR_THRESHOLD = 100
# ===================================

Path(FRAME_SAVE_DIR).mkdir(parents=True, exist_ok=True)

# Global state for LED watcher
last_led_state = None

# Global state for capture
frame_counter = 0
saved_frame_count = 0
last_capture_time = time.time()

def generate_secret() -> str:
    characters = string.digits
    return ''.join(secrets.choice(characters) for _ in range(6))

secret = generate_secret()

# ========== SETUP: CAMERA + UI ==========
print("[INFO] Starting WebSocketCamera...")
camera = WebSocketCamera(
    port=WEBSOCKET_PORT,
    secret=secret,
    encrypt=True,
    resolution=RESOLUTION,
    fps=15,
    auto_reconnect=True
)

ui = WebUI()

def on_connect(sid):
    ui.send_message("welcome", {
        "client_name": camera.name,
        "secret": secret,
        "status": camera.status,
        "protocol": camera.protocol,
        "ip": camera.ip,
        "port": camera.port
    })

ui.on_connect(on_connect)
camera.on_status_changed(lambda evt_type, data: ui.send_message(evt_type, data))

# ========== BLUR CHECK ==========
def is_blurry(frame: np.ndarray, threshold: float = BLUR_THRESHOLD) -> bool:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

# ========== FRAME SAVING ==========
def save_frame(frame: np.ndarray):
    global frame_counter, saved_frame_count, last_capture_time

    current_time = time.time()
    if current_time - last_capture_time < CAPTURE_INTERVAL_SECONDS:
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    if laplacian_var < BLUR_THRESHOLD:
        print(f"[INFO] Skipping blurry frame (var={laplacian_var:.1f})")
        return

    last_capture_time = current_time
    frame_counter += 1

    if saved_frame_count >= MAX_FRAMES:
        oldest_files = sorted(Path(FRAME_SAVE_DIR).glob("frame_*.jpg"))
        if oldest_files:
            oldest_files[0].unlink()
            saved_frame_count -= 1

    filename = f"frame_{frame_counter:03d}.jpg"
    filepath = Path(FRAME_SAVE_DIR) / filename

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    try:
        exif = pil_image.getexif()
        if 0x0112 in exif:
            del exif[0x0112]
        pil_image.save(str(filepath), exif=exif)
        success = True
    except (AttributeError, KeyError, ValueError) as e:
        pil_image.save(str(filepath))
        success = True
    except Exception as e:
        print(f"[ERROR] Failed to save with PIL: {e}")
        success = cv2.imwrite(str(filepath), frame)

    if success:
        saved_frame_count += 1
        print(f"[INFO] Saved frame {saved_frame_count}: {filename} (var={laplacian_var:.1f})")
    else:
        print(f"[ERROR] Failed to save frame")

# ========== CAPTURE LOOP ==========
def capture_loop():
    print("[INFO] Capture loop started. Waiting for camera...")

    while not camera.is_started():
        time.sleep(0.1)

    print("[INFO] Camera started. Waiting for iPhone connection...")

    while True:
        try:
            frame = camera.capture()
            if frame is not None and frame.size > 0:
                h, w = frame.shape[:2]
                if h > w:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                save_frame(frame)
            else:
                time.sleep(0.01)
        except Exception as e:
            print(f"[ERROR] In capture loop: {e}")
            time.sleep(0.1)

# ========== LED STATE WATCHER ==========
def led_watch_loop():
    """
    Polls /app/led_state every 100ms.
    On change, calls Bridge into the MCU sketch.
    """
    global last_led_state
    print("[INFO] LED watcher started")

    while True:
        try:
            with open(LED_STATE_FILE) as f:
                current = f.read().strip()
        except FileNotFoundError:
            current = None

        if current and current != last_led_state:
            if current == "NORMAL":
                Bridge.call("led_normal")
                print(f"[INFO] LED -> NORMAL")
            elif current == "DEFECT":
                Bridge.call("led_defect")
                print(f"[INFO] LED -> DEFECT")
            elif current == "IDLE":
                Bridge.call("led_idle")
                print(f"[INFO] LED -> IDLE")
            last_led_state = current

        time.sleep(0.1)

# ========== START ==========
print("=" * 60)
print("VideoLEDBridge - Merged App")
print("=" * 60)
print(f"WebSocket Server: {camera.url}")
print(f"Frames saved to: {FRAME_SAVE_DIR}")
print(f"LED state file: {LED_STATE_FILE}")
print("=" * 60)

# Start camera
camera.start()

# Start capture thread
capture_thread = threading.Thread(target=capture_loop, daemon=True)
capture_thread.start()

# Start LED watcher thread
led_thread = threading.Thread(target=led_watch_loop, daemon=True)
led_thread.start()

# Run App (keeps WebUI alive)
App.run()
