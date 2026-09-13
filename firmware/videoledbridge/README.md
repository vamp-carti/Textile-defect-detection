# 😀 VideoLEDBridge
# videoledbridge — App Lab project

MCU-side companion for the minimind defect detection pipeline. Two jobs:

1. Serves a QR code that pairs a phone as the video source. The phone streams
   frames over WebSocket; App Lab writes them to `frames/`.
2. Bridges pipeline state to the LED matrix on the UNO Q via App Lab's RPC
   layer, so the operator gets a visual cue (arrows / cross / blank) tied to
   defect detection.

## What it does

- Opens a WebSocket camera server (port 8080) and a web page with a QR code.
- Scanning the QR with the Arduino IoT Remote app on a phone pairs it and
  starts frame streaming.
- Writes each incoming frame to `frames/frame_NNN.jpg`.
- Watches `/app/led_state` for changes written by `minimind` and calls the
  MCU over the Arduino Bridge (`led_normal`, `led_defect`, `led_idle`).
- The MCU sketch switches the LED matrix animation accordingly.

## Layout

    app.yaml                 App Lab project manifest
    python/main.py           frame ingestion + state watcher + Bridge calls
    sketch/sketch.ino        LED state machine + RPC handlers
    sketch/frames.h          animation frame arrays
    sketch/sketch.yaml       Arduino sketch config
    assets/                  App Lab web UI (serves the QR pairing page)
    frames/                  runtime output — phone frames land here (gitignored)

## Setup on the board

App Lab expects projects under `~/ArduinoApps/`. Copy or link this project
there before running:

    cp -r <repo>/firmware/videoledbridge ~/ArduinoApps/videoledbridge
    # or, to keep the repo as source of truth:
    ln -s <repo>/firmware/videoledbridge ~/ArduinoApps/videoledbridge

Then open App Lab, select `videoledbridge`, and click Run.

## Pairing the phone

1. Launch the App Lab project. App Lab opens its web UI in a browser at
   `<board-name>.local:7000`.
2. Scan the QR code shown there with the Arduino IoT Remote app on the phone.
3. Once paired, the phone streams frames and `frames/frame_NNN.jpg` files
   start appearing on the board.

The pipeline (`minimind --input ~/ArduinoApps/videoledbridge/frames`) watches
this directory and processes each new frame as it lands.

## Interaction with the pipeline

Two independent channels:

- **Frames in:** phone → App Lab → `frames/frame_NNN.jpg` → `minimind`.
- **State out:** `minimind` → `led_state` → App Lab → MCU RPC → LED matrix.

`minimind` writes one of `NORMAL`, `DEFECT`, `IDLE` to
`~/ArduinoApps/videoledbridge/led_state`. The Python watcher polls that file
and calls the corresponding RPC method on the MCU.

Both paths use the App Lab project directory as the meeting point. The frame
path (`frames/`) and the state file (`led_state`) are the two files that
matter to the pipeline. Everything else in this folder is App Lab
configuration and the MCU sketch.

## Paths referenced by the pipeline

Two hardcoded paths in `src/main.cpp`:

- Input folder: `~/ArduinoApps/videoledbridge/frames` (or wherever you pass
  via `--input`).
- LED state file: `~/ArduinoApps/videoledbridge/led_state` (the
  `LedController` construction).

If you place the App Lab project elsewhere, update both.



