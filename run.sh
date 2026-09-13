#!/bin/bash
# Textile defect detection — launcher.
# Starts the minimind pipeline and the UI.

set -e

cd "$(dirname "$0")" || exit 1

# MNN / OpenCL runtime paths
export LD_LIBRARY_PATH="$PWD/third_party/mnn/lib:$LD_LIBRARY_PATH"
export RUSTICL_ENABLE=msm

INPUT="${INPUT:-$HOME/ArduinoApps/videoledbridge/frames}"
OUTPUT="${OUTPUT:-output}"
CALIB="${CALIB:-calibration_metrics.json}"

if [ ! -x build/minimind ]; then
    echo "[run.sh] build/minimind not found. Build first:"
    echo "  cmake -S . -B build && cmake --build build -j"
    exit 1
fi

echo "[run.sh] input  = $INPUT"
echo "[run.sh] output = $OUTPUT"
echo "[run.sh] calib  = $CALIB"

cleanup() {
    echo ""
    echo "[run.sh] shutting down..."
    [ -n "$MINIMIND_PID" ] && kill "$MINIMIND_PID" 2>/dev/null
    [ -n "$UI_PID" ] && kill "$UI_PID" 2>/dev/null
    wait 2>/dev/null
}
trap cleanup INT TERM EXIT

# Launch minimind
./build/minimind \
    --input "$INPUT" \
    --output "$OUTPUT" \
    --calibration "$CALIB" \
    --async &
MINIMIND_PID=$!

# Launch UI
python3 ui/ui.py &
UI_PID=$!

# Block until either exits
wait -n 2>/dev/null || wait
