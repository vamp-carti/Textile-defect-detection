#!/bin/bash
# ============================================================
# Minimind - Single Command Launcher
# ============================================================

cd "$(dirname "$0")" || exit

# Force debug OFF
export MINIMIND_DEBUG=0

# Set GPU environment
export RUSTICL_ENABLE=msm
export LD_LIBRARY_PATH="$PWD/cpp_gpu_runtime/lib:$PWD/opencv_lib:$LD_LIBRARY_PATH"

# Check if binary exists
if [ ! -f build/minimind ]; then
    echo "❌ Binary not found. Run make first."
    exit 1
fi

echo "🚀 Starting minimind..."
echo "   Press 'S' in the UI to start detection"
echo "   Press 'E' to export CSV"
echo "   Press 'Q' to quit"

# Launch C++ pipeline in background
./build/minimind --input input/ --output output/ --async &

# Wait a moment for DataSender to start
sleep 2

# Launch Python UI
python3 ui_client.py

# Clean up on exit
pkill minimind 2>/dev/null
