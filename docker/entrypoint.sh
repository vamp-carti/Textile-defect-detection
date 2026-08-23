#!/bin/bash
# Set library path
export LD_LIBRARY_PATH=/app/cpp_gpu_runtime/lib:/app/opencv_lib:$LD_LIBRARY_PATH

# Set default environment variables
export MINIMIND_MODEL_PATH=${MINIMIND_MODEL_PATH:-/app/models/mobilenetv4_conv_small_batch.mnn}
export MINIMIND_OPENCL_BACKEND=${MINIMIND_OPENCL_BACKEND:-/app/cpp_gpu_runtime/lib/libMNN_CL.so}
export MINIMIND_CALIBRATION_PATH=${MINIMIND_CALIBRATION_PATH:-/app/calibration_metrics.json}
export MINIMIND_DEFECT_THRESHOLD=${MINIMIND_DEFECT_THRESHOLD:-0.0092}
export MINIMIND_DEBUG=${MINIMIND_DEBUG:-0}
export MINIMIND_SAVE_DEFECTS=${MINIMIND_SAVE_DEFECTS:-1}

# Run the binary with passed arguments
exec /app/bin/minimind "$@"
