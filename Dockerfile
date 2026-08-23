# Syntax directive required for BuildKit cache mounts
# syntax=docker/dockerfile:1

# ==========================================
# STAGE 1: BASE RUNTIME & DEPENDENCIES
# ==========================================
FROM debian:trixie AS base

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    LD_LIBRARY_PATH=/app/cpp_gpu_runtime/lib:/app/opencv_lib:$LD_LIBRARY_PATH

# Install system runtime dependencies
RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    libstdc++6 \
    libgomp1 \
    libjpeg62-turbo \
    libpng16-16t64 \
    libtiff6 \
    libwebp7 \
    libzstd1 \
    liblzma5 \
    libopenexr-3-1-30 \
    libopencv-dev \
    nlohmann-json3-dev \
    ocl-icd-libopencl1 \
    && rm -rf /var/lib/apt/lists/*

# Create necessary directories
RUN mkdir -p /app/input /app/output /app/defects

# ==========================================
# STAGE 2: PRODUCTION RUNTIME (prod)
# ==========================================
FROM base AS prod

# Copy binary
COPY build/minimind /app/bin/minimind
COPY build/calibration_compute /app/bin/calibration_compute

# Copy runtime libraries
COPY cpp_gpu_runtime/lib/ /app/cpp_gpu_runtime/lib/
COPY opencv_lib/ /app/opencv_lib/

# Copy config files
COPY config.json /app/
COPY calibration_metrics.json /app/

# Copy model
COPY models/ /app/models/

# Copy entrypoint script
COPY docker/entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Create start script that runs calibration first
RUN echo '#!/bin/bash\n\
# Run calibration if calibration.png exists\n\
if [ -f /app/input/calibration.png ]; then\n\
    echo "========================================="\n\
    echo "Running calibration from /app/input/calibration.png..."\n\
    echo "========================================="\n\
    /app/bin/calibration_compute /app/input/calibration.png /app/calibration_metrics.json\n\
    echo "Calibration complete! JSON saved to /app/calibration_metrics.json"\n\
    echo "========================================="\n\
else\n\
    echo "Warning: /app/input/calibration.png not found. Using existing calibration_metrics.json"\n\
fi\n\
\n\
# Then run the main application\n\
exec /app/entrypoint.sh "$@"\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

ENTRYPOINT ["/app/start.sh"]
CMD ["--help"]
