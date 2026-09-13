cmake_minimum_required(VERSION 3.15)
project(minimind VERSION 1.0.0 LANGUAGES CXX)

# Use Clang for AArch64 optimizations (required for full TFLite)
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_C_COMPILER "clang")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenCL
find_package(OpenCL REQUIRED)

# Abseil (transitively required by TFLite GPU code)
set(absl_DIR "${CMAKE_SOURCE_DIR}/third_party/absl/lib/cmake/absl")
find_package(absl REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

# ============================================================
# SOURCE FILES (updated paths)
# ============================================================
set(SOURCES
    src/main.cpp
    src/core/config.cpp
    src/calibration/calibration.cpp
    src/calibration/calibration_compute.cpp
    src/detection/level1_detector.cpp
    src/benchmark/benchmark.cpp
    src/detection/roi_processor.cpp
    src/detection/roi_inference.cpp
    src/pipeline/producer.cpp
    src/pipeline/data_sender.cpp
    src/pipeline/consumer.cpp
    src/pipeline/io_worker.cpp
    src/pipeline/led_controller.cpp
)

# ============================================================
# HEADER FILES (for IDE visibility)
# ============================================================
set(HEADERS
    src/core/config.hpp
    src/core/pipeline_data.hpp
    src/calibration/calibration.hpp
    src/calibration/calibration_compute.hpp
    src/detection/level1_detector.hpp
    src/benchmark/benchmark.hpp
    src/detection/roi_processor.hpp
    src/detection/roi_inference.hpp
    src/pipeline/producer.hpp
    src/pipeline/consumer.hpp
    src/pipeline/gpu_queue.hpp
    src/pipeline/frame_buffer.hpp
    src/pipeline/io_worker.hpp
    src/pipeline/led_controller.hpp
)

# ============================================================
# EXECUTABLE
# ============================================================
add_executable(minimind ${SOURCES} ${HEADERS})

# Enable full TFLite Micro for hardware acceleration
target_compile_definitions(minimind PRIVATE
    EI_CLASSIFIER_USE_FULL_TFLITE=1
    DISABLEFLOAT16=1
)

# ============================================================
# EDGE IMPULSE SDK SOURCES
# ============================================================

# 1. Edge Impulse Core (ei_malloc, ei_free, ei_printf, run_classifier)
file(GLOB_RECURSE EI_CORE_SOURCES
    third_party/edge_impulse/sdk/classifier/ei_run_classifier_c.cpp
    third_party/edge_impulse/sdk/dsp/memory.cpp
    third_party/edge_impulse/sdk/dsp/image/processing.cpp
    third_party/edge_impulse/sdk/dsp/kissfft/kiss_fft.cpp
    third_party/edge_impulse/sdk/dsp/kissfft/kiss_fftr.cpp
    third_party/edge_impulse/sdk/dsp/dct/fast-dct-fft.cpp
    third_party/edge_impulse/sdk/porting/posix/ei_classifier_porting.cpp
    third_party/edge_impulse/sdk/porting/posix/debug_log.cpp
)

# 2. Your model source
file(GLOB_RECURSE MODEL_SOURCES
    third_party/edge_impulse/model/tflite/*.cpp
)

# CMSIS-DSP sources (required for full TFLite on AArch64)
file(GLOB_RECURSE CMSIS_SOURCES
    third_party/edge_impulse/sdk/CMSIS/DSP/Source/*.c
)

# Add all Edge Impulse sources to target
target_sources(minimind PRIVATE
    ${EI_CORE_SOURCES}
    ${MODEL_SOURCES}
    ${CMSIS_SOURCES}
)

find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(minimind PRIVATE nlohmann_json::nlohmann_json)
    target_compile_definitions(minimind PRIVATE USE_NLOHMANN_JSON)
    message(STATUS "nlohmann/json found - enabling JSON config support")
else()
    message(STATUS "nlohmann/json not found - using fallback config parsing")
endif()

# Include directories
target_include_directories(minimind PRIVATE
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/third_party/edge_impulse/sdk
    ${CMAKE_SOURCE_DIR}/third_party/edge_impulse/sdk/tensorflow
    ${CMAKE_SOURCE_DIR}/third_party/edge_impulse/sdk/third_party
    ${CMAKE_SOURCE_DIR}/third_party/edge_impulse/model/tflite
    ${CMAKE_SOURCE_DIR}/third_party/edge_impulse/model/parameters
    ${CMAKE_SOURCE_DIR}/third_party/tflite/headers
    ${CMAKE_SOURCE_DIR}/third_party/tflite/headers/flatbuffers/include
    src
    src/core
    src/calibration
    src/detection
    src/benchmark
    src/pipeline
    third_party/mnn/include
)

# MNN libraries
target_link_directories(minimind PRIVATE
    third_party/mnn/lib
    ${CMAKE_SOURCE_DIR}/third_party/tflite/lib
)

target_link_libraries(minimind PRIVATE
    ${OpenCV_LIBS}
    MNN
    ${OpenCL_LIBRARIES}
    -Wl,--no-as-needed
    dl
    rt
    tensorflow-lite
    farmhash
    fft2d_fftsg
    fft2d_fftsg2d
    ruy
    XNNPACK
    microkernels-prod
    cpuinfo
    pthreadpool
    pthread
    kleidiai
    # Abseil (transitively required by TFLite GPU objects)
    absl::strings
    absl::status
    absl::statusor
    absl::str_format
    absl::cord
    absl::cord_internal
    absl::cordz_functions
    absl::cordz_handle
    absl::cordz_info
    absl::cordz_sample_token
    absl::synchronization
    absl::graphcycles_internal
    absl::kernel_timeout_internal
    absl::stacktrace
    absl::symbolize
    absl::debugging_internal
    absl::demangle_internal
    absl::hash
    absl::city
    absl::low_level_hash
    absl::raw_hash_set
    absl::hashtablez_sampler
    absl::time
    absl::time_zone
    absl::civil_time
    absl::int128
    absl::throw_delegate
    absl::base
    absl::log_severity
    absl::raw_logging_internal
    absl::spinlock_wait
    absl::malloc_internal
    absl::bad_optional_access
    absl::bad_variant_access
)

# ============================================================
# CALIBRATION COMPUTE EXECUTABLE
# ============================================================
add_executable(calibration_compute 
    src/calibration/calibration_compute_main.cpp
    src/calibration/calibration_compute.cpp
    src/calibration/calibration.cpp
    src/core/config.cpp
)

target_include_directories(calibration_compute PRIVATE src)

target_link_libraries(calibration_compute 
    ${OpenCV_LIBS} 
    nlohmann_json::nlohmann_json
)

# ============================================================
# COMPILE OPTIONS
# ============================================================
target_compile_options(minimind PRIVATE
    -Wall
    -Wextra
    -O2
    -g
    -march=native
    -mtune=native
)

# ============================================================
# DEBUG OPTION
# ============================================================
option(ENABLE_DEBUG "Enable debug mode outputs" OFF)

if(ENABLE_DEBUG)
    target_compile_definitions(minimind PRIVATE DEBUG_MODE)
endif()
