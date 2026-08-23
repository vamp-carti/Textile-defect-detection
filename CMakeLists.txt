cmake_minimum_required(VERSION 3.15)
project(minimind VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find OpenCL
find_package(OpenCL REQUIRED)

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
)

# ============================================================
# EXECUTABLE
# ============================================================
add_executable(minimind ${SOURCES} ${HEADERS})

find_package(nlohmann_json QUIET)
if(nlohmann_json_FOUND)
    target_link_libraries(minimind PRIVATE nlohmann_json::nlohmann_json)
    target_compile_definitions(minimind PRIVATE USE_NLOHMANN_JSON)
    message(STATUS "nlohmann/json found - enabling JSON config support")
else()
    message(STATUS "nlohmann/json not found - using fallback config parsing")
endif()

# Include project headers + MNN headers
target_include_directories(minimind PRIVATE
    src
    src/core
    src/calibration
    src/detection
    src/benchmark
    src/pipeline
    cpp_gpu_runtime/include          # MNN headers are here
)

# MNN libraries
target_link_directories(minimind PRIVATE
    cpp_gpu_runtime/lib              # libMNN.so is here
)

# Link libraries
target_link_libraries(minimind PRIVATE
    ${OpenCV_LIBS}
    MNN
    ${OpenCL_LIBRARIES}
    dl
    pthread
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
)

# ============================================================
# DEBUG OPTION
# ============================================================
option(ENABLE_DEBUG "Enable debug mode outputs" OFF)

if(ENABLE_DEBUG)
    target_compile_definitions(minimind PRIVATE DEBUG_MODE)
endif()

# ============================================================
# CUSTOM RUN TARGETS
# ============================================================
add_custom_target(run
    COMMAND $<TARGET_FILE:minimind>
        --input ${CMAKE_SOURCE_DIR}/input
        --output ${CMAKE_SOURCE_DIR}/output_cpp
        --calibration ${CMAKE_SOURCE_DIR}/calibration_metrics.json
        $<$<BOOL:${ENABLE_DEBUG}>:--debug>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS minimind
    USES_TERMINAL
)

add_custom_target(run_async
    COMMAND $<TARGET_FILE:minimind>
        --input ${CMAKE_SOURCE_DIR}/input
        --output ${CMAKE_SOURCE_DIR}/output_cpp
        --calibration ${CMAKE_SOURCE_DIR}/calibration_metrics.json
        --async
        --queue-size 3
        $<$<BOOL:${ENABLE_DEBUG}>:--debug>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS minimind
    USES_TERMINAL
)
