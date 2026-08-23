#pragma once

#include "core/config.hpp"
#include <iostream>
#include <chrono>
#include <sstream>

namespace minimind {

// ============ DEBUG LOGGING MACROS ============

#define DBG_LOG(msg) \
    do { \
        if (minimind::Config::getInstance().isDebugMode()) { \
            std::cout << "[DEBUG] " << msg << std::endl; \
        } \
    } while(0)

#define DBG_LOG_DETECTION(msg) \
    do { \
        if (minimind::Config::getInstance().debug.log_detections) { \
            std::cout << "[DETECTION] " << msg << std::endl; \
        } \
    } while(0)

#define DBG_LOG_TIMING(msg) \
    do { \
        if (minimind::Config::getInstance().debug.log_timing) { \
            std::cout << "[TIMING] " << msg << std::endl; \
        } \
    } while(0)

// ============ PERFORMANCE TIMING MACROS ============

class ScopedTimer {
public:
    ScopedTimer(const std::string& name, bool enabled = true) 
        : m_name(name), m_enabled(enabled) {
        if (m_enabled) {
            m_start = std::chrono::high_resolution_clock::now();
        }
    }
    
    ~ScopedTimer() {
        if (m_enabled) {
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - m_start);
            std::cout << "[PERF] " << m_name << ": " << duration.count() << " us" << std::endl;
        }
    }
    
    double elapsedMs() const {
        if (!m_enabled) return 0.0;
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - m_start).count();
    }
    
private:
    std::string m_name;
    bool m_enabled;
    std::chrono::high_resolution_clock::time_point m_start;
};

#define SCOPED_TIMER(name) \
    minimind::ScopedTimer timer_##__LINE__(name, minimind::Config::getInstance().isPerformanceTimingEnabled())

#define MEASURE_TIME(name, code) \
    do { \
        if (minimind::Config::getInstance().isPerformanceTimingEnabled()) { \
            auto start = std::chrono::high_resolution_clock::now(); \
            code; \
            auto end = std::chrono::high_resolution_clock::now(); \
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start); \
            std::cout << "[PERF] " << name << ": " << duration.count() << " us" << std::endl; \
        } else { \
            code; \
        } \
    } while(0)

// ============ CONDITIONAL COMPILATION HELPERS ============

// For debug-only code that should be removed in release builds
#ifdef NDEBUG
    #define DEBUG_ONLY(code) 
    #define DEBUG_CODE_BEGIN if(false) {
    #define DEBUG_CODE_END }
#else
    #define DEBUG_ONLY(code) code
    #define DEBUG_CODE_BEGIN if(minimind::Config::getInstance().isDebugMode()) {
    #define DEBUG_CODE_END }
#endif

// ============ CONFIGURATION ACCESS SHORTCUTS ============

inline int DS_FACTOR() { 
    return minimind::Config::getInstance().image.ds_factor; 
}

inline int TARGET_H() { 
    return minimind::Config::getInstance().image.target_h; 
}

inline int TARGET_W() { 
    return minimind::Config::getInstance().image.target_w; 
}

inline int CLOSE_KSIZE_DS() { 
    return minimind::Config::getInstance().morphology.close_ksize_ds; 
}

inline int LINE_LEN_DS() { 
    return minimind::Config::getInstance().morphology.line_len_ds; 
}

inline int MIN_CLUSTER_SIZE_DS() { 
    return minimind::Config::getInstance().cluster.min_cluster_size_ds; 
}

inline double VOID_ENERGY_RATIO() { 
    return minimind::Config::getInstance().cluster.void_energy_ratio; 
}

inline int BG_DS_FACTOR() { 
    return minimind::Config::getInstance().background.bg_ds_factor; 
}

inline int KSIZE_DS_STAGE1() { 
    return minimind::Config::getInstance().background.ksize_ds_stage1; 
}

} // namespace minimind
