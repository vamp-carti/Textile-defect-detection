#pragma once

#include <string>
#include <cstdint>
#include <atomic>

namespace minimind {

// Forward declaration for internal use
struct ConfigInternal;

/**
 * @brief Central configuration management for the Minimind system
 * 
 * All configuration parameters are organized by functional category.
 * Debug flags can be toggled at runtime without recompilation.
 */
class Config {
public:
    // Singleton access
    static Config& getInstance();
    
    // Load configuration from file or environment
    bool loadFromFile(const std::string& path = "config.json");
    bool loadFromEnv();
    
    // Debug mode control - runtime togglable
    void setDebugMode(bool enabled);
    bool isDebugMode() const;
    
    // Performance timing control
    void setPerformanceTiming(bool enabled);
    bool isPerformanceTimingEnabled() const;
    
    // ============ ORIGINAL PARAMETERS (preserved exactly) ============
    
    // Image processing parameters - DS_FACTOR, TARGET_H, TARGET_W
    struct ImageParams {
        int ds_factor = 2;          // DS_FACTOR
        int target_h = 1080;        // TARGET_H
        int target_w = 1920;        // TARGET_W
    } image;
    
    // Closing and line parameters - CLOSE_KSIZE_DS, LINE_LEN_DS
    struct MorphologyParams {
        int close_ksize_ds = 11;    // CLOSE_KSIZE_DS
        int line_len_ds = 7;        // LINE_LEN_DS
    } morphology;
    
    // Cluster parameters - MIN_CLUSTER_SIZE_DS, VOID_ENERGY_RATIO
    struct ClusterParams {
        int min_cluster_size_ds = 67;      // MIN_CLUSTER_SIZE_DS
        double void_energy_ratio = 0.18;   // VOID_ENERGY_RATIO
    } cluster;
    
    // Background parameters - BG_DS_FACTOR, KSIZE_DS_STAGE1
    struct BackgroundParams {
        int bg_ds_factor = 4;       // BG_DS_FACTOR
        int ksize_ds_stage1 = 31;   // KSIZE_DS_STAGE1
    } background;
    
    // ============ DEBUG AND TIMING CONTROLS ============
    
    struct DebugControls {
        bool enabled = false;           // Master debug switch
        bool log_detections = false;    // Log each detection
        bool log_timing = false;        // Log timing information
        bool draw_debug_boxes = false;  // Draw debug visualization
        bool save_intermediate = false; // Save intermediate results
    } debug;
    
    struct TimingControls {
        bool enabled = false;           // Master timing switch
        bool measure_inference = false; // Measure inference time
        bool measure_preprocessing = false; // Measure preproc time
        bool measure_postprocessing = false; // Measure postproc time
    } timing;
    
    // ============ PRODUCTION SETTINGS ============
    
    struct ProductionSettings {
        int num_workers = 4;            // Number of pipeline workers
        size_t queue_size = 100;        // Pipeline queue size
        bool enable_warmup = true;      // Enable tensor warmup
        int warmup_iterations = 10;     // Number of warmup iterations
        std::string log_file = "minimind.log";
	bool save_defects = false;        // Save defect ROIs to disk
        std::string defect_output_dir = "defects";  //  Directory for defect images
	std::string model_path = "mobilenetv4_conv_small_batch.mnn";      
        std::string opencl_backend_path = "cpp_gpu_runtime/lib/libMNN_CL.so"; 
        float defect_threshold = 0.0092f;                                
        std::string calibration_path = "calibration_metrics.json";
    } production;
    
    // Reset all parameters to default values
    void resetToDefaults();
    
    // Validate current configuration
    bool validate() const;
    
private:
    Config() = default;
    ~Config() = default;
    Config(const Config&) = delete;
    Config& operator=(const Config&) = delete;
    
    // Internal state
    std::atomic<bool> m_debug_mode{false};
    std::atomic<bool> m_timing_enabled{false};
    
    // Helper to parse JSON config
    bool parseJsonConfig(const std::string& json_string);
};

} // namespace minimind
