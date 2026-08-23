#include "config.hpp"
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <iostream>

// Use nlohmann/json if available, otherwise fallback to manual parsing
#ifdef USE_NLOHMANN_JSON
#include <nlohmann/json.hpp>
#endif

namespace minimind {

// ============ Singleton Implementation ============

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

// ============ Debug Mode Control ============

void Config::setDebugMode(bool enabled) {
    m_debug_mode = enabled;
    debug.enabled = enabled;
    
    // If debug is enabled, also enable some sensible debug defaults
    if (enabled) {
        debug.log_detections = true;
        debug.log_timing = true;
    }
}

bool Config::isDebugMode() const {
    return m_debug_mode.load();
}

// ============ Performance Timing Control ============

void Config::setPerformanceTiming(bool enabled) {
    m_timing_enabled = enabled;
    timing.enabled = enabled;
    
    if (enabled) {
        timing.measure_inference = true;
        timing.measure_preprocessing = true;
        timing.measure_postprocessing = true;
    }
}

bool Config::isPerformanceTimingEnabled() const {
    return m_timing_enabled.load();
}

// ============ Configuration Loading ============

bool Config::loadFromFile(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        // Not an error if file doesn't exist - use defaults
        return false;
    }
    
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string content = buffer.str();
    
    // If JSON support is available, parse JSON
#ifdef USE_NLOHMANN_JSON
    return parseJsonConfig(content);
#else
    // Manual parse or warn
    std::cerr << "Warning: JSON support not enabled, using default config" << std::endl;
    return false;
#endif
}

bool Config::loadFromEnv() {
    bool loaded = false;
    const char* val = nullptr;
    
    // Image parameters
    if ((val = std::getenv("MINIMIND_DS_FACTOR"))) {
        image.ds_factor = std::atoi(val);
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_TARGET_H"))) {
        image.target_h = std::atoi(val);
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_TARGET_W"))) {
        image.target_w = std::atoi(val);
        loaded = true;
    }
    
    // Morphology parameters
    if ((val = std::getenv("MINIMIND_CLOSE_KSIZE_DS"))) {
        morphology.close_ksize_ds = std::atoi(val);
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_LINE_LEN_DS"))) {
        morphology.line_len_ds = std::atoi(val);
        loaded = true;
    }
    
    // Cluster parameters
    if ((val = std::getenv("MINIMIND_MIN_CLUSTER_SIZE_DS"))) {
        cluster.min_cluster_size_ds = std::atoi(val);
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_VOID_ENERGY_RATIO"))) {
        cluster.void_energy_ratio = std::atof(val);
        loaded = true;
    }
    
    // Background parameters
    if ((val = std::getenv("MINIMIND_BG_DS_FACTOR"))) {
        background.bg_ds_factor = std::atoi(val);
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_KSIZE_DS_STAGE1"))) {
        background.ksize_ds_stage1 = std::atoi(val);
        loaded = true;
    }
    
    // Debug and timing
    if ((val = std::getenv("MINIMIND_DEBUG"))) {
        setDebugMode(std::strcmp(val, "1") == 0 || 
                    std::strcmp(val, "true") == 0 ||
                    std::strcmp(val, "yes") == 0);
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_TIMING"))) {
        setPerformanceTiming(std::strcmp(val, "1") == 0 || 
                            std::strcmp(val, "true") == 0 ||
                            std::strcmp(val, "yes") == 0);
        loaded = true;
    }
    
    // Production settings
    if ((val = std::getenv("MINIMIND_NUM_WORKERS"))) {
        production.num_workers = std::atoi(val);
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_WARMUP"))) {
        production.enable_warmup = std::strcmp(val, "1") == 0 || 
                                  std::strcmp(val, "true") == 0 ||
                                  std::strcmp(val, "yes") == 0;
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_SAVE_DEFECTS"))) {    
        production.save_defects = std::strcmp(val, "1") == 0 || 
                                 std::strcmp(val, "true") == 0 ||
                                 std::strcmp(val, "yes") == 0;
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_DEFECT_DIR"))) {   
        production.defect_output_dir = val;
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_MODEL_PATH"))) {
        production.model_path = val;
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_OPENCL_BACKEND"))) {
        production.opencl_backend_path = val;
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_DEFECT_THRESHOLD"))) {
        production.defect_threshold = std::atof(val);
        loaded = true;
    }
    if ((val = std::getenv("MINIMIND_CALIBRATION_PATH"))) {
        production.calibration_path = val;
        loaded = true;
    }
    return loaded;
}

// ============ JSON Parsing (if available) ============

#ifdef USE_NLOHMANN_JSON
bool Config::parseJsonConfig(const std::string& json_string) {
    try {
        nlohmann::json j = nlohmann::json::parse(json_string);
        
        // Image parameters
        if (j.contains("image")) {
            auto& img = j["image"];
            if (img.contains("ds_factor")) image.ds_factor = img["ds_factor"];
            if (img.contains("target_h")) image.target_h = img["target_h"];
            if (img.contains("target_w")) image.target_w = img["target_w"];
        }
        
        // Morphology parameters
        if (j.contains("morphology")) {
            auto& morph = j["morphology"];
            if (morph.contains("close_ksize_ds")) morphology.close_ksize_ds = morph["close_ksize_ds"];
            if (morph.contains("line_len_ds")) morphology.line_len_ds = morph["line_len_ds"];
        }
        
        // Cluster parameters
        if (j.contains("cluster")) {
            auto& clust = j["cluster"];
            if (clust.contains("min_cluster_size_ds")) cluster.min_cluster_size_ds = clust["min_cluster_size_ds"];
            if (clust.contains("void_energy_ratio")) cluster.void_energy_ratio = clust["void_energy_ratio"];
        }
        
        // Background parameters
        if (j.contains("background")) {
            auto& bg = j["background"];
            if (bg.contains("bg_ds_factor")) background.bg_ds_factor = bg["bg_ds_factor"];
            if (bg.contains("ksize_ds_stage1")) background.ksize_ds_stage1 = bg["ksize_ds_stage1"];
        }
        
        // Debug controls
        if (j.contains("debug")) {
            auto& dbg = j["debug"];
            if (dbg.contains("enabled")) setDebugMode(dbg["enabled"]);
            if (dbg.contains("log_detections")) debug.log_detections = dbg["log_detections"];
            if (dbg.contains("log_timing")) debug.log_timing = dbg["log_timing"];
            if (dbg.contains("draw_debug_boxes")) debug.draw_debug_boxes = dbg["draw_debug_boxes"];
            if (dbg.contains("save_intermediate")) debug.save_intermediate = dbg["save_intermediate"];
        }
        
        // Timing controls
        if (j.contains("timing")) {
            auto& tm = j["timing"];
            if (tm.contains("enabled")) setPerformanceTiming(tm["enabled"]);
            if (tm.contains("measure_inference")) timing.measure_inference = tm["measure_inference"];
            if (tm.contains("measure_preprocessing")) timing.measure_preprocessing = tm["measure_preprocessing"];
            if (tm.contains("measure_postprocessing")) timing.measure_postprocessing = tm["measure_postprocessing"];
        }
        
        // Production settings
        if (j.contains("production")) {
            auto& prod = j["production"];
            if (prod.contains("num_workers")) production.num_workers = prod["num_workers"];
            if (prod.contains("queue_size")) production.queue_size = prod["queue_size"];
            if (prod.contains("enable_warmup")) production.enable_warmup = prod["enable_warmup"];
            if (prod.contains("warmup_iterations")) production.warmup_iterations = prod["warmup_iterations"];
            if (prod.contains("log_file")) production.log_file = prod["log_file"];
	    if (prod.contains("save_defects")) production.save_defects = prod["save_defects"];     
            if (prod.contains("defect_output_dir")) production.defect_output_dir = prod["defect_output_dir"];
	    if (prod.contains("model_path")) production.model_path = prod["model_path"];
            if (prod.contains("opencl_backend_path")) production.opencl_backend_path = prod["opencl_backend_path"];
            if (prod.contains("defect_threshold")) production.defect_threshold = prod["defect_threshold"];
            if (prod.contains("calibration_path")) production.calibration_path = prod["calibration_path"];
        }
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON config: " << e.what() << std::endl;
        return false;
    }
}
#endif

// ============ Reset and Validation ============

void Config::resetToDefaults() {
    // Reset all parameters to their default values (as defined in the structs)
    image = ImageParams();
    morphology = MorphologyParams();
    cluster = ClusterParams();
    background = BackgroundParams();
    debug = DebugControls();
    timing = TimingControls();
    production = ProductionSettings();
    
    m_debug_mode = false;
    m_timing_enabled = false;
}

bool Config::validate() const {
    // Validate image parameters
    if (image.ds_factor < 1 || image.ds_factor > 10) {
        std::cerr << "Invalid DS_FACTOR: " << image.ds_factor << " (must be 1-10)" << std::endl;
        return false;
    }
    if (image.target_h < 100 || image.target_h > 4000) {
        std::cerr << "Invalid TARGET_H: " << image.target_h << " (must be 100-4000)" << std::endl;
        return false;
    }
    if (image.target_w < 100 || image.target_w > 4000) {
        std::cerr << "Invalid TARGET_W: " << image.target_w << " (must be 100-4000)" << std::endl;
        return false;
    }
    
    // Validate morphology parameters
    if (morphology.close_ksize_ds < 1 || morphology.close_ksize_ds % 2 == 0) {
        std::cerr << "Invalid CLOSE_KSIZE_DS: " << morphology.close_ksize_ds 
                  << " (must be odd and >= 1)" << std::endl;
        return false;
    }
    if (morphology.line_len_ds < 1 || morphology.line_len_ds > 50) {
        std::cerr << "Invalid LINE_LEN_DS: " << morphology.line_len_ds 
                  << " (must be 1-50)" << std::endl;
        return false;
    }
    
    // Validate cluster parameters
    if (cluster.min_cluster_size_ds < 1) {
        std::cerr << "Invalid MIN_CLUSTER_SIZE_DS: " << cluster.min_cluster_size_ds 
                  << " (must be >= 1)" << std::endl;
        return false;
    }
    if (cluster.void_energy_ratio < 0.0 || cluster.void_energy_ratio > 1.0) {
        std::cerr << "Invalid VOID_ENERGY_RATIO: " << cluster.void_energy_ratio 
                  << " (must be 0.0-1.0)" << std::endl;
        return false;
    }
    
    // Validate background parameters
    if (background.bg_ds_factor < 1 || background.bg_ds_factor > 10) {
        std::cerr << "Invalid BG_DS_FACTOR: " << background.bg_ds_factor 
                  << " (must be 1-10)" << std::endl;
        return false;
    }
    if (background.ksize_ds_stage1 < 1 || background.ksize_ds_stage1 % 2 == 0) {
        std::cerr << "Invalid KSIZE_DS_STAGE1: " << background.ksize_ds_stage1 
                  << " (must be odd and >= 1)" << std::endl;
        return false;
    }
    
    return true;
}

} // namespace minimind
