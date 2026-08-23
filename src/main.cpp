#include <unistd.h>
#include <fcntl.h>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>
#include <mutex>
#include <chrono>
#include <thread>
#include <memory>
#include <signal.h>
#include <fstream>
#include <atomic>
#include <cstdio>  // Added for std::remove
#include <sstream>

#include <opencv2/opencv.hpp>

#include "pipeline/data_sender.hpp"
#include "core/config.hpp"
#include "detection/level1_detector.hpp"
#include "detection/roi_processor.hpp"
#include "detection/roi_inference.hpp"
#include "core/pipeline_data.hpp"
#include "pipeline/producer.hpp"
#include "pipeline/consumer.hpp"

struct DefectReport {
    int frame_id;
    int roi_id;
    std::string predicted_class;
    float confidence;
    float defect_probability;
    std::chrono::system_clock::time_point timestamp;
};

namespace fs = std::filesystem;
using namespace minimind;

// Global flags for pipeline control
std::atomic<bool> g_pipeline_running{false};  
std::atomic<bool> g_export_requested{false};
std::atomic<bool> g_should_exit{false};
std::vector<DefectReport> g_defect_reports;
std::mutex g_defect_mutex;

// Global pointer for signal handler
minimind::DataSender* g_data_sender = nullptr;

// =====================================================================
// SYSTEM STATS READING
// =====================================================================

struct SystemStats {
    float cpu_usage = 0.0f;
    float gpu_temp = 0.0f;
    float memory_usage_mb = 0.0f;
};

SystemStats readSystemStats() {
    SystemStats stats;
    
#ifdef __linux__
    // CPU usage
    static long prev_idle = 0, prev_total = 0;
    std::ifstream stat_file("/proc/stat");
    std::string line;
    if (std::getline(stat_file, line)) {
        std::istringstream ss(line);
        std::string cpu;
        long user, nice, system, idle, iowait, irq, softirq, steal;
        ss >> cpu >> user >> nice >> system >> idle >> iowait >> irq >> softirq >> steal;
        long total = user + nice + system + idle + iowait + irq + softirq + steal;
        if (prev_total > 0) {
            long diff_total = total - prev_total;
            long diff_idle = idle - prev_idle;
            if (diff_total > 0) {
                stats.cpu_usage = 100.0f * (1.0f - (float)diff_idle / diff_total);
            }
        }
        prev_idle = idle;
        prev_total = total;
    }
    
    // Temperature
    std::ifstream temp_file("/sys/class/thermal/thermal_zone0/temp");
    if (temp_file.is_open()) {
        int temp_val;
        temp_file >> temp_val;
        stats.gpu_temp = temp_val / 1000.0f;
    }
    
    // Memory
    std::ifstream mem_file("/proc/meminfo");
    long mem_total = 0, mem_available = 0;
    std::string key;
    long value;
    std::string unit;
    while (mem_file >> key >> value >> unit) {
        if (key == "MemTotal:") mem_total = value;
        if (key == "MemAvailable:") mem_available = value;
    }
    if (mem_total > 0) {
        stats.memory_usage_mb = (mem_total - mem_available) / 1024.0f;
    }
#endif
    
    return stats;
}

// =====================================================================
// EXPORT CSV FUNCTION
// =====================================================================
void exportDefectReport() {

    std::string output_dir = Config::getInstance().production.defect_output_dir;
    if (output_dir.empty()) {
        output_dir = "output/";
    }
    
    // Create directory if it doesn't exist
    std::string cmd = "mkdir -p " + output_dir;
    system(cmd.c_str());
    
    std::string filename = output_dir + "defect_report_" + 
        std::to_string(std::time(nullptr)) + ".csv";
    
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "\n[UI] Failed to export CSV to: " << filename << std::endl;
        return;
    }
    
    file << "Frame ID,ROI ID,Predicted Class,Confidence (%),Defect Probability (%),Timestamp\n";
    
    std::lock_guard<std::mutex> lock(g_defect_mutex);
    
    for (const auto& report : g_defect_reports) {
        auto time_t = std::chrono::system_clock::to_time_t(report.timestamp);
        std::string time_str = std::ctime(&time_t);
        time_str.pop_back(); // Remove newline
        
        file << report.frame_id << ","
             << report.roi_id << ","
             << report.predicted_class << ","
             << std::fixed << std::setprecision(2) << report.confidence * 100 << ","
             << std::fixed << std::setprecision(2) << report.defect_probability * 100 << ","
             << time_str << "\n";
    }
    
    file.close();
    std::cout << "\n[UI] Defect report exported to: " << filename 
              << " (" << g_defect_reports.size() << " defects)" << std::endl;
}

void signalHandler(int sig) {
    (void)sig;
    std::cout << "\nShutting down..." << std::endl;
    if (g_data_sender) {
        g_data_sender->stop();
    }
    exit(0);
}

// =====================================================================
// HELP
// =====================================================================

void printHelp() {
    std::cout
        << "Usage: ./minimind "
        << "--input <path> "
        << "--output <path> "
        << "[--calibration <path>] "
        << "[--parity <ref_dir>] "
        << "[--debug] "
        << "[--async] "
        << "[--queue-size <n>]\n";
}

// =====================================================================
// COMMAND HANDLING - No UI dependency
// =====================================================================

void checkCommands(DataSender& data_sender) {
    std::ifstream cmd_file("/tmp/minimind_command.txt");
    if (cmd_file.is_open()) {
        std::string command;
        std::getline(cmd_file, command);
        cmd_file.close();
        
        // Delete the command file after reading
        std::remove("/tmp/minimind_command.txt");
        
        // Process command
        if (command == "START") {
            std::cout << "\n[UI] Received START command - Starting pipeline" << std::endl;
            g_pipeline_running = true;
            
            // Update UI data
            minimind::UIData ui_data;
            ui_data.status = "RUNNING";
            ui_data.gpu_status = "ONLINE";
            data_sender.updateData(ui_data);
            
        } else if (command == "STOP") {
            std::cout << "\n[UI] Received STOP command - Pausing pipeline" << std::endl;
            g_pipeline_running = false;
            
            // Update UI data
            minimind::UIData ui_data;
            ui_data.status = "PAUSED";
            data_sender.updateData(ui_data);
            
        } else if (command == "EXPORT") {
            std::cout << "\n[UI] Received EXPORT command" << std::endl;
            g_export_requested = true;
            
        } else if (command == "QUIT") {
            std::cout << "\n[UI] Received QUIT command" << std::endl;
            g_pipeline_running = false;
            g_should_exit = true;
	   
	   minimind::UIData ui_data;
           ui_data.status = "STOPPED";
           data_sender.updateData(ui_data);
        }
    }
}

// =====================================================================
// SYNC MODE (Original single-threaded pipeline)
// =====================================================================

void runSyncMode(
    const std::vector<cv::String>& filenames,
    const std::string& output_path,
    Level1Detector& detector,
    ROIProcessor& roi_processor,
    ROIInference& roi_inference,
    minimind::DataSender& data_sender
) {
    auto total_start = std::chrono::high_resolution_clock::now();
    std::cout << "Running in SYNC mode (single-threaded)...\n\n";

    // Create data object
    minimind::UIData ui_data;
    ui_data.status = "WAITING FOR START";
    ui_data.gpu_status = "ONLINE";
    
    int frame_id = 0;
    (void)output_path; // Suppress unused parameter warning

    for (const auto& filename : filenames) {
        // Check for commands
        checkCommands(data_sender);
        
        if (g_should_exit) break;
        
        // Wait for start command
        while (!g_pipeline_running && !g_should_exit) {
            std::cout << "[UI] Waiting for START command..." << std::endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            checkCommands(data_sender);
        }
        
        if (g_should_exit) break;
        
        ++frame_id;
        fs::path p(filename);

        cv::Mat bgr = cv::imread(filename, cv::IMREAD_COLOR);
        if (bgr.empty()) {
            std::cerr << "Failed to read image: " << filename << "\n";
            continue;
        }

        cv::Mat rgb, gray;
        cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
        cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);

        cv::Mat result_mask = detector.process(gray, frame_id);
        const auto& components = detector.getLastComponents();

        std::string out_file = (fs::path(output_path) / p.filename()).string();
        cv::imwrite(out_file, result_mask);

        std::vector<ROIResult> rois;
        if (!components.empty()) {
            rois = roi_processor.process(rgb, components);
        }

        std::vector<ROIInferenceResult> inference_results;
        if (!rois.empty()) {
            inference_results = roi_inference.processBatch(rois, rgb);
        }

        // Update data
        ui_data.frames_processed = frame_id;
        ui_data.total_components = components.size();
        ui_data.total_rois = rois.size();
        ui_data.total_inferences = inference_results.size();
        
        // Update FPS roughly
        auto now = std::chrono::high_resolution_clock::now();
        static auto last_time = now;
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_time).count();
        if (elapsed > 0) {
            ui_data.fps = 1000.0f / elapsed;
        }
        last_time = now;
        
        data_sender.updateData(ui_data);

        if (Config::getInstance().isDebugMode()) {
            std::cout << "[Frame " << frame_id << "] " << p.filename().string()
                      << " | Components: " << components.size()
                      << " | ROIs: " << rois.size() << "\n";
        }
    }

    ui_data.status = "COMPLETE";
    data_sender.updateData(ui_data);

    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(
        total_end - total_start
    ).count();

    std::cout << "\n============================================================\n";
    std::cout << "SYNC PIPELINE WALL-CLOCK TIME: " << total_time_ms << " ms\n";
    std::cout << "Frames processed: " << filenames.size() << "\n";
    std::cout << "Avg time per frame: " << total_time_ms / filenames.size() << " ms\n";
    std::cout << "============================================================\n";
}

// =====================================================================
// ASYNC MODE (Producer-Consumer)
// =====================================================================

void runAsyncMode(
    const std::vector<cv::String>& filenames,
    const std::string& output_path,
    Level1Detector& detector,
    ROIProcessor& roi_processor,
    ROIInference& roi_inference,
    size_t queue_size,
    minimind::DataSender& data_sender
) {
    (void)output_path;
    (void)filenames;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Reset global flags - pipeline starts paused
    g_pipeline_running = false;
    g_export_requested = false;
    g_should_exit = false;
    
    // Create data object
    minimind::UIData ui_data;
    ui_data.status = "WAITING FOR START";
    data_sender.updateData(ui_data);

    auto queue = std::make_shared<PipelineQueue>(queue_size);
    PipelineStats stats;

    // Create producer and consumer threads
    std::thread producer(
        producerThread,
        queue,
        std::cref(filenames),
        std::ref(detector),
        std::ref(roi_processor),
        std::ref(stats)
    );

    std::thread consumer(
        consumerThread,
        queue,
        std::ref(roi_inference),
        std::ref(stats)
    );
    
    auto last_update = std::chrono::steady_clock::now();
    bool should_continue = true;

    std::cout << "[UI] Pipeline initialized. Press 'S' in Python UI to start." << std::endl;

    while (should_continue && !g_should_exit) {
        // Check for commands from Python UI
        checkCommands(data_sender);
        
        // Check if pipeline should run or be paused
        if (!g_pipeline_running) {
            // Pipeline is paused - just wait and update UI
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            // Update status to show paused
            ui_data.status = "PAUSED";
            data_sender.updateData(ui_data);
            
            // Check if threads are still alive
            if (!producer.joinable() && !consumer.joinable()) {
                should_continue = false;
                break;
            }
            continue;
        }
        
        // Pipeline is running - update stats
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - last_update).count();
        
        if (elapsed >= 100) {
            // Read system stats
            SystemStats sys_stats = readSystemStats();
            ui_data.cpu_usage = sys_stats.cpu_usage;
            ui_data.gpu_temp = sys_stats.gpu_temp;
            ui_data.memory_usage_mb = sys_stats.memory_usage_mb;
            
            // Update pipeline stats
            ui_data.frames_processed = stats.frames_processed.load();
            ui_data.queue_size = queue->size();
            ui_data.active_count = queue->activeCount();
            ui_data.total_components = stats.total_components.load();
            ui_data.total_rois = stats.total_rois.load();
            ui_data.total_inferences = stats.total_inferences.load();
            
            if (stats.frames_processed.load() > 0) {
                float avg_time = stats.total_level1_time.load() + 
                                stats.total_roi_time.load() + 
                                stats.total_inference_time.load();
                avg_time /= stats.frames_processed.load();
                ui_data.avg_time_ms = avg_time;
                ui_data.fps = 1000.0f / avg_time;
            }
            
            ui_data.status = "RUNNING";
            data_sender.updateData(ui_data);
            last_update = now;
        }
        
        // Handle export request
        if (g_export_requested) {
            std::cout << "[UI] Export requested - saving CSV" << std::endl;
            exportDefectReport();
            std::cout << "[UI] Export complete" << std::endl;
            g_export_requested = false;
        }
        
        // Check if producer and consumer are done
        if (!producer.joinable() && !consumer.joinable()) {
            should_continue = false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Clean up threads
    if (producer.joinable()) {
        producer.join();
    }
    if (consumer.joinable()) {
        consumer.join();
    }

    ui_data.status = "COMPLETE";
    data_sender.updateData(ui_data);
    
    auto total_end = std::chrono::high_resolution_clock::now();
    double total_time_ms = std::chrono::duration<double, std::milli>(
        total_end - total_start
    ).count();

    stats.printSummary();
}

// =====================================================================
// MAIN
// =====================================================================

int main(int argc, char** argv) {
    // ============================================================
    // ARGUMENT PARSING
    // ============================================================

    std::string input_path;
    std::string output_path;
    std::string calibration_path;
    std::string parity_dir;
    bool debug_mode = false;
    bool async_mode = false;
    size_t queue_size = 0;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "--help") {
            printHelp();
            return 0;

        } else if (arg == "--input" && i + 1 < argc) {
            input_path = argv[++i];

        } else if (arg == "--output" && i + 1 < argc) {
            output_path = argv[++i];

        } else if (arg == "--calibration" && i + 1 < argc) {
            calibration_path = argv[++i];

        } else if (arg == "--parity" && i + 1 < argc) {
            parity_dir = argv[++i];

        } else if (arg == "--debug") {
            debug_mode = true;

        } else if (arg == "--async") {
            async_mode = true;

        } else if (arg == "--queue-size" && i + 1 < argc) {
            queue_size = std::stoul(argv[++i]);
            if (queue_size < 1) queue_size = 1;
            if (queue_size > 10) queue_size = 10;
        }
    }

    if (input_path.empty() || output_path.empty()) {
        std::cerr << "Error: Input and output paths are required.\n";
        printHelp();
        return 1;
    }

    // ============================================================
    // LOAD CONFIG
    // ============================================================

    Config::getInstance().loadFromFile("config.json");
    Config::getInstance().loadFromEnv();

    if (debug_mode) {
        Config::getInstance().setDebugMode(true);
    }

    // ============================================================
    // RESOLVE PARAMETERS (CLI > config > default)
    // ============================================================

    if (calibration_path.empty()) {
        calibration_path = Config::getInstance().production.calibration_path;
    }

    if (queue_size == 0) {
        queue_size = Config::getInstance().production.queue_size;
    }

    const std::string& model_path = Config::getInstance().production.model_path;
    const std::string& opencl_backend_path = Config::getInstance().production.opencl_backend_path;
    float defect_threshold = Config::getInstance().production.defect_threshold;

    // ============================================================
    // START DATA SENDER
    // ============================================================
    std::cout << "Starting DataSender on port 9999..." << std::endl;
    minimind::DataSender data_sender(9999);
    g_data_sender = &data_sender;
    signal(SIGINT, signalHandler);
    
    data_sender.start();

    // Initialize UI data
    minimind::UIData ui_data;
    ui_data.status = "WAITING FOR START";
    data_sender.updateData(ui_data);

    // ============================================================
    // INITIALIZE COMPONENTS
    // ============================================================

    Level1Detector detector(calibration_path, debug_mode);
    if (!detector.initialize()) {
        std::cerr << "Level1Detector initialization failed.\n";
        return 1;
    }

    ROIProcessor roi_processor;

    ROIInference roi_inference(
        model_path,
        opencl_backend_path,
        defect_threshold
    );

    // ============================================================
    // OUTPUT DIRECTORY
    // ============================================================

    if (!fs::exists(output_path)) {
        fs::create_directories(output_path);
    }

    // ============================================================
    // COLLECT INPUT FILES
    // ============================================================

    std::vector<cv::String> filenames;

    if (fs::is_directory(input_path)) {
        cv::glob(input_path + "/*.png", filenames);
        if (filenames.empty()) {
            cv::glob(input_path + "/*.jpg", filenames);
        }
    } else {
        filenames.push_back(input_path);
    }

    if (filenames.empty()) {
        std::cerr << "No input images found.\n";
        return 1;
    }

    // ============================================================
    // RUN PIPELINE
    // ============================================================

    if (async_mode) {
        runAsyncMode(
            filenames,
            output_path,
            detector,
            roi_processor,
            roi_inference,
            queue_size,
            data_sender
        );
    } else {
        runSyncMode(
            filenames,
            output_path,
            detector,
            roi_processor,
            roi_inference,
            data_sender
        );
    }

    // ============================================================
    // KEYBOARD HANDLING (Fallback if Python UI not available)
    // ============================================================
    
    std::cout << "\nPress 'Q' to quit\n";
    
    char key;
    while (true) {
        key = getchar();
        if (key == 'q' || key == 'Q') {
            break;
        }
    }
    
    data_sender.stop();

    // ============================================================
    // BENCHMARK REPORTS
    // ============================================================

    if (Config::getInstance().isPerformanceTimingEnabled()) {
        detector.getBenchmarkTimer().report();
        roi_processor.getBenchmarkTimer().report();
        roi_inference.getBenchmarkTimer().report();
    }

    return 0;
}
