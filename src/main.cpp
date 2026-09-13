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
#include <deque>
#include <cstdio>
#include <sstream>
#include <iomanip>

#include <opencv2/opencv.hpp>

#include "pipeline/data_sender.hpp"
#include "core/config.hpp"
#include "detection/level1_detector.hpp"
#include "detection/roi_processor.hpp"
#include "detection/roi_inference.hpp"
#include "core/pipeline_data.hpp"
#include "pipeline/producer.hpp"
#include "pipeline/consumer.hpp"
#include "pipeline/gpu_queue.hpp"
#include "pipeline/frame_buffer.hpp"
#include "core/defect_report.hpp"
#include "pipeline/io_worker.hpp"
#include "calibration/calibration_compute.hpp"
#include "calibration/calibration.hpp"
#include "pipeline/led_controller.hpp"

namespace fs = std::filesystem;
using namespace minimind;

// These atomics bridge the UI command loop, producer/consumer threads, and
// signal handling without sharing a lock across the long-running workers.
std::atomic<bool> g_pipeline_running{false};  
std::atomic<bool> g_export_requested{false};
std::atomic<bool> g_should_exit{false};
std::atomic<bool> g_recalibrate_requested{false};
std::vector<DefectReport> g_defect_reports;
std::mutex g_defect_mutex;
std::shared_ptr<minimind::LedController> g_led_controller;

std::atomic<bool> g_overshoot_requested{false};
std::atomic<bool> g_recalibration_in_progress{false};

std::mutex        g_overshoot_mutex;
std::string       g_overshoot_file;
int               g_overshoot_frame_id = 0;

std::mutex                               g_reprocess_mutex;
std::deque<std::pair<std::string, bool>> g_reprocess_queue;

// Global pointer for signal handler
minimind::DataSender* g_data_sender = nullptr;

// =====================================================================
// SYSTEM STATS READING
// =====================================================================

struct SystemStats {
    float cpu_usage = 0.0f;
    float cpu_temp = 0.0f;
    float gpu_temp = 0.0f;
    float memory_usage_mb = 0.0f;
};

SystemStats readSystemStats() {
    // Linux exposes the board telemetry through procfs and thermal sysfs; on
    // other platforms the zero-initialized values remain valid placeholders.
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
    
    // GPU temperature (zone7 = gpu-thermal)
    std::ifstream gpu_temp_file("/sys/class/thermal/thermal_zone7/temp");
    if (gpu_temp_file.is_open()) {
        int temp_val;
        gpu_temp_file >> temp_val;
        stats.gpu_temp = temp_val / 1000.0f;
    }
    
    // CPU temperature (average of zone3 + zone4)
    int cpu_temp_sum = 0;
    int cpu_temp_count = 0;
    const char* cpu_zones[] = {
        "/sys/class/thermal/thermal_zone3/temp",
        "/sys/class/thermal/thermal_zone4/temp"
    };
    for (const auto* path : cpu_zones) {
        std::ifstream f(path);
        if (f.is_open()) {
            int t;
            f >> t;
            cpu_temp_sum += t;
            cpu_temp_count++;
        }
    }
    if (cpu_temp_count > 0) {
        stats.cpu_temp = (cpu_temp_sum / cpu_temp_count) / 1000.0f;
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
    // Export takes a snapshot under the report mutex so inference can continue
    // while the CSV is written to disk.

    std::string output_dir = Config::getInstance().production.defect_output_dir;
    if (output_dir.empty()) {
        output_dir = "output/";
    } else if (output_dir.back() != '/') {
        output_dir += "/";
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
            if (g_led_controller) g_led_controller->setState("NORMAL");
            
            // Update UI data
            minimind::UIData ui_data;
            ui_data.status = "RUNNING";
            ui_data.gpu_status = "ONLINE";
            data_sender.updateData(ui_data);
            
        } else if (command == "STOP") {
            std::cout << "\n[UI] Received STOP command - Pausing pipeline" << std::endl;
            g_pipeline_running = false;
            if (g_led_controller) g_led_controller->setState("IDLE");
            
            // Update UI data
            minimind::UIData ui_data;
            ui_data.status = "PAUSED";
            data_sender.updateData(ui_data);
            
        } else if (command == "EXPORT") {
            std::cout << "\n[UI] Received EXPORT command" << std::endl;
            g_export_requested = true;
            
        } else if (command == "RECALIBRATE") {
            std::cout << "\n[UI] Received RECALIBRATE command" << std::endl;
            g_recalibrate_requested = true;

        } else if (command == "QUIT") {
            std::cout << "\n[UI] Received QUIT command" << std::endl;
            g_pipeline_running = false;
            g_should_exit = true;
            if (g_led_controller) g_led_controller->setState("IDLE");
            
            // Auto-save CSV on quit
            std::cout << "[UI] Auto-saving CSV before quit" << std::endl;
            exportDefectReport();
            
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

// Pick the newest image in a folder by modification time.
static std::string findNewestImage(const std::string& folder) {
    namespace fs = std::filesystem;
    std::string newest;
    std::filesystem::file_time_type newest_time{};

    if (!fs::exists(folder) || !fs::is_directory(folder)) {
        return newest;
    }

    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") continue;

        auto t = fs::last_write_time(entry.path());
        if (newest.empty() || t > newest_time) {
            newest = entry.path().string();
            newest_time = t;
        }
    }
    return newest;
}

// Performs a full stop-the-world recalibration.
// If source_file is empty, picks the newest image from input_path.
// Returns true on success, false if the calibration source could not be
// read, computeFromFrame threw, or the detector reload failed.
static bool performRecalibration(
    const std::string& source_file,
    const std::string& input_path,
    Level1Detector& detector,
    UIData& ui_data,
    minimind::DataSender& data_sender)
{
    g_recalibration_in_progress = true;

    // 1. Choose source file
    std::string source = source_file;
    if (source.empty()) {
        source = findNewestImage(input_path);
    }
    if (source.empty()) {
        std::cout << "[Calibration] No source image found in "
                  << input_path << std::endl;
        ui_data.calibration_state = "failed";
        data_sender.updateData(ui_data);
        g_recalibration_in_progress = false;
        return false;
    }
    std::cout << "[Calibration] Source frame: " << source << std::endl;

    // 2. Load, compute
    ui_data.calibration_state = "running";
    ui_data.calibration_progress = 0.0f;
    data_sender.updateData(ui_data);

    cv::Mat gray = cv::imread(source, cv::IMREAD_GRAYSCALE);
    if (gray.empty()) {
        std::cout << "[Calibration] Failed to read " << source << std::endl;
        ui_data.calibration_state = "failed";
        data_sender.updateData(ui_data);
        g_recalibration_in_progress = false;
        return false;
    }

    CalibrationData calib;
    try {
        calib = CalibrationComputer::computeFromFrame(gray,
            [&ui_data, &data_sender](int pct) {
                ui_data.calibration_progress = (float)pct;
                data_sender.updateData(ui_data);
            });
    } catch (const std::exception& e) {
        std::cout << "[Calibration] computeFromFrame failed: "
                  << e.what() << std::endl;
        ui_data.calibration_state = "failed";
        data_sender.updateData(ui_data);
        g_recalibration_in_progress = false;
        return false;
    }

    // 3. Save JSON
    std::string calib_path = Config::getInstance().production.calibration_path;
    if (calib_path.empty()) calib_path = "calibration_metrics.json";
    CalibrationComputer::saveToJson(calib_path, calib);
    std::cout << "[Calibration] Wrote " << calib_path << std::endl;

    // 4. Save source frame with 256x256 box overlay
    {
        std::string out_dir = "output/calibration/";
        std::string mkdir_cmd = "mkdir -p " + out_dir;
        system(mkdir_cmd.c_str());

        int patch_size = 256;
        int cy = gray.rows / 2;
        int cx = gray.cols / 2;
        int y1 = std::max(0, cy - patch_size / 2);
        int x1 = std::max(0, cx - patch_size / 2);
        int y2 = std::min(gray.rows, y1 + patch_size);
        int x2 = std::min(gray.cols, x1 + patch_size);

        cv::Mat vis;
        cv::cvtColor(gray, vis, cv::COLOR_GRAY2BGR);
        cv::rectangle(vis, cv::Point(x1, y1), cv::Point(x2, y2),
                      cv::Scalar(0, 0, 255), 2);

        std::string vis_path = out_dir + "calibration_source.jpg";
        cv::imwrite(vis_path, vis);
        ui_data.calibration_source_path = vis_path;
        std::cout << "[Calibration] Saved source visualization to "
                  << vis_path << std::endl;
    }

    // 5. Reload detector
    if (!detector.reloadCalibration(calib_path)) {
        std::cout << "[Calibration] Detector reload failed" << std::endl;
        ui_data.calibration_state = "failed";
        data_sender.updateData(ui_data);
        g_recalibration_in_progress = false;
        return false;
    }

    ui_data.calibration_state = "done";
    ui_data.calibration_progress = 100.0f;
    data_sender.updateData(ui_data);

    g_recalibration_in_progress = false;
    return true;
}

// =====================================================================
// ASYNC MODE (Producer-Consumer)
// =====================================================================

void runAsyncMode(
    const std::string& input_path,
    const std::string& output_path,
    Level1Detector& detector,
    ROIProcessor& roi_processor,
    ROIInference& roi_inference,
    size_t queue_size,
    minimind::DataSender& data_sender
) {
    (void)output_path;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Reset global flags - pipeline starts paused
    g_pipeline_running = false;
    g_export_requested = false;
    g_should_exit = false;
    
    // Create data object
    minimind::UIData ui_data;
    ui_data.status = "WAITING FOR START";
    data_sender.updateData(ui_data);

    auto gpu_queue = std::make_shared<GPUQueue>(16);  // Fixed batch size of 16
    auto frame_buffer = std::make_shared<FrameBuffer>(5);  // Keep last 5 frames
    PipelineStats stats;

    auto io_worker = std::make_shared<IoWorker>();
    roi_inference.setIoWorker(io_worker.get());

    auto led_controller = std::make_shared<LedController>(
        "/home/arduino/ArduinoApps/videoledbridge/led_state");
    // This path is specific to the Arduino Uno Q App Lab deployment; the LED
    // bridge is optional when frames are supplied directly to the input folder.
    g_led_controller = led_controller;
    led_controller->setState("IDLE");
    
    // Create producer and consumer threads
    std::thread producer(
        producerThread,
        gpu_queue,
        frame_buffer,
        io_worker,
        led_controller,
        std::cref(input_path),
        std::ref(detector),
        std::ref(roi_processor),
        std::ref(roi_inference),
        std::ref(stats),
        std::ref(ui_data)
    );

    std::thread consumer(
        consumerThread,
        gpu_queue,
        frame_buffer,
        io_worker,
        led_controller,
        std::ref(roi_inference),
        std::ref(stats),
        std::ref(ui_data)
    );
    
    auto last_update = std::chrono::steady_clock::now();
    bool should_continue = true;

    std::cout << "[UI] Pipeline initialized. Press 'S' in Python UI to start." << std::endl;

    while (should_continue && !g_should_exit) {
        // Check for commands from Python UI
        checkCommands(data_sender);
        
        if (!g_pipeline_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            if (!g_recalibration_in_progress.load()) {
                ui_data.status = "PAUSED";
                data_sender.updateData(ui_data);
            }
            
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
            ui_data.cpu_temp = sys_stats.cpu_temp;
            ui_data.gpu_temp = sys_stats.gpu_temp;
            ui_data.memory_usage_mb = sys_stats.memory_usage_mb;
            
            // Update pipeline stats
            ui_data.frames_processed = stats.frames_processed.load();
            ui_data.queue_size = 0;
            ui_data.active_count = 0;
            ui_data.total_components = stats.total_components.load();
            ui_data.total_rois = stats.total_rois.load();
            ui_data.total_inferences = stats.total_inferences.load();
            
            if (stats.frames_processed.load() > 0) {
                float avg_time = stats.total_end_to_end_time.load() / 
                                 stats.frames_processed.load();
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
        
        // -------- Manual recalibration (UI button) --------
        // Stop-the-world is required because reloadCalibration rebuilds
        // kernels and preallocated detector buffers used by process().
        if (g_recalibrate_requested) {
            g_recalibrate_requested = false;

            bool was_running = g_pipeline_running.load();
            g_pipeline_running = false;
            gpu_queue->drain();

            ui_data.status = "RECALIBRATING";
            data_sender.updateData(ui_data);

            {
                auto wait_start = std::chrono::steady_clock::now();
                while (gpu_queue->size() > 0) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - wait_start).count();
                    if (elapsed > 5000) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            }

            performRecalibration(/*source=*/"", input_path, detector,
                                 ui_data, data_sender);

            if (was_running) g_pipeline_running = true;
        }

        // -------- Auto recalibration (ROI overshoot) --------
        if (g_overshoot_requested) {
            g_overshoot_requested = false;

            std::string source_file;
            int source_frame_id = 0;
            {
                std::lock_guard<std::mutex> lk(g_overshoot_mutex);
                source_file = g_overshoot_file;
                source_frame_id = g_overshoot_frame_id;
            }

            std::cout << "\n[Overshoot] Auto-recalibrating from frame "
                      << source_frame_id << " (" << source_file << ")\n";

            g_pipeline_running = false;
            gpu_queue->drain();

            ui_data.status = "RECALIBRATING";
            data_sender.updateData(ui_data);

            {
                auto wait_start = std::chrono::steady_clock::now();
                while (gpu_queue->size() > 0) {
                    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - wait_start).count();
                    if (elapsed > 5000) break;
                    std::this_thread::sleep_for(std::chrono::milliseconds(50));
                }
            }

            bool ok = performRecalibration(source_file, input_path, detector,
                                           ui_data, data_sender);

            if (ok) {
                {
                    std::lock_guard<std::mutex> lk(g_reprocess_mutex);
                    g_reprocess_queue.push_back({source_file, true});
                }
                ui_data.status = "PAUSED";
            } else {
                ui_data.status = "CALIBRATION_FAILED";
            }

            ui_data.roi_overshoot = false;
            data_sender.updateData(ui_data);
        }

        // Check if producer and consumer are done
        if (!producer.joinable() && !consumer.joinable()) {
            should_continue = false;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    
    // Clean up threads
    gpu_queue->stop();
    if (producer.joinable()) {
        producer.join();
    }
    if (consumer.joinable()) {
        consumer.join();
    }
    if (io_worker) {
        io_worker->flush();
    }

    // animation stops when threads exit. 
    if (g_led_controller) g_led_controller->setState("IDLE");

    // Auto-save CSV after pipeline completes
    std::cout << "[UI] Pipeline complete - auto-saving CSV" << std::endl;
    exportDefectReport();

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
    // In async mode, the producer watches `input_path` continuously and
    // picks up files as they appear. No up-front glob, and empty folder is
    // a valid startup state. In sync mode, we still need a static list.

    std::vector<cv::String> filenames;
    if (!async_mode) {
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
    }

    // ============================================================
    // RUN PIPELINE
    // ============================================================

    if (async_mode) {
        runAsyncMode(
            input_path,
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
