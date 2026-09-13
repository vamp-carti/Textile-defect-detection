#include "pipeline/producer.hpp"
#include <filesystem>
#include <iostream>
#include <chrono>
#include <thread>
#include <deque>
#include <mutex>
#include <iomanip>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include "pipeline/gpu_queue.hpp"
#include "pipeline/frame_buffer.hpp"
#include "roi_inference.hpp"
#include "pipeline/io_worker.hpp"
#include "core/config.hpp"
#include "pipeline/led_controller.hpp"

extern std::atomic<bool> g_pipeline_running;
extern std::atomic<bool> g_should_exit;
extern std::atomic<bool> g_overshoot_requested;
extern std::mutex        g_overshoot_mutex;
extern std::string       g_overshoot_file;
extern int               g_overshoot_frame_id;

extern std::mutex                               g_reprocess_mutex;
extern std::deque<std::pair<std::string, bool>> g_reprocess_queue;

namespace minimind {

namespace {

double msSince(const std::chrono::high_resolution_clock::time_point& start) {
    return std::chrono::duration<double, std::milli>(
        std::chrono::high_resolution_clock::now() - start
    ).count();
}

// Returns all image files in `folder` whose mtime is strictly greater than
// `since`, sorted by mtime ascending. The watermark avoids filename parsing or
// filesystem watchers; the caller advances it after successful processing.
static std::vector<std::string> findNewFilesSince(
    const std::string& folder,
    std::filesystem::file_time_type since)
{
    namespace fs = std::filesystem;
    std::vector<std::pair<fs::file_time_type, std::string>> candidates;

    if (!fs::exists(folder) || !fs::is_directory(folder)) return {};

    for (const auto& entry : fs::directory_iterator(folder)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        if (ext != ".png" && ext != ".jpg" && ext != ".jpeg") continue;

        auto t = fs::last_write_time(entry.path());
        // A file that cannot yet be decoded is left below the watermark and is
        // retried on the next poll, which covers partially-written input files.
        if (t > since) {
            candidates.emplace_back(t, entry.path().string());
        }
    }

    std::sort(candidates.begin(), candidates.end(),
              [](const auto& a, const auto& b) { return a.first < b.first; });

    std::vector<std::string> result;
    result.reserve(candidates.size());
    for (auto& kv : candidates) result.push_back(std::move(kv.second));
    return result;
}

bool loadFrameData(
    IoWorker& io_worker,
    const std::string& filename,
    int frame_id,
    std::shared_ptr<FrameData>& data
) {
    cv::Mat bgr = io_worker.wait_for_frame(filename);
    if (bgr.empty()) {
        if (Config::getInstance().isDebugMode()) {
            std::cerr << "[Producer] Failed to read: " << filename << "\n";
        }
        return false;
    }

    data = std::make_shared<FrameData>();
    data->frame_id = frame_id;
    data->filename = filename;

    cv::cvtColor(bgr, data->grayscale_image, cv::COLOR_BGR2GRAY);
    cv::cvtColor(bgr, data->rgb_image, cv::COLOR_BGR2RGB);

    return true;
}

} // anonymous namespace

void producerThread(
    std::shared_ptr<GPUQueue> gpu_queue,
    std::shared_ptr<FrameBuffer> frame_buffer,
    std::shared_ptr<IoWorker> io_worker,
    std::shared_ptr<LedController> led_controller,
    const std::string& input_folder,
    Level1Detector& detector,
    ROIProcessor& roi_processor,
    ROIInference& cpu_inference,
    PipelineStats& stats,
    UIData& ui_data
) {
    bool debug_mode = Config::getInstance().isDebugMode();

    if (debug_mode) {
        std::cout << "[Producer] Thread started. Processing " 
                  << input_folder << "\n";
    }

    namespace fs = std::filesystem;

    int frame_id = 0;

    // Files with mtime strictly greater than this are considered "new".
    // Initialized to now so pre-existing files are picked up on first poll.
    fs::file_time_type last_seen_mtime = fs::file_time_type::clock::now();

    constexpr int kPollMs = 200;

    while (!g_should_exit) {

        // Gate: pause when STOP is active
        if (!g_pipeline_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            continue;
        }

        // Check the reprocess queue first. Files injected here come from
        // the main loop after an auto-recalibration; they take priority
        // over newly-arrived files.
        std::string reprocess_path;
        bool is_reprocess_retry = false;
        {
            std::lock_guard<std::mutex> lk(g_reprocess_mutex);
            if (!g_reprocess_queue.empty()) {
                auto front = g_reprocess_queue.front();
                g_reprocess_queue.pop_front();
                reprocess_path = front.first;
                is_reprocess_retry = front.second;
            }
        }

        std::vector<std::string> new_files;
        if (!reprocess_path.empty()) {
            new_files.push_back(reprocess_path);
        } else {
            new_files = findNewFilesSince(input_folder, last_seen_mtime);
        }

        if (new_files.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(kPollMs));
            continue;
        }

        io_worker->request_decode(new_files[0]);

        for (size_t idx = 0; idx < new_files.size(); ++idx) {
            if (g_should_exit) break;

            // Gate: pause mid-batch
            while (!g_pipeline_running && !g_should_exit) {
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
            if (g_should_exit) break;

            const std::string& filename = new_files[idx];

        ++frame_id;

        auto frame_start_tp = std::chrono::high_resolution_clock::now();

        // Signal that a new frame has started (triggers processing of pending ROIs)
        gpu_queue->markFrameStarted(frame_id);

        auto load_start_tp = std::chrono::high_resolution_clock::now();
        std::shared_ptr<FrameData> data;
        if (!loadFrameData(*io_worker, filename, frame_id, data)) {
            // Decode failed (likely a partially-written file). Leave
            // last_seen_mtime where it is so this file is retried on the
            // next poll. Abandon the rest of this batch.
            if (debug_mode) {
                std::cout << "[Producer] Decode failed for " << filename
                          << " — will retry\n";
            }
            break;
        }
        auto load_end_tp = std::chrono::high_resolution_clock::now();

        // Advance the watermark NOW — the file has been read and its
        // content captured. Any later branch (CPU defect, GPU push,
        // no-components) must not re-read this same file.
        try {
            last_seen_mtime = fs::last_write_time(filename);
        } catch (const fs::filesystem_error&) {
            // File vanished between read and this call.
        }

        data->producer_start_ms = std::chrono::duration<double, std::milli>(
            frame_start_tp.time_since_epoch()
        ).count();

        // Prefetch next file's decode BEFORE heavy work (Level1).
        if (idx + 1 < new_files.size()) {
            io_worker->request_decode(new_files[idx + 1]);
        }

        if (debug_mode) {
            std::cout << "[Producer] Frame " << frame_id 
                      << ": " << filename << "\n";
        }

        auto level1_start = std::chrono::high_resolution_clock::now();

        detector.process(data->grayscale_image, data->frame_id);
        data->components = detector.getLastComponents();

        auto level1_end = std::chrono::high_resolution_clock::now();
        data->level1_time_ms = std::chrono::duration<double, std::milli>(
            level1_end - level1_start
        ).count();

        data->releaseGrayscale();
        data->level1_done = true;

        // Print per-stage benchmark breakdown for every frame
        if (debug_mode) {
            std::cout << "  [Level1 Benchmark] Frame " << frame_id << ":" << std::endl;
            detector.getBenchmarkTimer().report();
        }

                if (!data->components.empty()) {
            auto roi_start = std::chrono::high_resolution_clock::now();

            data->rois = roi_processor.process(
                data->rgb_image,
                data->components
            );

            auto roi_end = std::chrono::high_resolution_clock::now();
            data->roi_time_ms = std::chrono::duration<double, std::milli>(
                roi_end - roi_start
            ).count();

            data->roi_done = true;

            // Cap at 20 ROIs
            if (data->rois.size() > 20) {
                std::cout << "[Producer] ALERT: Frame " << frame_id
                          << " has " << data->rois.size()
                          << " ROIs, RECALIBRATE\n";

                // If this is a retry of an overshooting frame after
                // auto-recalibration, do NOT re-trigger. Cap and continue.
                if (!is_reprocess_retry) {
                    {
                        std::lock_guard<std::mutex> lk(g_overshoot_mutex);
                        g_overshoot_file = filename;
                        g_overshoot_frame_id = frame_id;
                    }
                    g_overshoot_requested = true;

                    ui_data.roi_overshoot = true;
                    ui_data.roi_overshoot_frame_id = frame_id;
                } else {
                    std::cout << "[Producer] Overshoot on retry — capping, "
                              << "not re-triggering\n";
                }

                data->rois.resize(20);
            }

                // CPU checks the highest-priority four ROIs inline so obvious
                // defects can finish without waiting for a GPU batch. Four is a
                // tuned split between fast early rejection and GPU batching.
            bool defect_found = false;
            int cpu_roi_count = std::min(4, (int)data->rois.size());

            auto cpu_start_tp = std::chrono::high_resolution_clock::now();

            std::chrono::high_resolution_clock::time_point defect_detected_tp;
            bool defect_tp_captured = false;

            for (int i = 0; i < cpu_roi_count; i++) {
                auto result = cpu_inference.inferCPU(data->rois[i]);
                if (result.defect) {
                    defect_detected_tp = std::chrono::high_resolution_clock::now();
                    defect_tp_captured = true;

                    DefectImages imgs = cpu_inference.saveDefectFrame(data->rois[i], result, data->rgb_image, "CPU");
                    if (!imgs.roi_display.empty())  io_worker->enqueue_write(std::move(imgs.roi_display),  imgs.roi_path);
                    if (!imgs.full_display.empty()) io_worker->enqueue_write(std::move(imgs.full_display), imgs.full_path);
                    
                    // Update UI data with defect
                    ui_data.has_defect = true;
                    ui_data.last_defect.frame_id = data->frame_id;
                    ui_data.last_defect.roi_id = data->rois[i].box.number;
                    ui_data.last_defect.predicted_class = result.predicted_label;
                    ui_data.last_defect.confidence = result.confidence;
                    ui_data.last_defect.defect_probability = result.defect_probability;
                    
                    // Add to rolling defect list for UI
                    ui_data.addDefect(
                        data->frame_id,
                        data->rois[i].box.number,
                        result.predicted_label,
                        result.confidence,
                        result.defect_probability
                    );

                    if (g_pipeline_running.load()) {
                        led_controller->setState("DEFECT");
                    }

                    defect_found = true;
                    break;
                }
            }

            auto cpu_end_tp = std::chrono::high_resolution_clock::now();

            // A CPU hit is terminal for this frame: discard queued work and
            // preserve the low-latency defect path.
            if (defect_found) {
                if (debug_mode) {
                    std::cout << "  [Producer] CPU DEFECT FOUND - discarding " 
                              << (data->rois.size() - cpu_roi_count) 
                              << " remaining ROIs\n";
                }
                data->rois.clear();
                data->inference_done = true;
                data->releaseRGB();
                stats.frames_processed++;
                
                double load_ms = std::chrono::duration<double, std::milli>(
                    load_end_tp - load_start_tp
                ).count();
                double total_ms = defect_tp_captured
                    ? std::chrono::duration<double, std::milli>(
                          defect_detected_tp - frame_start_tp).count()
                    : msSince(frame_start_tp);
                double cpu_ms = std::chrono::duration<double, std::milli>(
                    cpu_end_tp - cpu_start_tp
                ).count();

                stats.total_components += data->components.size();
                stats.total_rois += data->rois.size();
                stats.total_level1_time.store(stats.total_level1_time.load() + (float)data->level1_time_ms);
                stats.total_roi_time.store(stats.total_roi_time.load() + (float)data->roi_time_ms);
                stats.total_load_time.store(stats.total_load_time.load() + (float)load_ms);
                stats.total_cpu_inference_time.store(stats.total_cpu_inference_time.load() + (float)cpu_ms);
                stats.total_end_to_end_time.store(stats.total_end_to_end_time.load() + (float)total_ms);

                if (debug_mode) {
                    std::cout << "  [Producer] TIMING Frame " << frame_id
                              << " | Load: " << std::fixed << std::setprecision(1) << load_ms << " ms"
                              << " | Level1: " << data->level1_time_ms << " ms"
                              << " | ROI: " << data->roi_time_ms << " ms"
                              << " | CPU: " << cpu_ms << " ms"
                              << " | Total: " << total_ms << " ms" << std::endl;
                }
                
                continue;
            }

            // Keep the source frame alive for the asynchronous GPU consumer;
            // the first four ROIs have already been handled by the CPU path.
            // FrameBuffer is keyed by frame id so the queue need only carry ROIs.
            int remaining_rois = data->rois.size() - cpu_roi_count;
            if (remaining_rois > 0) {
                frame_buffer->store(data->frame_id, data->rgb_image, data->producer_start_ms);
                
                for (size_t i = cpu_roi_count; i < data->rois.size(); i++) {
                    gpu_queue->push(data->rois[i]);
                }
                
            } else {
                data->releaseRGB();
                stats.frames_processed++;
                if (g_pipeline_running.load()) {
                    led_controller->setState("NORMAL");
                }

            }
        } else {
            data->roi_done = true;
            data->releaseRGB();
            stats.frames_processed++;

            if (g_pipeline_running.load()) {
                led_controller->setState("NORMAL");
            }

            if (debug_mode) {
                std::cout << "  [Producer] No components, skipping ROI\n";
            }
        }

        double load_ms = std::chrono::duration<double, std::milli>(
            load_end_tp - load_start_tp
        ).count();
        double total_ms = msSince(frame_start_tp);

        stats.total_components += data->components.size();
        stats.total_rois += data->rois.size();
        stats.total_level1_time.store(stats.total_level1_time.load() + (float)data->level1_time_ms);
        stats.total_roi_time.store(stats.total_roi_time.load() + (float)data->roi_time_ms);
        stats.total_load_time.store(stats.total_load_time.load() + (float)load_ms);

        if (debug_mode) {
            std::cout << "  [Producer] TIMING Frame " << frame_id
                      << " | Load: " << std::fixed << std::setprecision(1) << load_ms << " ms"
                      << " | Level1: " << data->level1_time_ms << " ms"
                      << " | ROI: " << data->roi_time_ms << " ms"
                      << " | Total: " << total_ms << " ms" << std::endl;
        }

        // not re-discovered on the next poll. Guard with try/catch in case
        // the ingestion script's ring buffer deleted the file in the interim.
        try {
            last_seen_mtime = fs::last_write_time(filename);
        } catch (const fs::filesystem_error&) {
            // File vanished. Leave the watermark where it is.
        }

        }  // end for (idx)
    }  // end while (!g_should_exit)

    if (debug_mode) {
        std::cout << "[Producer] Thread finished.\n";
    }
}

} // namespace minimind
