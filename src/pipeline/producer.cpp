#include "pipeline/producer.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include <opencv2/opencv.hpp>

#include "core/config.hpp"

extern std::atomic<bool> g_pipeline_running;
namespace minimind {

namespace {

bool loadFrameData(
    const std::string& filename,
    int frame_id,
    std::shared_ptr<FrameData>& data
) {
    cv::Mat bgr = cv::imread(filename, cv::IMREAD_COLOR);
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
    std::shared_ptr<PipelineQueue> queue,
    const std::vector<cv::String>& filenames,
    Level1Detector& detector,
    ROIProcessor& roi_processor,
    PipelineStats& stats
) {
    bool debug_mode = Config::getInstance().isDebugMode();

    if (debug_mode) {
        std::cout << "[Producer] Thread started. Processing " 
                  << filenames.size() << " frames.\n";
    }

    int frame_id = 0;

    for (const auto& filename : filenames) {
        // Wait if pipeline is paused
        while (!g_pipeline_running && !queue->shouldStop()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (queue->shouldStop()) {
            if (debug_mode) {
                std::cout << "[Producer] Stop signal received. Exiting.\n";
            }
            break;
        }

        ++frame_id;

        std::shared_ptr<FrameData> data;
        if (!loadFrameData(filename, frame_id, data)) {
            continue;
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

        if (debug_mode) {
            std::cout << "  [Producer] Level1: " << data->components.size() 
                      << " components, " << data->level1_time_ms << " ms\n";
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

            if (debug_mode) {
                std::cout << "  [Producer] ROI: " << data->rois.size() 
                          << " ROIs, " << data->roi_time_ms << " ms\n";
            }
        } else {
            data->roi_done = true;
            data->releaseRGB();

            if (debug_mode) {
                std::cout << "  [Producer] No components, skipping ROI\n";
            }
        }

        if (!queue->push(data)) {
            std::cerr << "[Producer] Failed to push frame " 
                      << frame_id << " (queue stopped)\n";
            break;
        }

        stats.total_components += data->components.size();
        stats.total_rois += data->rois.size();
        stats.total_level1_time.store(stats.total_level1_time.load() + (float)data->level1_time_ms);
        stats.total_roi_time.store(stats.total_roi_time.load() + (float)data->roi_time_ms);

        if (debug_mode) {
            std::cout << "  [Producer] Queued frame " << frame_id 
                      << " (queue size: " << queue->size() << ")\n";
        }
    }

    queue->stop();
    if (debug_mode) {
        std::cout << "[Producer] Thread finished.\n";
    }
}

} // namespace minimind
