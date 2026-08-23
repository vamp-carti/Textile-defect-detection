#include "pipeline/consumer.hpp"

#include <iostream>
#include <chrono>
#include <thread>
#include "core/config.hpp"
extern std::atomic<bool> g_pipeline_running;

namespace minimind {

void consumerThread(
    std::shared_ptr<PipelineQueue> queue,
    ROIInference& roi_inference,
    PipelineStats& stats
) {
    bool debug_mode = Config::getInstance().isDebugMode();

    if (debug_mode) {
        std::cout << "[Consumer] Thread started.\n";
    }

    int frames_inferred = 0;

    while (true) {
	while (!g_pipeline_running && !queue->shouldStop()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        if (queue->shouldStop() && queue->size() == 0) {
            if (debug_mode) {
                std::cout << "[Consumer] Queue empty and stop signal received. Exiting.\n";
            }
            break;
        }

        auto data = queue->pop();
        if (!data) {
            if (debug_mode) {
                std::cout << "[Consumer] Received null data. Exiting.\n";
            }
            break;
        }

        if (data->inference_done) {
            queue->markCompleted();
            continue;
        }

        if (data->rois.empty()) {
            if (debug_mode) {
                std::cout << "[Consumer] Frame " << data->frame_id 
                          << ": No ROIs to infer.\n";
            }
            data->inference_done = true;
            stats.frames_processed++;
            queue->markCompleted();
            continue;
        }

        auto inference_start = std::chrono::high_resolution_clock::now();

        if (debug_mode) {
            std::cout << "[Consumer] Frame " << data->frame_id 
                      << ": Running inference on " << data->rois.size() << " ROIs\n";
        }

        data->inference_results = roi_inference.processBatch(data->rois, data->rgb_image);

        auto inference_end = std::chrono::high_resolution_clock::now();
        data->inference_time_ms = std::chrono::duration<double, std::milli>(
            inference_end - inference_start
        ).count();

        data->has_defect = false;
        for (const auto& result : data->inference_results) {
            if (result.defect) {
                data->has_defect = true;
                break;
            }
        }

        data->inference_done = true;

        frames_inferred++;
        stats.frames_processed++;
        stats.total_inferences += data->inference_results.size();
        stats.total_inference_time.store(stats.total_inference_time.load() + (float)data->inference_time_ms);
        
        if (data->has_defect) {
            stats.frames_with_defects++;
        }

        if (debug_mode) {
            std::cout << "[Consumer] Frame " << data->frame_id 
                      << ": Inference done in " << data->inference_time_ms << " ms"
                      << ", " << data->inference_results.size() << " results"
                      << (data->has_defect ? " [DEFECT FOUND]" : " [CLEAR]")
                      << "\n";
        }

        data->releaseRGB();
        queue->markCompleted();
    }

    if (debug_mode) {
        std::cout << "[Consumer] Thread finished. Inferred " 
                  << frames_inferred << " frames.\n";
    }
}

} // namespace minimind
