#include "pipeline/consumer.hpp"

#include <iostream>
#include <set>
#include <chrono>
#include <thread>
#include "pipeline/gpu_queue.hpp"
#include "pipeline/frame_buffer.hpp"
#include <map>
#include "core/config.hpp"
#include "pipeline/io_worker.hpp"

extern std::atomic<bool> g_pipeline_running;

namespace minimind {

void consumerThread(
    std::shared_ptr<GPUQueue> gpu_queue,
    std::shared_ptr<FrameBuffer> frame_buffer,
    std::shared_ptr<IoWorker> io_worker,
    std::shared_ptr<LedController> led_controller,
    ROIInference& gpu_inference,
    PipelineStats& stats,
    UIData& ui_data
) {
    bool debug_mode = Config::getInstance().isDebugMode();

    if (debug_mode) {
        std::cout << "[Consumer] Thread started.\n";
    }

    int frames_inferred = 0;

    while (true) {
        // Batches may contain a full group or a partial group released when the
        // producer starts another frame or shuts down.
        auto batch = gpu_queue->popBatch();
        
        if (batch.empty()) {
            if (debug_mode) {
                std::cout << "[Consumer] Received empty batch. Exiting.\n";
            }
            break;
        }

        if (debug_mode) {
            std::cout << "[Consumer] Processing batch of " 
                      << batch.size() << " ROIs\n";
        }

        auto inference_start = std::chrono::high_resolution_clock::now();

        // Process batch on GPU
        // Retrieve the frame from frame_buffer for this batch
        cv::Mat frame = frame_buffer->retrieve(batch.front().frame_id);
        auto results = gpu_inference.processBatch(batch, frame);

        auto inference_end = std::chrono::high_resolution_clock::now();
        double inference_time_ms = std::chrono::duration<double, std::milli>(
            inference_end - inference_start
        ).count();

        // Only the first positive result per frame is retained; later ROIs from
        // that frame cannot change its binary verdict. This is the pipeline's
        // early-reject contract.
        std::map<int, bool> frame_has_defect;       // frame_id -> true once a defect is seen
        std::map<int, double> frame_flag_time_ms;   // frame_id -> moment the defect was flagged

        for (size_t i = 0; i < results.size(); ++i) {
            const auto& result = results[i];
            const auto& roi = batch[i];

            int frame_id = result.frame_id;

            if (frame_has_defect.find(frame_id) != frame_has_defect.end()) {
                continue;
            }

            if (result.defect) {
                frame_has_defect[frame_id] = true;
                frame_flag_time_ms[frame_id] = result.flag_time_ms;

                ui_data.addDefect(
                    frame_id,
                    roi.box.number,
                    result.predicted_label,
                    result.confidence,
                    result.defect_probability
                );

                frame_buffer->remove(frame_id);

                if (debug_mode) {
                    std::cout << "[Consumer] GPU defect recorded for frame "
                              << frame_id << "\n";
                }
            }
        }

        // Cleanup frames that did not produce a defect and accumulate per-frame stats.
        // The batch may contain ROIs from multiple frames; count each unique frame once.
        std::set<int> frames_in_batch;
        for (const auto& roi : batch) frames_in_batch.insert(roi.frame_id);

        for (int fid : frames_in_batch) {
            double start_ms = frame_buffer->getProducerStartMs(fid);
            bool had_defect = (frame_has_defect.find(fid) != frame_has_defect.end());

            if (frame_has_defect.find(fid) != frame_has_defect.end()) {
                // Defect frame: e2e ends at the flag moment
                double flag_ms = frame_flag_time_ms[fid];
                if (start_ms > 0.0 && flag_ms > 0.0) {
                    double e2e_ms = flag_ms - start_ms;
                    stats.total_end_to_end_time.store(
                        stats.total_end_to_end_time.load() + (float)e2e_ms);
                }
                // frame_buffer->remove(fid) already called in the defect branch
            } else {
                // Non-defect frame: e2e ends when the consumer finishes this batch
                if (start_ms > 0.0) {
                    double now_ms = std::chrono::duration<double, std::milli>(
                        std::chrono::high_resolution_clock::now().time_since_epoch()
                    ).count();
                    double e2e_ms = now_ms - start_ms;
                    stats.total_end_to_end_time.store(
                        stats.total_end_to_end_time.load() + (float)e2e_ms);
                }
                if (debug_mode) {
                    std::cout << "[Consumer] No defect for frame "
                              << fid << ", removing from buffer\n";
                }
                frame_buffer->remove(fid);
            }

            stats.frames_processed++;

            // LED: switch animation based on this frame's verdict,
            // but only if the pipeline is still running. Otherwise the
            // STOP/QUIT handler's IDLE state must not be clobbered.
            if (g_pipeline_running.load()) {
                led_controller->setState(had_defect ? "DEFECT" : "NORMAL");
            }
        }

        stats.total_inferences += results.size();
        stats.total_inference_time.store(stats.total_inference_time.load() + (float)inference_time_ms);

        if (!frame_has_defect.empty()) {
            stats.frames_with_defects += frame_has_defect.size();
        }

        if (debug_mode) {
            std::cout << "[Consumer] Batch done in " << inference_time_ms 
                      << " ms, " << results.size() << " results\n";
        }
    }

    if (debug_mode) {
        std::cout << "[Consumer] Thread finished. Inferred " 
                  << frames_inferred << " frames.\n";
    }
}

} // namespace minimind
