#pragma once

#include <vector>
#include <memory>

#include <opencv2/core.hpp>

#include "pipeline/gpu_queue.hpp"
#include "pipeline/frame_buffer.hpp"
#include "core/pipeline_data.hpp"
#include "detection/level1_detector.hpp"
#include "detection/roi_processor.hpp"
#include "pipeline/data_sender.hpp"
#include "pipeline/io_worker.hpp"
#include "pipeline/led_controller.hpp"

namespace minimind {

// Localizes defects, creates prioritized ROIs, runs CPU early rejection, and
// sends remaining ROIs to the asynchronous GPU consumer.
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
);

} // namespace minimind
