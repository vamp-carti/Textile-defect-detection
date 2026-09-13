#pragma once

#include <memory>

#include "pipeline/gpu_queue.hpp"
#include "pipeline/frame_buffer.hpp"
#include "core/pipeline_data.hpp"
#include "detection/roi_inference.hpp"
#include "pipeline/data_sender.hpp"
#include "pipeline/io_worker.hpp"
#include "pipeline/led_controller.hpp"

namespace minimind {

// Drains GPU batches, records frame verdicts, updates UI/LED state, and tracks
// completion and timing in the shared pipeline statistics.
void consumerThread(
    std::shared_ptr<GPUQueue> gpu_queue,
    std::shared_ptr<FrameBuffer> frame_buffer,
    std::shared_ptr<IoWorker> io_worker,
    std::shared_ptr<LedController> led_controller,
    ROIInference& gpu_inference,
    PipelineStats& stats,
    UIData& ui_data
);

} // namespace minimind
