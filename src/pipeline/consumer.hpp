#pragma once

#include <memory>

#include "core/pipeline_data.hpp"
#include "detection/roi_inference.hpp"

namespace minimind {

void consumerThread(
    std::shared_ptr<PipelineQueue> queue,
    ROIInference& roi_inference,
    PipelineStats& stats
);

} // namespace minimind
