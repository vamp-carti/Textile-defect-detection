#pragma once

#include <vector>
#include <memory>

#include <opencv2/core.hpp>

#include "core/pipeline_data.hpp"
#include "detection/level1_detector.hpp"
#include "detection/roi_processor.hpp"

namespace minimind {

void producerThread(
    std::shared_ptr<PipelineQueue> queue,
    const std::vector<cv::String>& filenames,
    Level1Detector& detector,
    ROIProcessor& roi_processor,
    PipelineStats& stats
);

} // namespace minimind
