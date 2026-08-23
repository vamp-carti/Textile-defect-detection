#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "calibration/calibration.hpp"

namespace minimind {

class CalibrationComputer {
public:
    static CalibrationData computeFromFrame(const cv::Mat& grayscale_frame);
    static bool saveToJson(const std::string& path, const CalibrationData& data);
    static bool saveVisualization(const std::string& image_path,
                                   const cv::Mat& original_frame,
                                   const cv::Mat& patch);
};

} // namespace minimind
