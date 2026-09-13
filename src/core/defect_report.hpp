#pragma once

#include <string>
#include <chrono>

struct DefectReport {
    int frame_id;
    int roi_id;
    std::string predicted_class;
    float confidence;
    float defect_probability;
    std::chrono::system_clock::time_point timestamp;
};
