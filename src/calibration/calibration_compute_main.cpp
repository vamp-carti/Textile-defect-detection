#include "calibration/calibration_compute.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: ./calibration_compute <input_image> [output_json]" << std::endl;
        return 1;
    }
    
    std::string image_path = argv[1];
    std::string output_path = "calibration_metrics.json";
    if (argc >= 3) {
        output_path = argv[2];
    }
    
    cv::Mat img = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (img.empty()) {
        std::cerr << "Failed to load image: " << image_path << std::endl;
        return 1;
    }
    
    std::cout << "Computing calibration from: " << image_path << std::endl;
    auto calib = minimind::CalibrationComputer::computeFromFrame(img);
    minimind::CalibrationComputer::saveToJson(output_path, calib);
    
    std::cout << "Calibration saved to: " << output_path << std::endl;
    return 0;
}
