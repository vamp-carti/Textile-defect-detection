#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "level1_detector.hpp"
#include "benchmark.hpp"

namespace minimind {

struct ROIComponent {
    int frame_id;
    int id;

    int x;
    int y;
    int width;
    int height;
    int area;

    double centroid_x;
    double centroid_y;

    bool covered = false;
};

struct ROIBox {
    int frame_id;

    int x;
    int y;
    int width;
    int height;

    std::string priority;
    int number;

    int tile_sub = 0;
};

struct ROIResult {
    int frame_id;
    ROIBox box;
    cv::Mat image;

    int crop_x;
    int crop_y;
    int crop_width;
    int crop_height;
};

class ROIProcessor {
public:
    std::vector<ROIResult> process(
        const cv::Mat& original_rgb,
        const std::vector<DefectComponent>& components
    );
    BenchmarkTimer& getBenchmarkTimer();

private:
    BenchmarkTimer benchmark_;

    static constexpr double SCALE_X = 2.0;
    static constexpr double SCALE_Y = 2.0;

    static constexpr double P2_MAX_DIST = 105.0;
    static constexpr int P2_MAX_SPAN = 175;

    static constexpr int P1_AREA = 700;

    static constexpr double P3_MAX_DIST_1 = 105.0;
    static constexpr double P3_MAX_DIST_2 = 157.5;

    static constexpr double NMS_DISTANCE = 112.0;

    static constexpr int ROI_SIZE = 224;
    static constexpr int TILE_STRIDE = 112;

    // Preallocated buffers
    std::vector<ROIComponent> projected_components_;
    std::vector<ROIBox> projected_boxes_;
    std::vector<ROIBox> deduped_boxes_;
    std::vector<ROIResult> results_;

    void projectTo4K(
        const std::vector<DefectComponent>& components,
        std::vector<ROIComponent>& output
    );

    void generateROIs(
        std::vector<ROIComponent>& components,
        std::vector<ROIBox>& output
    );

    std::vector<std::vector<int>> bfsSpatialClustering(
        const std::vector<ROIComponent>& components,
        const std::vector<int>& indices
    );

    void updateCoverage(
        std::vector<ROIComponent>& components,
        const std::vector<ROIBox>& boxes
    );

    void centroidNMS(
        const std::vector<ROIBox>& boxes,
        std::vector<ROIBox>& output
    );

    void extractROIs(
        const cv::Mat& original_rgb,
        const std::vector<ROIBox>& boxes,
        std::vector<ROIResult>& output
    );
};

} // namespace minimind
