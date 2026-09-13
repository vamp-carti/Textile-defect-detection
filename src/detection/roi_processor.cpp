#include "roi_processor.hpp"
#include "core/config.hpp"
#include <sstream>
#include <iomanip>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <iostream>
namespace minimind {

namespace {

double centroidDistance(
    double x1, double y1,
    double x2, double y2
) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;

    return std::sqrt(dx * dx + dy * dy);
}

} // anonymous namespace

void ROIProcessor::projectTo4K(
    const std::vector<DefectComponent>& components,
    const cv::Mat& original_rgb,
    std::vector<ROIComponent>& output
) {
    // Level 1 works in a configured coordinate system; ROI classification must
    // return to the source frame so crops and overlays line up with the camera.
    output.clear();
    output.reserve(components.size());

    // DefectComponent coordinates are in Level1's working space,
    // which is target_w x target_h (from config). original_rgb is the
    // frame at its native resolution. The scale is the ratio of the two,
    // computed per frame — it must not be hardcoded.
    const int target_w = Config::getInstance().image.target_w;
    const int target_h = Config::getInstance().image.target_h;

    const double scale_x = static_cast<double>(original_rgb.cols) / target_w;
    const double scale_y = static_cast<double>(original_rgb.rows) / target_h;

    for (const auto& c : components) {
        ROIComponent r;

        r.frame_id = c.frame_id;
        r.id = c.id;

        r.x = static_cast<int>(std::round(c.x * scale_x));
        r.y = static_cast<int>(std::round(c.y * scale_y));

        r.width = static_cast<int>(std::round(c.width * scale_x));
        r.height = static_cast<int>(std::round(c.height * scale_y));

        r.area = static_cast<int>(std::round(c.area * scale_x * scale_y));

        r.centroid_x = c.centroid_x * scale_x;
        r.centroid_y = c.centroid_y * scale_y;

        r.covered = false;

        output.push_back(r);
    }
}

void ROIProcessor::updateCoverage(
    std::vector<ROIComponent>& components,
    const std::vector<ROIBox>& boxes
) {
    for (auto& comp : components) {
        if (comp.covered)
            continue;

        const int cx1 = comp.x;
        const int cy1 = comp.y;
        const int cx2 = comp.x + comp.width;
        const int cy2 = comp.y + comp.height;

        for (const auto& box : boxes) {
            if (cx1 >= box.x &&
                cy1 >= box.y &&
                cx2 <= box.x + box.width &&
                cy2 <= box.y + box.height) {

                comp.covered = true;
                break;
            }
        }
    }
}

std::vector<std::vector<int>> ROIProcessor::bfsSpatialClustering(
    const std::vector<ROIComponent>& components,
    const std::vector<int>& indices
) {
    // P2 candidates close in centroid and overall span are treated as one
    // meaningful region rather than producing many redundant crops.
    std::vector<std::vector<int>> clusters;

    const int n = static_cast<int>(indices.size());

    std::vector<bool> visited(n, false);

    for (int i = 0; i < n; ++i) {
        if (visited[i])
            continue;

        visited[i] = true;

        std::vector<int> cluster;
        std::vector<int> queue;

        cluster.push_back(indices[i]);
        queue.push_back(indices[i]);

        while (!queue.empty()) {
            const int current_index = queue.back();
            queue.pop_back();

            const auto& current = components[current_index];

            for (int j = 0; j < n; ++j) {
                if (visited[j])
                    continue;

                const int target_index = indices[j];
                const auto& target = components[target_index];

                const double dist = centroidDistance(
                    current.centroid_x,
                    current.centroid_y,
                    target.centroid_x,
                    target.centroid_y
                );

                if (dist >= P2_MAX_DIST)
                    continue;

                int min_x = std::numeric_limits<int>::max();
                int min_y = std::numeric_limits<int>::max();
                int max_x = 0;
                int max_y = 0;

                for (int idx : cluster) {
                    const auto& c = components[idx];

                    min_x = std::min(min_x, c.x);
                    min_y = std::min(min_y, c.y);
                    max_x = std::max(max_x, c.x + c.width);
                    max_y = std::max(max_y, c.y + c.height);
                }

                min_x = std::min(min_x, target.x);
                min_y = std::min(min_y, target.y);
                max_x = std::max(max_x, target.x + target.width);
                max_y = std::max(max_y, target.y + target.height);

                if ((max_x - min_x) <= P2_MAX_SPAN &&
                    (max_y - min_y) <= P2_MAX_SPAN) {

                    visited[j] = true;
                    cluster.push_back(target_index);
                    queue.push_back(target_index);
                }
            }
        }

        clusters.push_back(cluster);
    }

    return clusters;
}

void ROIProcessor::generateROIs(
    std::vector<ROIComponent>& components,
    std::vector<ROIBox>& output
) {
    // Priority order is intentional: large isolated defects first, then dense
    // mid-sized clusters, then remaining components as fallback coverage. The
    // area and distance constants below are tuned heuristics, not model limits.
    output.clear();

    if (components.empty())
        return;

    const int frame_id = components[0].frame_id;

    // ============================================================
    // PRIORITY 1
    // ============================================================

    std::vector<int> p1;

    for (int i = 0; i < static_cast<int>(components.size()); ++i) {
        if (components[i].area > P1_AREA &&
            !components[i].covered) {
            p1.push_back(i);
        }
    }

    std::sort(
        p1.begin(),
        p1.end(),
        [&](int a, int b) {
            return components[a].area > components[b].area;
        }
    );

    int p1_count = 0;

    for (int idx : p1) {
        const auto& c = components[idx];

        ++p1_count;

        output.push_back({
            frame_id,
            c.x,
            c.y,
            c.width,
            c.height,
            "P1",
            p1_count,
            0
        });

        components[idx].covered = true;
    }

    updateCoverage(components, output);

    // ============================================================
    // PRIORITY 2
    // ============================================================

    std::vector<int> p2_pool;

    for (int i = 0; i < static_cast<int>(components.size()); ++i) {
        const auto& c = components[i];

        if (c.area >= 270 &&
            c.area <= 700 &&
            !c.covered) {

            p2_pool.push_back(i);
        }
    }

    if (!p2_pool.empty()) {

        auto clusters = bfsSpatialClustering(components, p2_pool);

        struct ClusterScore {
            double score;
            std::vector<int> cluster;
            int min_x;
            int min_y;
            int width;
            int height;
        };

        std::vector<ClusterScore> scored;

        for (const auto& cluster : clusters) {

            int total_area = 0;

            int min_x = std::numeric_limits<int>::max();
            int min_y = std::numeric_limits<int>::max();
            int max_x = 0;
            int max_y = 0;

            for (int idx : cluster) {
                const auto& c = components[idx];

                total_area += c.area;

                min_x = std::min(min_x, c.x);
                min_y = std::min(min_y, c.y);

                max_x = std::max(max_x, c.x + c.width);
                max_y = std::max(max_y, c.y + c.height);
            }

            const int bw = std::max(1, max_x - min_x);
            const int bh = std::max(1, max_y - min_y);

            const double density =
                static_cast<double>(total_area) /
                static_cast<double>(bw * bh);

            const double score = total_area * density;

            scored.push_back({
                score,
                cluster,
                min_x,
                min_y,
                bw,
                bh
            });
        }

        std::sort(
            scored.begin(),
            scored.end(),
            [](const ClusterScore& a, const ClusterScore& b) {
                return a.score > b.score;
            }
        );

        int p2_count = 0;

        for (const auto& cluster : scored) {

            bool has_uncovered = false;

            for (int idx : cluster.cluster) {
                if (!components[idx].covered) {
                    has_uncovered = true;
                    break;
                }
            }

            if (!has_uncovered)
                continue;

            ++p2_count;

            output.push_back({
                frame_id,
                cluster.min_x,
                cluster.min_y,
                cluster.width,
                cluster.height,
                "P2",
                p2_count,
                0
            });

            for (int idx : cluster.cluster)
                components[idx].covered = true;
        }
    }

    updateCoverage(components, output);

    // ============================================================
    // PRIORITY 3
    // ============================================================

    int p3_count = 0;

    for (auto& comp : components) {

        if (comp.covered)
            continue;

        bool merged = false;

        for (auto& box : output) {

            const double box_cx = box.x + box.width / 2.0;
            const double box_cy = box.y + box.height / 2.0;

            const double dist = centroidDistance(
                comp.centroid_x,
                comp.centroid_y,
                box_cx,
                box_cy
            );

            const int new_x1 = std::min(box.x, comp.x);
            const int new_y1 = std::min(box.y, comp.y);

            const int new_x2 = std::max(box.x + box.width, comp.x + comp.width);
            const int new_y2 = std::max(box.y + box.height, comp.y + comp.height);

            const int span_w = new_x2 - new_x1;
            const int span_h = new_y2 - new_y1;

            if (dist < P3_MAX_DIST_1 &&
                span_w <= 175 &&
                span_h <= 175) {

                box.x = new_x1;
                box.y = new_y1;
                box.width = span_w;
                box.height = span_h;

                comp.covered = true;
                merged = true;
                break;
            }

            if (dist >= P3_MAX_DIST_1 &&
                dist <= P3_MAX_DIST_2 &&
                span_w <= 175 &&
                span_h <= 175) {

                box.x = new_x1;
                box.y = new_y1;
                box.width = span_w;
                box.height = span_h;

                comp.covered = true;
                merged = true;
                break;
            }
        }

        if (!merged) {

            ++p3_count;

            output.push_back({
                frame_id,
                comp.x,
                comp.y,
                comp.width,
                comp.height,
                "P3",
                p3_count,
                0
            });

            comp.covered = true;
        }
    }
}

void ROIProcessor::centroidNMS(
    const std::vector<ROIBox>& boxes,
    std::vector<ROIBox>& output
) {
    output.clear();

    for (const auto& box : boxes) {

        const double cx = box.x + box.width / 2.0;
        const double cy = box.y + box.height / 2.0;

        bool keep = true;

        for (const auto& kept : output) {

            const double kept_cx = kept.x + kept.width / 2.0;
            const double kept_cy = kept.y + kept.height / 2.0;

            if (centroidDistance(cx, cy, kept_cx, kept_cy) < NMS_DISTANCE) {
                keep = false;
                break;
            }
        }

        if (keep)
            output.push_back(box);
    }
}

void ROIProcessor::extractROIs(
    const cv::Mat& original_rgb,
    const std::vector<ROIBox>& boxes,
    std::vector<ROIResult>& output
) {
    output.clear();

    if (original_rgb.empty())
        return;

    const int img_w = original_rgb.cols;
    const int img_h = original_rgb.rows;

    for (const auto& box : boxes) {

        // ========================================================
        // NORMAL 224x224 BOX
        // ========================================================

        if (box.width <= ROI_SIZE &&
            box.height <= ROI_SIZE) {

            const int center_x = static_cast<int>(box.x + box.width / 2);
            const int center_y = static_cast<int>(box.y + box.height / 2);

            int x1 = center_x - 112;
            int y1 = center_y - 112;

            int x2 = x1 + 224;
            int y2 = y1 + 224;

            if (x1 < 0) {
                x2 += -x1;
                x1 = 0;
            }

            if (y1 < 0) {
                y2 += -y1;
                y1 = 0;
            }

            if (x2 > img_w) {
                x1 -= (x2 - img_w);
                x2 = img_w;
            }

            if (y2 > img_h) {
                y1 -= (y2 - img_h);
                y2 = img_h;
            }

            x1 = std::max(0, x1);
            y1 = std::max(0, y1);
            x2 = std::min(img_w, x2);
            y2 = std::min(img_h, y2);

            cv::Mat crop = original_rgb(cv::Rect(x1, y1, x2 - x1, y2 - y1));

            cv::Mat roi;

            if (crop.rows != 224 || crop.cols != 224) {
                cv::resize(crop, roi, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
            } else {
                roi = crop.clone();
            }

            output.push_back({
                box.frame_id,
                box,
                roi,
                x1,
                y1,
                x2 - x1,
                y2 - y1
            });
        }

        // ========================================================
        // LARGE BOX → 224x224 TILES
        // ========================================================

        else {

            int tile_sub = 0;

            for (int ty = box.y; ty < box.y + box.height; ty += TILE_STRIDE) {

                for (int tx = box.x; tx < box.x + box.width; tx += TILE_STRIDE) {

                    ++tile_sub;

                    int x1 = tx;
                    int y1 = ty;

                    int x2 = std::min(img_w, x1 + 224);
                    int y2 = std::min(img_h, y1 + 224);

                    if ((x2 - x1) < 224 && x2 == img_w) {
                        x1 = std::max(0, img_w - 224);
                    }

                    if ((y2 - y1) < 224 && y2 == img_h) {
                        y1 = std::max(0, img_h - 224);
                    }

                    x1 = std::max(0, x1);
                    y1 = std::max(0, y1);

                    x2 = std::min(img_w, x1 + 224);
                    y2 = std::min(img_h, y1 + 224);

                    cv::Mat crop = original_rgb(cv::Rect(x1, y1, x2 - x1, y2 - y1));

                    cv::Mat roi;

                    if (crop.rows != 224 || crop.cols != 224) {
                        cv::resize(crop, roi, cv::Size(224, 224), 0, 0, cv::INTER_AREA);
                    } else {
                        roi = crop.clone();
                    }

                    ROIBox tile_box = box;
                    tile_box.tile_sub = tile_sub;
                    
                    output.push_back({
                        box.frame_id,
                        tile_box,
                        roi,
                        x1,
                        y1,
                        x2 - x1,
                        y2 - y1
                    });
                }
            }
        }
    }
}

BenchmarkTimer& ROIProcessor::getBenchmarkTimer() {
    return benchmark_;
}

std::vector<ROIResult> ROIProcessor::process(
    const cv::Mat& original_rgb,
    const std::vector<DefectComponent>& components
) {
    if (components.empty())
        return {};

    benchmark_.start("09_roi_projection");    
    projectTo4K(components, original_rgb, projected_components_);
    benchmark_.stop("09_roi_projection");

    benchmark_.start("10_roi_generation");
    generateROIs(projected_components_, projected_boxes_);
    benchmark_.stop("10_roi_generation");

    benchmark_.start("11_roi_nms");
    centroidNMS(projected_boxes_, deduped_boxes_);
    benchmark_.stop("11_roi_nms");

    benchmark_.start("12_roi_extraction");
    extractROIs(original_rgb, deduped_boxes_, results_);
    benchmark_.stop("12_roi_extraction");

    if (Config::getInstance().isDebugMode()) {
        if (Config::getInstance().debug.verbose_roi) {
            std::cout << "\n[ROI DEBUG] Generated " << results_.size() << " ROIs:\n";
            for (const auto& roi : results_) {
                std::cout << "  ROI " << roi.box.number 
                          << " | Priority: " << roi.box.priority
                          << " | Area: " << (roi.box.width * roi.box.height)
                          << " | Position: (" << roi.box.x << ", " << roi.box.y << ")\n";
            }
        } else {
            std::cout << "  [ROI] Generated " << results_.size() << " ROIs\n";
        }

        // ============================================================
        // DEBUG: draw all ROI boxes + crop rects on the original frame
        // Remove this whole block when done debugging.
        // ============================================================
        {
            cv::Mat overlay;
            cv::cvtColor(original_rgb, overlay, cv::COLOR_RGB2BGR);

            for (const auto& roi : results_) {
                cv::Rect box_rect(
                    roi.box.x, roi.box.y,
                    roi.box.width, roi.box.height
                );
                cv::rectangle(overlay, box_rect, cv::Scalar(0, 255, 0), 3);

                cv::Rect crop_rect(
                    roi.crop_x, roi.crop_y,
                    roi.crop_width, roi.crop_height
                );
                cv::rectangle(overlay, crop_rect, cv::Scalar(0, 0, 255), 1);

                std::string label =
                    "#" + std::to_string(roi.box.number) +
                    " " + roi.box.priority;
                cv::putText(
                    overlay, label,
                    cv::Point(roi.box.x, std::max(15, roi.box.y - 5)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 0), 2
                );
            }

            static bool overlay_dir_created = false;
            if (!overlay_dir_created) {
                system("mkdir -p output/debug");
                overlay_dir_created = true;
            }

            std::stringstream ss;
            ss << "output/debug/rois_overlay_frame_"
               << std::setw(6) << std::setfill('0')
               << components.front().frame_id
               << ".png";
            cv::imwrite(ss.str(), overlay);
            std::cout << "  [ROI] Overlay written: " << ss.str() << std::endl;
        }
        // ============================================================
        // END DEBUG overlay
        // ============================================================
    }

    return results_;
}

} // namespace minimind
