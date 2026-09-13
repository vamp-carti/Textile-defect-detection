#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

#include "benchmark.hpp"
#include "calibration.hpp"

namespace minimind {

struct DefectComponent {
    int frame_id;
    int id;
    int x;
    int y;
    int width;
    int height;
    int area;
    double centroid_x;
    double centroid_y;
};

class Level1Detector {
public:
    explicit Level1Detector(const std::string& calibration_path, bool debug_mode = false);
    bool initialize();
    bool reloadCalibration(const std::string& path);
    cv::Mat process(const cv::Mat& input, int frame_id);

    const StageLatency& getLastBenchmark() const;
    const BenchmarkTimer& getBenchmarkTimer() const { return timer_; }
    const std::vector<DefectComponent>& getLastComponents() const { return last_components_; }

private:
    std::string calibration_path_;
    CalibrationData calib_;
    bool debug_mode_;
    BenchmarkTimer timer_;
    std::vector<DefectComponent> last_components_;
    bool buildRuntimeState_();

    // Precomputed kernels and DFT canvases
    cv::Mat kernel_real_;
    cv::Mat kernel_imag_;
    cv::Mat dft_kernel_real_;
    cv::Mat dft_kernel_imag_;
    cv::Mat close_kernel_ds_;
    cv::Mat line_kernel_ds_;
    cv::Mat oil_close_kernel_ds_;
    cv::Mat oil_open_kernel_ds_;

    // Preallocated buffers for runtime memory reuse
    cv::Mat img_ds2_;
    cv::Mat bg_ds8_;
    cv::Mat bg_1080p_;
    cv::Mat norm_img_ds2_;
    cv::Mat padded_norm_;
    cv::Mat dft_input_canvas_;
    cv::Mat planes_[2];
    cv::Mat complex_img_;
    cv::Mat spec_real_;
    cv::Mat spec_imag_;
    cv::Mat img_dft_;
    cv::Mat idft_real_;
    cv::Mat idft_imag_;
    cv::Mat struct_energy_ds_;
    cv::Mat mean_i_;
    cv::Mat mean_i2_;
    cv::Mat local_variance_;
    cv::Mat raw_oil_mask_;
    cv::Mat solid_oil_mask_;
    cv::Mat unified_mask_;
    cv::Mat macro_mask_;
    cv::Mat final_mask_;
    cv::Mat bg_float_buf_;
    
    // Preallocated buffers for Z-Score Envelope (Stage 04)
    cv::Mat closed_struct_;
    cv::Mat morph_struct_;
    cv::Mat col_energy_broadcast_;
    cv::Mat col_int_broadcast_;
    cv::Mat norm_img_ds2_float_;
    cv::Mat struct_mask_;
    cv::Mat intensity_mask_;
    cv::Mat combined_pixel_mask_;

    // Stage 05
    cv::Mat oil_closed_;
    cv::Mat oil_opened_;
};

} // namespace minimind
