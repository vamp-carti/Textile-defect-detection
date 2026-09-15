#include "level1_detector.hpp"
#include "config.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

namespace minimind {

Level1Detector::Level1Detector(const std::string& calibration_path, bool debug_mode)
    : calibration_path_(calibration_path), debug_mode_(debug_mode) {}

bool Level1Detector::initialize() {
    cv::setNumThreads(4);

    if (!loadCalibration(calibration_path_, calib_)) {
        return false;
    }

    return buildRuntimeState_();
}

bool Level1Detector::reloadCalibration(const std::string& path) {
    CalibrationData new_calib;
    if (!loadCalibration(path, new_calib)) {
        std::cerr << "[Level1Detector] Failed to reload calibration from " << path << std::endl;
        return false;
    }
    calib_ = new_calib;
    calibration_path_ = path;
    if (!buildRuntimeState_()) {
        std::cerr << "[Level1Detector] Rebuild of runtime state failed after reload" << std::endl;
        return false;
    }
    if (debug_mode_) {
        std::cout << "[Level1Detector] Calibration reloaded and kernels rebuilt from " << path << std::endl;
    }
    return true;
}

bool Level1Detector::buildRuntimeState_() {
    // Build every reusable kernel and buffer after calibration load/reload.
    // Keeping these allocations out of process() is important on the A53 CPU;
    // recalibration therefore happens while the pipeline is stopped.
    // 1. Gabor kernels measure oriented textile structure in the downsampled image.
    // The downsample factor and target dimensions come from configuration and
    // are the coordinate system used by Level 1 components.
    double psi_real = 0.0;
    double psi_imag = CV_PI / 2.0;

    cv::Mat g_real = cv::getGaborKernel(cv::Size(calib_.ksize_ds, calib_.ksize_ds), 
                                        calib_.sigma_ds, calib_.theta, calib_.lambd_ds, 
                                        calib_.gamma, psi_real, CV_32F);
    cv::Mat g_imag = cv::getGaborKernel(cv::Size(calib_.ksize_ds, calib_.ksize_ds), 
                                        calib_.sigma_ds, calib_.theta, calib_.lambd_ds, 
                                        calib_.gamma, psi_imag, CV_32F);

    cv::Scalar mean_real = cv::mean(g_real);
    cv::Scalar mean_imag = cv::mean(g_imag);
    kernel_real_ = g_real - mean_real[0];
    kernel_imag_ = g_imag - mean_imag[0];

    // 2. Morphology kernels clean and connect candidate masks at each scale.
    close_kernel_ds_ = cv::getStructuringElement(cv::MORPH_RECT, 
        cv::Size(Config::getInstance().morphology.close_ksize_ds, 
                 Config::getInstance().morphology.close_ksize_ds));

    cv::Mat base_line = cv::Mat::zeros(Config::getInstance().morphology.line_len_ds, 
                                       Config::getInstance().morphology.line_len_ds, CV_8U);
    base_line.col(Config::getInstance().morphology.line_len_ds / 2) = 255;

    double angle_deg = calib_.theta * 180.0 / CV_PI + 90.0;
    double angle_rad = angle_deg * CV_PI / 180.0;
    double cos_a = std::cos(angle_rad);
    double sin_a = std::sin(angle_rad);
    float cx = static_cast<float>(Config::getInstance().morphology.line_len_ds) / 2.0f;
    float cy = static_cast<float>(Config::getInstance().morphology.line_len_ds) / 2.0f;

    cv::Mat rot_mat = (cv::Mat_<double>(2, 3) << 
        cos_a, -sin_a, cx * (1 - cos_a) + cy * sin_a,
        sin_a,  cos_a, cy * (1 - cos_a) - cx * sin_a
    );
    cv::warpAffine(base_line, line_kernel_ds_, rot_mat, 
                   cv::Size(Config::getInstance().morphology.line_len_ds, 
                            Config::getInstance().morphology.line_len_ds), 
                   cv::INTER_NEAREST, cv::BORDER_CONSTANT, cv::Scalar(0));

    oil_close_kernel_ds_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(7, 7));
    oil_open_kernel_ds_ = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));

    // 3. DFT setup: pad to OpenCV's efficient dimensions and cache kernel spectra.
    int pad_size_ds = calib_.ksize_ds;
    int raw_padded_h_ds = Config::getInstance().image.target_h + 2 * pad_size_ds;
    int raw_padded_w_ds = Config::getInstance().image.target_w + 2 * pad_size_ds;

    int opt_h = cv::getOptimalDFTSize(raw_padded_h_ds);
    int opt_w = cv::getOptimalDFTSize(raw_padded_w_ds);

    dft_kernel_real_ = cv::Mat::zeros(opt_h, opt_w, CV_32F);
    cv::Mat roi_r(dft_kernel_real_, cv::Rect(0, 0, calib_.ksize_ds, calib_.ksize_ds));
    kernel_real_.copyTo(roi_r);

    dft_kernel_imag_ = cv::Mat::zeros(opt_h, opt_w, CV_32F);
    cv::Mat roi_i(dft_kernel_imag_, cv::Rect(0, 0, calib_.ksize_ds, calib_.ksize_ds));
    kernel_imag_.copyTo(roi_i);

    cv::dft(dft_kernel_real_, dft_kernel_real_, cv::DFT_REAL_OUTPUT);
    cv::dft(dft_kernel_imag_, dft_kernel_imag_, cv::DFT_REAL_OUTPUT);

    spec_real_.create(opt_h, opt_w, CV_32F);
    spec_imag_.create(opt_h, opt_w, CV_32F);
    idft_real_.create(opt_h, opt_w, CV_32F);
    idft_imag_.create(opt_h, opt_w, CV_32F);

    int bg_w = Config::getInstance().image.target_w / Config::getInstance().background.bg_ds_factor;
    int bg_h = Config::getInstance().image.target_h / Config::getInstance().background.bg_ds_factor;

    bg_float_buf_.create(Config::getInstance().image.target_h, 
                         Config::getInstance().image.target_w, CV_32F);
    
    col_energy_broadcast_.create(Config::getInstance().image.target_h, 
                                  Config::getInstance().image.target_w, CV_32F);
    col_int_broadcast_.create(Config::getInstance().image.target_h, 
                              Config::getInstance().image.target_w, CV_32F);
    norm_img_ds2_float_.create(Config::getInstance().image.target_h, 
                               Config::getInstance().image.target_w, CV_32F);
    struct_mask_.create(Config::getInstance().image.target_h, 
                        Config::getInstance().image.target_w, CV_8U);
    intensity_mask_.create(Config::getInstance().image.target_h, 
                           Config::getInstance().image.target_w, CV_8U);
    combined_pixel_mask_.create(Config::getInstance().image.target_h, 
                                Config::getInstance().image.target_w, CV_8U);

    img_ds2_.create(Config::getInstance().image.target_h, 
                    Config::getInstance().image.target_w, CV_8U);
    bg_ds8_.create(bg_h, bg_w, CV_8U);
    bg_1080p_.create(Config::getInstance().image.target_h, 
                     Config::getInstance().image.target_w, CV_8U);
    norm_img_ds2_.create(Config::getInstance().image.target_h, 
                         Config::getInstance().image.target_w, CV_8U);
    padded_norm_.create(Config::getInstance().image.target_h + 2 * pad_size_ds, 
                        Config::getInstance().image.target_w + 2 * pad_size_ds, CV_8U);
    dft_input_canvas_.create(opt_h, opt_w, CV_32F);
    img_dft_.create(opt_h, opt_w, CV_32F);
    struct_energy_ds_.create(Config::getInstance().image.target_h, 
                             Config::getInstance().image.target_w, CV_32F);
    mean_i_.create(Config::getInstance().image.target_h, 
                   Config::getInstance().image.target_w, CV_32F);
    mean_i2_.create(Config::getInstance().image.target_h, 
                    Config::getInstance().image.target_w, CV_32F);
    local_variance_.create(Config::getInstance().image.target_h, 
                           Config::getInstance().image.target_w, CV_32F);
    raw_oil_mask_.create(Config::getInstance().image.target_h, 
                         Config::getInstance().image.target_w, CV_8U);
    solid_oil_mask_.create(Config::getInstance().image.target_h, 
                           Config::getInstance().image.target_w, CV_8U);
    unified_mask_.create(Config::getInstance().image.target_h, 
                         Config::getInstance().image.target_w, CV_8U);
    macro_mask_.create(Config::getInstance().image.target_h, 
                       Config::getInstance().image.target_w, CV_8U);
    oil_closed_.create(Config::getInstance().image.target_h, 
                       Config::getInstance().image.target_w, CV_8U);
    oil_opened_.create(Config::getInstance().image.target_h, 
                       Config::getInstance().image.target_w, CV_8U);
    closed_struct_.create(Config::getInstance().image.target_h, 
                          Config::getInstance().image.target_w, CV_8U);
    morph_struct_.create(Config::getInstance().image.target_h, 
                         Config::getInstance().image.target_w, CV_8U);

    return true;
}

cv::Mat Level1Detector::process(const cv::Mat& input, int frame_id) {
    timer_.reset();
    timer_.start("00_total");

    timer_.start("01_io_read");
    timer_.stop("01_io_read");

    timer_.start("02_illum_norm");
    cv::resize(input, img_ds2_, cv::Size(Config::getInstance().image.target_w, 
                                         Config::getInstance().image.target_h), 0, 0, cv::INTER_AREA);

    int bg_w = Config::getInstance().image.target_w / Config::getInstance().background.bg_ds_factor;
    int bg_h = Config::getInstance().image.target_h / Config::getInstance().background.bg_ds_factor;
    cv::resize(img_ds2_, bg_ds8_, cv::Size(bg_w, bg_h), 0, 0, cv::INTER_AREA);

    int blur_pad_ds = std::max(1, 125 / (Config::getInstance().background.bg_ds_factor * 
                                        Config::getInstance().image.ds_factor));
    cv::Mat padded_bg;
    cv::copyMakeBorder(bg_ds8_, padded_bg, blur_pad_ds, blur_pad_ds, blur_pad_ds, blur_pad_ds, cv::BORDER_REFLECT_101);
    cv::GaussianBlur(padded_bg, padded_bg, cv::Size(Config::getInstance().background.ksize_ds_stage1, 
                                                    Config::getInstance().background.ksize_ds_stage1), 0, 0);

    cv::Mat bg_ds8_blur = padded_bg(cv::Rect(blur_pad_ds, blur_pad_ds, bg_w, bg_h));

    cv::resize(bg_ds8_blur, bg_1080p_, cv::Size(Config::getInstance().image.target_w, 
                                                Config::getInstance().image.target_h), 0, 0, cv::INTER_LINEAR);

    timer_.start("02f_divide_normalize");
    bg_1080p_.convertTo(bg_float_buf_, CV_32F);
    cv::max(bg_float_buf_, 1.0f, bg_float_buf_);
    cv::divide(img_ds2_, bg_float_buf_, norm_img_ds2_, 128.0, CV_32F);
    norm_img_ds2_.convertTo(norm_img_ds2_, CV_8U);
    timer_.stop("02f_divide_normalize");

    timer_.stop("02_illum_norm");

    timer_.start("03_gabor_filter");

    timer_.start("03a_fft_reflect_pad");
    int pad_size_ds = calib_.ksize_ds;
    cv::copyMakeBorder(norm_img_ds2_, padded_norm_, pad_size_ds, pad_size_ds, pad_size_ds, pad_size_ds, cv::BORDER_REFLECT_101);
    timer_.stop("03a_fft_reflect_pad");

    timer_.start("03b_fft_zero_pad");
    int opt_h = dft_kernel_real_.rows;
    int opt_w = dft_kernel_real_.cols;
    dft_input_canvas_.create(opt_h, opt_w, CV_32F);
    dft_input_canvas_.setTo(0);
    cv::Mat roi(dft_input_canvas_, cv::Rect(0, 0, padded_norm_.cols, padded_norm_.rows));
    padded_norm_.convertTo(roi, CV_32F);
    timer_.stop("03b_fft_zero_pad");

    timer_.start("03c_fft_forward");
    cv::dft(dft_input_canvas_, img_dft_, cv::DFT_REAL_OUTPUT);
    timer_.stop("03c_fft_forward");

    timer_.start("03d_fft_mulspectrums");
    cv::mulSpectrums(img_dft_, dft_kernel_real_, spec_real_, cv::DFT_REAL_OUTPUT);
    timer_.stop("03d_fft_mulspectrums");
    cv::mulSpectrums(img_dft_, dft_kernel_imag_, spec_imag_, cv::DFT_REAL_OUTPUT);

    timer_.start("03e_fft_inverse");
    cv::idft(spec_real_, idft_real_, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    cv::idft(spec_imag_, idft_imag_, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);
    timer_.stop("03e_fft_inverse");

    timer_.start("03f_fft_magnitude");
    cv::Mat crop_real = idft_real_(cv::Rect(pad_size_ds, pad_size_ds, 
                                            Config::getInstance().image.target_w, 
                                            Config::getInstance().image.target_h));
    cv::Mat crop_imag = idft_imag_(cv::Rect(pad_size_ds, pad_size_ds, 
                                            Config::getInstance().image.target_w, 
                                            Config::getInstance().image.target_h));
    cv::magnitude(crop_real, crop_imag, struct_energy_ds_);
    timer_.stop("03f_fft_magnitude");

    timer_.stop("03_gabor_filter");

    timer_.start("04_zscore_envelope");

    timer_.start("04_sampling");
    std::vector<cv::Mat> sampled_rows;
    for (int r = 0; r < struct_energy_ds_.rows; r += 4) {
        sampled_rows.push_back(struct_energy_ds_.row(r));
    }
    cv::Mat struct_sampled;
    if (!sampled_rows.empty()) {
        cv::vconcat(sampled_rows, struct_sampled);
    } else {
        struct_sampled = struct_energy_ds_;
    }
    timer_.stop("04_sampling");

    timer_.start("04_reduce_energy");
    cv::Mat col_energy_mean;
    cv::reduce(struct_sampled, col_energy_mean, 0, cv::REDUCE_AVG, CV_32F);
    timer_.stop("04_reduce_energy");

    timer_.start("04_sampling_int");
    std::vector<cv::Mat> norm_sampled_rows;
    for (int r = 0; r < norm_img_ds2_.rows; r += 4) {
        norm_sampled_rows.push_back(norm_img_ds2_.row(r));
    }
    cv::Mat norm_sampled;
    if (!norm_sampled_rows.empty()) {
        cv::vconcat(norm_sampled_rows, norm_sampled);
    } else {
        norm_sampled = norm_img_ds2_;
    }
    timer_.stop("04_sampling_int");

    timer_.start("04_reduce_int");
    cv::Mat col_int_mean;
    cv::reduce(norm_sampled, col_int_mean, 0, cv::REDUCE_AVG, CV_32F);
    timer_.stop("04_reduce_int");

    timer_.start("04_gaussian_blur");
    cv::GaussianBlur(col_energy_mean, col_energy_mean, cv::Size(101, 1), 0, 0);
    cv::GaussianBlur(col_int_mean, col_int_mean, cv::Size(101, 1), 0, 0);
    timer_.stop("04_gaussian_blur");

    timer_.start("04_repeat");
    cv::repeat(col_energy_mean, Config::getInstance().image.target_h, 1, col_energy_broadcast_);
    cv::repeat(col_int_mean, Config::getInstance().image.target_h, 1, col_int_broadcast_);
    timer_.stop("04_repeat");

    timer_.start("04_struct_ops");
    cv::Mat t_struct_high = col_energy_broadcast_ + calib_.delta_struct;
    cv::Mat struct_fray_mask = struct_energy_ds_ > t_struct_high;
    cv::Mat struct_void_abs_floor = col_energy_broadcast_ * Config::getInstance().cluster.void_energy_ratio;
    cv::Mat struct_void_abs_mask = struct_energy_ds_ < struct_void_abs_floor;
    cv::bitwise_or(struct_fray_mask, struct_void_abs_mask, struct_mask_);
    timer_.stop("04_struct_ops");

    timer_.start("04_int_ops");
    norm_img_ds2_.convertTo(norm_img_ds2_float_, CV_32F);
    cv::Mat t_int_low = col_int_broadcast_ - calib_.delta_int;
    cv::Mat t_int_high = col_int_broadcast_ + calib_.delta_int;
    cv::Mat dark_mask = norm_img_ds2_float_ < t_int_low;
    cv::Mat light_mask = norm_img_ds2_float_ > t_int_high;
    cv::bitwise_or(dark_mask, light_mask, intensity_mask_);
    timer_.stop("04_int_ops");

    timer_.start("04_combine");
    cv::bitwise_or(struct_mask_, intensity_mask_, combined_pixel_mask_);
    combined_pixel_mask_.convertTo(combined_pixel_mask_, CV_8U, 255.0);
    timer_.stop("04_combine");

    timer_.stop("04_zscore_envelope");

    timer_.start("05_struct_morph");
        
    cv::morphologyEx(combined_pixel_mask_, closed_struct_, cv::MORPH_CLOSE, close_kernel_ds_);
    
    cv::Point anchor_center(Config::getInstance().morphology.line_len_ds / 2, 
                            Config::getInstance().morphology.line_len_ds / 2);
    cv::morphologyEx(closed_struct_, morph_struct_, cv::MORPH_OPEN, line_kernel_ds_, 
                     anchor_center, 1, cv::BORDER_CONSTANT, cv::Scalar(0));
    timer_.stop("05_struct_morph");

    timer_.start("06_variance_stream");
    cv::boxFilter(img_ds2_, mean_i_, CV_32F, cv::Size(calib_.V_WIN, calib_.V_WIN));
    cv::sqrBoxFilter(img_ds2_, mean_i2_, CV_32F, cv::Size(calib_.V_WIN, calib_.V_WIN));
    local_variance_ = mean_i2_ - (mean_i_ .mul(mean_i_));

    raw_oil_mask_ = local_variance_ > calib_.var_limit;
    raw_oil_mask_.convertTo(raw_oil_mask_, CV_8U, 255.0);

    cv::Mat oil_closed, oil_opened;
    cv::morphologyEx(raw_oil_mask_, oil_closed_, cv::MORPH_CLOSE, oil_close_kernel_ds_);
    cv::morphologyEx(oil_closed_, oil_opened_, cv::MORPH_OPEN, oil_open_kernel_ds_);

    solid_oil_mask_ = oil_opened_;

    cv::bitwise_or(morph_struct_, solid_oil_mask_, unified_mask_);
    timer_.stop("06_variance_stream");
    
    unified_mask_.copyTo(macro_mask_);
    
    int border_crop = calib_.border_crop;
    macro_mask_.rowRange(0, border_crop).setTo(0);
    macro_mask_.rowRange(Config::getInstance().image.target_h - border_crop, 
                         Config::getInstance().image.target_h).setTo(0);
    macro_mask_.colRange(0, border_crop).setTo(0);
    macro_mask_.colRange(Config::getInstance().image.target_w - border_crop, 
                         Config::getInstance().image.target_w).setTo(0);
    
    timer_.start("07_cc_filtering");

    // Run CC ONCE on the final cropped mask
    cv::Mat final_labels, final_stats, final_centroids;
    int n_final = cv::connectedComponentsWithStats(macro_mask_, final_labels, final_stats, final_centroids, 8, CV_32S);

    last_components_.clear();
    last_components_.reserve(n_final - 1);

    for (int i = 1; i < n_final; ++i) {
        int area = final_stats.at<int>(i, cv::CC_STAT_AREA);
        if (area >= Config::getInstance().cluster.min_cluster_size_ds) {
            DefectComponent comp;
            comp.frame_id = frame_id;
            comp.id = i;
            comp.x = final_stats.at<int>(i, cv::CC_STAT_LEFT);
            comp.y = final_stats.at<int>(i, cv::CC_STAT_TOP);
            comp.width = final_stats.at<int>(i, cv::CC_STAT_WIDTH);
            comp.height = final_stats.at<int>(i, cv::CC_STAT_HEIGHT);
            comp.area = area;
            comp.centroid_x = final_centroids.at<double>(i, 0);
            comp.centroid_y = final_centroids.at<double>(i, 1);
            last_components_.push_back(comp);
        }
    }                     
    
    timer_.stop("07_cc_filtering");

    cv::resize(macro_mask_, final_mask_, input.size(), 0, 0, cv::INTER_NEAREST);

    timer_.stop("00_total");

    if (Config::getInstance().debug.save_level1_maps) {
        static bool debug_dir_created = false;
        if (!debug_dir_created) {
            std::string cmd = "mkdir -p output/debug";
            system(cmd.c_str());
            debug_dir_created = true;
        }

        cv::imwrite("output/debug/01_norm.png", norm_img_ds2_);
        cv::imwrite("output/debug/02_struct_energy.png", struct_energy_ds_);
        cv::imwrite("output/debug/03_struct_mask.png", struct_mask_ * 255);
        cv::imwrite("output/debug/04_intensity_mask.png", intensity_mask_ * 255);
        cv::imwrite("output/debug/05_combined_mask.png", combined_pixel_mask_);
        cv::imwrite("output/debug/06_struct_morph.png", morph_struct_);
        cv::imwrite("output/debug/07_oil_mask.png", solid_oil_mask_);
        cv::imwrite("output/debug/08_unified_mask.png", unified_mask_);
        cv::imwrite("output/debug/09_macro_mask.png", macro_mask_);
        cv::imwrite("output/debug/10_final_mask.png", final_mask_);
    }

    return final_mask_;
}

} // namespace minimind
