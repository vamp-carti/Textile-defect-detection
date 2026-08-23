#include "calibration/calibration_compute.hpp"

#include <fstream>
#include <cmath>
#include <algorithm>

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>
#include <nlohmann/json.hpp>

#include "core/config.hpp"

namespace minimind {

namespace {

cv::Mat extractCentralPatch(const cv::Mat& img, int patch_size = 256) {
    int h = img.rows;
    int w = img.cols;
    
    if (h < patch_size || w < patch_size) {
        throw std::runtime_error("Image dimensions are smaller than required patch size");
    }
    
    int cy = h / 2;
    int cx = w / 2;
    int y1 = cy - patch_size / 2;
    int x1 = cx - patch_size / 2;
    
    return img(cv::Rect(x1, y1, patch_size, patch_size)).clone();
}

void computeFFTParams(const cv::Mat& patch, double& fx, double& fy, 
                      double& theta, double& lambd, int& ksize, 
                      double& sigma, double& gamma) {
    int P_SIZE = patch.rows;
    int w = patch.cols;
    int h = patch.rows;
    
    cv::Mat float_patch;
    patch.convertTo(float_patch, CV_32F);
    
    cv::Mat f_transform;
    cv::dft(float_patch, f_transform, cv::DFT_COMPLEX_OUTPUT);
    
    cv::Mat planes[2];
    cv::split(f_transform, planes);
    cv::Mat mag_spectrum;
    cv::magnitude(planes[0], planes[1], mag_spectrum);
    
    mag_spectrum(cv::Rect(0, 0, 15, 15)) = 0;
    mag_spectrum(cv::Rect(0, mag_spectrum.rows - 14, 15, 14)) = 0;
    
    double minVal, maxVal;
    cv::Point minLoc, maxLoc;
    cv::minMaxLoc(mag_spectrum, &minVal, &maxVal, &minLoc, &maxLoc);
    
    int fx_idx = maxLoc.x;
    int fy_idx = maxLoc.y;
    if (fy_idx > P_SIZE / 2) {
        fy_idx -= P_SIZE;
    }
    
    fx = fx_idx * (double)w / P_SIZE;
    fy = fy_idx * (double)h / P_SIZE;
    
    theta = std::atan2(fy, fx);
    lambd = w / std::sqrt(fx * fx + fy * fy);
    
    ksize = static_cast<int>(1.5 * lambd);
    if (ksize % 2 == 0) {
        ksize++;
    }
    
    sigma = 0.35 * lambd;
    gamma = 1.0;
}

void computeVarianceStats(const cv::Mat& patch, int V_WIN, 
                          double& var_mean, double& var_std, double& var_limit) {
    cv::Mat patch_f;
    patch.convertTo(patch_f, CV_32F);
    
    cv::Mat patch_mean, patch_mean2, patch_var;
    cv::boxFilter(patch_f, patch_mean, CV_32F, cv::Size(V_WIN, V_WIN));
    cv::sqrBoxFilter(patch_f, patch_mean2, CV_32F, cv::Size(V_WIN, V_WIN));
    patch_var = patch_mean2 - patch_mean.mul(patch_mean);
    
    cv::Scalar mean, stddev;
    cv::meanStdDev(patch_var, mean, stddev);
    var_mean = mean[0];
    var_std = stddev[0];
    
    double K_SIGMA = 4.0;
    var_limit = var_mean + K_SIGMA * var_std;
}

void computeStructuralStats(const cv::Mat& patch, int ksize, double sigma, 
                            double theta, double lambd, double gamma,
                            double& struct_mean, double& struct_std,
                            double& int_mean, double& int_std) {
    cv::Mat kernel_real = cv::getGaborKernel(
        cv::Size(ksize, ksize), sigma, theta, lambd, gamma, 0, CV_64F
    );
    cv::Mat kernel_imag = cv::getGaborKernel(
        cv::Size(ksize, ksize), sigma, theta, lambd, gamma, CV_PI / 2, CV_64F
    );
    
    kernel_real -= cv::mean(kernel_real)[0];
    kernel_imag -= cv::mean(kernel_imag)[0];
    
    cv::Mat patch_blur;
    cv::GaussianBlur(patch, patch_blur, cv::Size(251, 251), 0);
    cv::Mat patch_norm;
    cv::divide(patch, patch_blur, patch_norm, 128.0);
    
    int pad = ksize / 2;
    cv::Mat padded;
    cv::copyMakeBorder(patch_norm, padded, pad, pad, pad, pad, cv::BORDER_REFLECT101);
    
    cv::Mat f_real_padded, f_imag_padded;
    cv::filter2D(padded, f_real_padded, CV_64F, kernel_real);
    cv::filter2D(padded, f_imag_padded, CV_64F, kernel_imag);
    
    cv::Mat f_real = f_real_padded(cv::Rect(pad, pad, patch.cols, patch.rows));
    cv::Mat f_imag = f_imag_padded(cv::Rect(pad, pad, patch.cols, patch.rows));
    
    cv::Mat struct_energy;
    cv::magnitude(f_real, f_imag, struct_energy);
    
    cv::Scalar struct_mean_s, struct_std_s;
    cv::meanStdDev(struct_energy, struct_mean_s, struct_std_s);
    struct_mean = struct_mean_s[0];
    struct_std = struct_std_s[0];
    
    cv::Scalar int_mean_s, int_std_s;
    cv::meanStdDev(patch_norm, int_mean_s, int_std_s);
    int_mean = int_mean_s[0];
    int_std = int_std_s[0];
}

std::vector<std::pair<double, double>> extractSpectralPeaks(const cv::Mat& patch, 
                                                             int top_n = 12, 
                                                             int dc_radius = 8) {
    int P_SIZE = patch.rows;
    cv::Mat float_patch;
    patch.convertTo(float_patch, CV_32F);
    
    cv::Mat f_2d;
    cv::dft(float_patch, f_2d, cv::DFT_COMPLEX_OUTPUT);
    
    int cx = f_2d.cols / 2;
    int cy = f_2d.rows / 2;
    
    cv::Mat q0(f_2d, cv::Rect(0, 0, cx, cy));
    cv::Mat q1(f_2d, cv::Rect(cx, 0, cx, cy));
    cv::Mat q2(f_2d, cv::Rect(0, cy, cx, cy));
    cv::Mat q3(f_2d, cv::Rect(cx, cy, cx, cy));
    
    cv::Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
    
    cv::Mat planes[2];
    cv::split(f_2d, planes);
    cv::Mat mag_sq;
    cv::magnitude(planes[0], planes[1], mag_sq);
    mag_sq = mag_sq.mul(mag_sq);
    
    cv::Mat mask = cv::Mat::ones(mag_sq.size(), CV_8U);
    cv::circle(mask, cv::Point(cx, cy), dc_radius, 0, -1);
    mag_sq.setTo(0, mask == 0);
    
    cv::Mat mag_flat = mag_sq.reshape(1, 1);
    cv::Mat sorted_indices;
    cv::sortIdx(mag_flat, sorted_indices, cv::SORT_EVERY_ROW | cv::SORT_DESCENDING);
    
    std::vector<std::pair<double, double>> peak_coords;
    int num_peaks = std::min(top_n, sorted_indices.cols);
    
    for (int i = 0; i < num_peaks; ++i) {
        int idx = sorted_indices.at<int>(0, i);
        int py = idx / mag_sq.cols;
        int px = idx % mag_sq.cols;
        
        double norm_y = (double)(py - cy) / P_SIZE;
        double norm_x = (double)(px - cx) / P_SIZE;
        peak_coords.push_back({norm_y, norm_x});
    }
    
    return peak_coords;
}

} // anonymous namespace

CalibrationData CalibrationComputer::computeFromFrame(const cv::Mat& grayscale_frame) {
    if (grayscale_frame.empty()) {
        throw std::runtime_error("Input frame is empty");
    }
    
    if (grayscale_frame.channels() != 1) {
        throw std::runtime_error("Input must be grayscale (1 channel)");
    }
    
    int P_SIZE = 256;
    cv::Mat patch = extractCentralPatch(grayscale_frame, P_SIZE);
    
    double fx, fy, theta, lambd, sigma;
    int ksize;
    double gamma = 1.0;
    computeFFTParams(patch, fx, fy, theta, lambd, ksize, sigma, gamma);
    
    int V_WIN = 15;
    double var_mean, var_std, var_limit;
    computeVarianceStats(patch, V_WIN, var_mean, var_std, var_limit);
    double K_SIGMA = 4.0;
    
    double struct_mean, struct_std, int_mean, int_std;
    computeStructuralStats(patch, ksize, sigma, theta, lambd, gamma,
                           struct_mean, struct_std, int_mean, int_std);
    
    std::vector<std::pair<double, double>> peak_coords = extractSpectralPeaks(patch, 12, 8);
    
    CalibrationData calib;
    
    calib.fx = fx;
    calib.fy = fy;
    calib.theta = theta;
    calib.lambd = lambd;
    calib.ksize = ksize;
    calib.sigma = sigma;
    calib.gamma = gamma;
    
    calib.V_WIN = V_WIN;
    calib.K_SIGMA = K_SIGMA;
    calib.calib_var_mean = var_mean;
    calib.calib_var_std = var_std;
    calib.var_limit = var_limit;
    
    calib.calib_struct_mean = struct_mean;
    calib.calib_struct_std = struct_std;
    calib.calib_int_mean = int_mean;
    calib.calib_int_std = int_std;
    
    calib.peak_coords_normalized = peak_coords;
    calib.peak_radius_px = 5;
    calib.top_n_peaks = 12;
    
    int ds_factor = Config::getInstance().image.ds_factor;
    
    calib.lambd_ds = calib.lambd / ds_factor;
    calib.sigma_ds = calib.sigma / ds_factor;
    
    int computed_ksize_ds = static_cast<int>(calib.ksize / ds_factor);
    if (computed_ksize_ds % 2 == 0) {
        computed_ksize_ds += 1;
    }
    calib.ksize_ds = std::max(3, computed_ksize_ds);
    
    calib.delta_struct = calib.K_SIGMA * calib.calib_struct_std;
    calib.delta_int = calib.K_SIGMA * calib.calib_int_std;
    
    int computed_v_win = static_cast<int>(calib.V_WIN / ds_factor);
    if (computed_v_win % 2 == 0) {
        computed_v_win += 1;
    }
    calib.v_win_ds = std::max(3, computed_v_win);
    
    calib.border_crop = std::max(4, calib.ksize_ds / 2);
    
    return calib;
}

bool CalibrationComputer::saveToJson(const std::string& path, const CalibrationData& data) {
    nlohmann::json j;
    
    j["fx"] = data.fx;
    j["fy"] = data.fy;
    j["theta"] = data.theta;
    j["lambd"] = data.lambd;
    j["ksize"] = data.ksize;
    j["sigma"] = data.sigma;
    j["gamma"] = data.gamma;
    
    j["K_SIGMA"] = data.K_SIGMA;
    j["V_WIN"] = data.V_WIN;
    j["calib_var_mean"] = data.calib_var_mean;
    j["calib_var_std"] = data.calib_var_std;
    j["var_limit"] = data.var_limit;
    
    j["calib_struct_mean"] = data.calib_struct_mean;
    j["calib_struct_std"] = data.calib_struct_std;
    j["calib_int_mean"] = data.calib_int_mean;
    j["calib_int_std"] = data.calib_int_std;
    
    j["peak_coords_normalized"] = nlohmann::json::array();
    for (const auto& p : data.peak_coords_normalized) {
        j["peak_coords_normalized"].push_back({p.first, p.second});
    }
    j["peak_radius_px"] = data.peak_radius_px;
    
    std::ofstream file(path);
    if (!file.is_open()) {
        return false;
    }
    file << j.dump(4);
    return true;
}

bool CalibrationComputer::saveVisualization(const std::string& image_path,
                                             const cv::Mat& original_frame,
                                             const cv::Mat& patch) {
    if (original_frame.empty() || patch.empty()) {
        return false;
    }
    
    int h = original_frame.rows;
    int w = original_frame.cols;
    int patch_size = patch.rows;
    int cy = h / 2;
    int cx = w / 2;
    int y1 = cy - patch_size / 2;
    int x1 = cx - patch_size / 2;
    
    cv::Mat vis_img;
    cv::cvtColor(original_frame, vis_img, cv::COLOR_GRAY2BGR);
    cv::rectangle(vis_img, cv::Point(x1, y1), cv::Point(x1 + patch_size, y1 + patch_size), 
                  cv::Scalar(0, 0, 255), 2);
    
    return cv::imwrite(image_path, vis_img);
}

} // namespace minimind
