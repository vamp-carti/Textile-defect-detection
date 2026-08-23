#pragma once

#include <string>
#include <vector>
#include <opencv2/core.hpp>

namespace minimind {

struct CalibrationData {
    double fx = 0.0;
    double fy = 0.0;

    // NWER peak data (new)
    std::vector<std::pair<double, double>> peak_coords_normalized;
    int peak_radius_px = 5;
    int top_n_peaks = 12;

    double theta = 0.0;
    double lambd = 10.0;
    int ksize = 31;
    double sigma = 4.0;
    double gamma = 0.5;
    double K_SIGMA = 3.0;
    int V_WIN = 9;

    double var_limit = 15.0;
    double calib_var_mean = 10.0;
    double calib_var_std = 2.0;

    double calib_struct_mean = 50.0;
    double calib_struct_std = 5.0;

    double calib_int_mean = 128.0;
    double calib_int_std = 10.0;

    // Derived precomputed values
    double lambd_ds = 0.0;
    double sigma_ds = 0.0;
    int ksize_ds = 3;
    double delta_struct = 0.0;
    double delta_int = 0.0;
    int v_win_ds = 3;
    int border_crop = 4;
};

bool loadCalibration(const std::string& path, CalibrationData& calib);

} // namespace minimind
