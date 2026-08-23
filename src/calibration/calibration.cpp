#include "calibration/calibration.hpp"

#include <fstream>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include "core/config.hpp"

namespace minimind {

bool loadCalibration(const std::string& path, CalibrationData& calib) {
    cv::FileStorage fs(path, cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!fs.isOpened()) {
        std::cerr << "Failed to open calibration file: " << path << std::endl;
        return false;
    }

    fs["theta"] >> calib.theta;
    fs["lambd"] >> calib.lambd;
    fs["ksize"] >> calib.ksize;
    fs["sigma"] >> calib.sigma;
    fs["gamma"] >> calib.gamma;
    fs["K_SIGMA"] >> calib.K_SIGMA;
    fs["V_WIN"] >> calib.V_WIN;
    fs["var_limit"] >> calib.var_limit;
    fs["calib_var_mean"] >> calib.calib_var_mean;
    fs["calib_var_std"] >> calib.calib_var_std;
    fs["calib_struct_mean"] >> calib.calib_struct_mean;
    fs["calib_struct_std"] >> calib.calib_struct_std;
    fs["calib_int_mean"] >> calib.calib_int_mean;
    fs["calib_int_std"] >> calib.calib_int_std;
    fs.release();

    // Precompute derived parameters
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

    return true;
}

} // namespace minimind
