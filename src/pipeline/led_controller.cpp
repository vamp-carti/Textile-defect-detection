#include "pipeline/led_controller.hpp"

#include <fstream>
#include <cstdio>
#include <iostream>

namespace minimind {

LedController::LedController(const std::string& state_path)
    : state_path_(state_path)
    , tmp_path_(state_path + ".tmp")
{}

void LedController::setState(const std::string& state) {
    // The App Lab watcher may read this file concurrently, so publish each
    // state through a same-filesystem rename rather than an in-place write.
    std::lock_guard<std::mutex> lk(mtx_);

    if (state == last_state_) return;  // no-op on repeat

    // Write to tmp, then rename. rename() is atomic on the same filesystem.
    {
        std::ofstream f(tmp_path_, std::ios::trunc);
        if (!f.is_open()) {
            std::cerr << "[LedController] Failed to open " << tmp_path_ << std::endl;
            return;
        }
        f << state << "\n";
    }

    if (std::rename(tmp_path_.c_str(), state_path_.c_str()) != 0) {
        std::cerr << "[LedController] rename failed: "
                  << tmp_path_ << " -> " << state_path_ << std::endl;
        return;
    }

    last_state_ = state;
}

} // namespace minimind
