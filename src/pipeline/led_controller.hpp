#pragma once

#include <string>
#include <mutex>

namespace minimind {

// Writes pipeline state to a shared file that the App Lab container polls.
// The container then calls Bridge.call("led_normal"/"led_defect") to switch
// the MCU LED matrix animation. Only writes on state change; the write is
// atomic (.tmp + rename) so the container never reads a partial file.
class LedController {
public:
    explicit LedController(const std::string& state_path);
    ~LedController() = default;

    LedController(const LedController&) = delete;
    LedController& operator=(const LedController&) = delete;

    // state must be "NORMAL" or "DEFECT".
    void setState(const std::string& state);

private:
    std::string state_path_;
    std::string tmp_path_;
    std::string last_state_;
    std::mutex  mtx_;
};

} // namespace minimind
