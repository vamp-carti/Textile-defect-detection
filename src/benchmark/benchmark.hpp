#pragma once

#include <chrono>
#include <string>
#include <unordered_map>

#include "core/config.hpp"

namespace minimind {

struct StageLatency {
    double total_time_ms = 0.0;
    double min_time_ms = 1e9;
    double max_time_ms = 0.0;
    int count = 0;

    void update(double time_ms) {
        total_time_ms += time_ms;
        if (time_ms < min_time_ms) min_time_ms = time_ms;
        if (time_ms > max_time_ms) max_time_ms = time_ms;
        count++;
    }

    double average() const {
        return count > 0 ? total_time_ms / count : 0.0;
    }
};

class BenchmarkTimer {
public:
    void start(const std::string& stage_name);
    void stop(const std::string& stage_name);
    void reset();
    void report() const;

private:
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> start_times_;
    std::unordered_map<std::string, StageLatency> latencies_;
};

} // namespace minimind
