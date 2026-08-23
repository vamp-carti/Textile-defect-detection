#include "benchmark.hpp"

#include <iostream>
#include <iomanip>

namespace minimind {

void BenchmarkTimer::start(const std::string& stage_name) {
    if (!Config::getInstance().isPerformanceTimingEnabled()) {
        return;
    }
    start_times_[stage_name] = std::chrono::steady_clock::now();
}

void BenchmarkTimer::stop(const std::string& stage_name) {
    if (!Config::getInstance().isPerformanceTimingEnabled()) {
        return;
    }
    auto end_time = std::chrono::steady_clock::now();
    if (start_times_.find(stage_name) != start_times_.end()) {
        auto duration = std::chrono::duration<double, std::milli>(end_time - start_times_[stage_name]).count();
        latencies_[stage_name].update(duration);
    }
}

void BenchmarkTimer::reset() {
    if (!Config::getInstance().isPerformanceTimingEnabled()) {
        return;
    }
    start_times_.clear();
    latencies_.clear();
}

void BenchmarkTimer::report() const {
    if (!Config::getInstance().isPerformanceTimingEnabled()) {
        return;
    }
    
    std::cout << "\n============================================================\n";
    std::cout << "STAGE LATENCY BENCHMARK REPORT\n";
    std::cout << "============================================================\n";
    std::cout << std::left << std::setw(30) << "Stage"
              << std::right << std::setw(12) << "Avg (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "Max (ms)" << "\n";
    std::cout << "------------------------------------------------------------\n";

    double total_avg = 0.0;
    if (latencies_.find("00_total") != latencies_.end()) {
        total_avg = latencies_.at("00_total").average();
    }

    for (const auto& [stage, latency] : latencies_) {
        if (stage == "00_total") continue;
        double avg = latency.average();
        std::cout << std::left << std::setw(30) << stage
                  << std::right << std::setw(12) << std::fixed << std::setprecision(3) << avg
                  << std::setw(12) << latency.min_time_ms
                  << std::setw(12) << latency.max_time_ms << "\n";
    }
    std::cout << "------------------------------------------------------------\n";
    if (latencies_.find("00_total") != latencies_.end()) {
        const auto& tot = latencies_.at("00_total");
        std::cout << std::left << std::setw(30) << "00_total"
                  << std::right << std::setw(12) << tot.average()
                  << std::setw(12) << tot.min_time_ms
                  << std::setw(12) << tot.max_time_ms << "\n";
    }
    std::cout << "============================================================\n";
}

} // namespace minimind
