#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <dlfcn.h>
#include "MNN/Interpreter.hpp"

int main() {
    void* cl = dlopen("/home/arduino/mnn_test/libMNN_CL.so",
                      RTLD_NOW | RTLD_GLOBAL);

    if (!cl) {
        std::cerr << "Failed to load OpenCL backend: "
                  << dlerror() << "\n";
        return 1;
    }

    auto net = MNN::Interpreter::createFromFile(
        "/home/arduino/mnn_test/mobilenetv4_conv_small_fp32.mnn");

    if (!net) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_CPU;
    config.numThread = 1;

    auto runtime = MNN::Interpreter::createRuntime({config});

    if (runtime.first.find(MNN_FORWARD_CPU) == runtime.first.end()) {
        std::cerr << "CPU runtime was not created\n";
        return 1;
    }

    auto session = net->createSession(config, runtime);

    if (!session) {
        std::cerr << "OpenCL session creation failed\n";
        return 1;
    }

    std::cout << "Warming up...\n";

    for (int i = 0; i < 20; ++i) {
        net->runSession(session);
    }

    constexpr int N = 1000;
    std::vector<double> times;
    times.reserve(N);

    std::cout << "Benchmarking " << N << " runs...\n";

    for (int i = 0; i < N; ++i) {
        auto start = std::chrono::steady_clock::now();

        net->runSession(session);

        auto end = std::chrono::steady_clock::now();

        double ms =
            std::chrono::duration<double, std::milli>(end - start).count();

        times.push_back(ms);
    }

    std::sort(times.begin(), times.end());

    double mean =
        std::accumulate(times.begin(), times.end(), 0.0) / N;

    double p50 = times[N / 2];
    double p95 = times[static_cast<int>(N * 0.95)];
    double min = times.front();
    double max = times.back();

    std::cout << "\n=== MNN OpenCL Benchmark ===\n";
    std::cout << "Runs:   " << N << "\n";
    std::cout << "Min:    " << min << " ms\n";
    std::cout << "Median: " << p50 << " ms\n";
    std::cout << "Mean:   " << mean << " ms\n";
    std::cout << "P95:    " << p95 << " ms\n";
    std::cout << "Max:    " << max << " ms\n";

    return 0;
}
