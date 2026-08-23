#include <iostream>
#include <dlfcn.h>
#include "MNN/Interpreter.hpp"
#include "source/core/Backend.hpp"

int main() {
    void* cl = dlopen("/home/arduino/mnn_test/libMNN_CL.so",
                      RTLD_NOW | RTLD_GLOBAL);

    if (!cl) {
        std::cerr << "OpenCL backend load failed: " << dlerror() << "\n";
        return 1;
    }

    auto net = MNN::Interpreter::createFromFile(
        "/home/arduino/mnn_test/mobilenetv4_conv_small_fp32.mnn");

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;
    config.numThread = 1;

    auto runtime = MNN::Interpreter::createRuntime({config});

    auto it = runtime.first.find(MNN_FORWARD_OPENCL);
    if (it == runtime.first.end()) {
        std::cerr << "FAILED: OpenCL runtime was not created\n";
        return 1;
    }

    std::cout << "OpenCL runtime created\n";

    auto session = net->createSession(config, runtime);
    if (!session) {
        std::cerr << "FAILED: OpenCL session creation\n";
        return 1;
    }

    std::cout << "Running inference...\n";
    net->runSession(session);

    float gpu_ms = it->second->onGetLastGpuTimeMs();

    std::cout << "GPU time: " << gpu_ms << " ms\n";

    if (gpu_ms >= 0.0f) {
        std::cout << "GPU EXECUTION CONFIRMED\n";
    } else {
        std::cout << "GPU TIME UNAVAILABLE\n";
    }

    return 0;
}
