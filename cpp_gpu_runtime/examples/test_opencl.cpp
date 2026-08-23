#include <iostream>
#include <dlfcn.h>
#include "MNN/Interpreter.hpp"

int main() {
    void* handle = dlopen("/home/arduino/mnn_test/libMNN_CL.so", RTLD_NOW | RTLD_GLOBAL);

    if (!handle) {
        std::cerr << "FAILED TO LOAD MNN_CL: " << dlerror() << "\n";
        return 1;
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;

    auto runtime = MNN::Interpreter::createRuntime({config});

    if (!runtime.second) {
        std::cerr << "OPENCL RUNTIME FAILED\n";
        return 1;
    }

    std::cout << "OPENCL RUNTIME INITIALIZED\n";
    return 0;
}
