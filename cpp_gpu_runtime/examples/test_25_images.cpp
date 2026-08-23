#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <dlfcn.h>

#include "MNN/Interpreter.hpp"
#include "MNN/Tensor.hpp"

const int W = 224;
const int H = 224;
const int C = 3;

const std::vector<std::string> classes = {
    "cuts", "hole", "lint", "normal", "oil"
};

float softmax_exp(float x) {
    return std::exp(x);
}

int main() {
    // Load OpenCL backend explicitly
    void* cl = dlopen(
        "/home/arduino/mnn_test/libMNN_CL.so",
        RTLD_NOW | RTLD_GLOBAL
    );

    if (!cl) {
        std::cerr << "FAILED OpenCL: " << dlerror() << "\n";
        return 1;
    }

    auto net = MNN::Interpreter::createFromFile(
        "/home/arduino/mnn_test/mobilenetv4_conv_small_fp32.mnn"
    );

    if (!net) {
        std::cerr << "FAILED: model load\n";
        return 1;
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;
    config.numThread = 1;

    auto session = net->createSession(config);

    if (!session) {
        std::cerr << "FAILED: OpenCL session\n";
        return 1;
    }

    auto input = net->getSessionInput(session, nullptr);

    if (!input) {
        std::cerr << "FAILED: input\n";
        return 1;
    }

    std::cout << "Input: "
              << input->length(0) << "x"
              << input->length(1) << "x"
              << input->length(2) << "x"
              << input->length(3) << "\n";

    int total = 0;
    int correct = 0;

    std::cout << "\n";
    std::cout << std::left
              << std::setw(28) << "IMAGE"
              << std::setw(10) << "TRUE"
              << std::setw(10) << "PRED"
              << std::setw(10) << "CONF"
              << "RESULT\n";

    std::cout << std::string(75, '-') << "\n";

    for (const auto& true_class : classes) {
        for (int n = 1; n <= 5; ++n) {

            std::string path =
                "/home/arduino/ArduinoApps/defect-detection/test/" +
                (true_class == "hole" ? "holes" : true_class) + "/" + std::to_string(n) + ".png";

            int iw, ih, channels;

            unsigned char* image = stbi_load(
                path.c_str(),
                &iw, &ih, &channels,
                3
            );

            if (!image) {
                std::cerr << "\nFAILED TO LOAD: " << path
                          << "\nReason: " << stbi_failure_reason() << "\n";
                continue;
            }

            // Resize RGB image to 224x224 using nearest-neighbor.
            // We explicitly do this here so preprocessing is deterministic.
            std::vector<unsigned char> resized(W * H * 3);

            for (int y = 0; y < H; ++y) {
                int sy = y * ih / H;

                for (int x = 0; x < W; ++x) {
                    int sx = x * iw / W;

                    for (int c = 0; c < 3; ++c) {
                        resized[(y * W + x) * 3 + c] =
                            image[(sy * iw + sx) * 3 + c];
                    }
                }
            }

            stbi_image_free(image);

            // MNN input is NCHW FP32.
            // Same normalization as PyTorch:
            //
            // x = (pixel / 255 - mean) / std
            //
            const float mean[3] = {
                0.485f, 0.456f, 0.406f
            };

            const float std[3] = {
                0.229f, 0.224f, 0.225f
            };

            auto input_host = new MNN::Tensor(
                input,
                MNN::Tensor::CAFFE
            );

            float* dst = input_host->host<float>();

            for (int c = 0; c < 3; ++c) {
                for (int y = 0; y < H; ++y) {
                    for (int x = 0; x < W; ++x) {

                        int idx =
                            (y * W + x) * 3 + c;

                        float pixel =
                            resized[idx] / 255.0f;

                        dst[
                            c * H * W +
                            y * W +
                            x
                        ] = (pixel - mean[c]) / std[c];
                    }
                }
            }

            input->copyFromHostTensor(input_host);
            delete input_host;

            // Run inference
            net->runSession(session);

            auto output =
                net->getSessionOutput(session, nullptr);

            if (!output) {
                std::cerr << "FAILED: output\n";
                return 1;
            }

            auto output_host = new MNN::Tensor(
                output,
                MNN::Tensor::CAFFE
            );

            output->copyToHostTensor(output_host);

            float* logits = output_host->host<float>();

            // Softmax
            float max_logit = logits[0];

            for (int i = 1; i < 5; ++i)
                max_logit = std::max(max_logit, logits[i]);

            float sum = 0.0f;
            float probs[5];

            for (int i = 0; i < 5; ++i) {
                probs[i] =
                    std::exp(logits[i] - max_logit);
                sum += probs[i];
            }

            for (int i = 0; i < 5; ++i)
                probs[i] /= sum;

            int pred =
                std::max_element(probs, probs + 5) - probs;

            int true_idx =
                std::find(
                    classes.begin(),
                    classes.end(),
                    true_class
                ) - classes.begin();

            bool ok = pred == true_idx;

            if (ok)
                correct++;

            total++;

            std::string filename =
                (true_class == "hole" ? "holes" : true_class) + "/" + std::to_string(n) + ".png";

            std::cout
                << std::left
                << std::setw(28) << filename
                << std::setw(10) << true_class
                << std::setw(10) << classes[pred]
                << std::fixed << std::setprecision(4)
                << std::setw(10) << probs[pred]
                << (ok ? "OK" : "WRONG")
                << "\n";

            delete output_host;
        }
    }

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "MNN 25-IMAGE TEST\n";
    std::cout << "========================================\n";
    std::cout << "Correct: " << correct << "/" << total << "\n";

    if (total > 0) {
        std::cout << "Accuracy: "
                  << std::fixed
                  << std::setprecision(2)
                  << (100.0 * correct / total)
                  << "%\n";
    }

    return 0;
}
