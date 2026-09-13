#include <iostream>
#include "pipeline/io_worker.hpp"
#include "roi_inference.hpp"
#include <iomanip>
#include <unistd.h>
#include <fcntl.h>
#include <memory>
#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <stdexcept>
#include <vector>
#include <iomanip>
#include <fstream>
#include <MNN/Tensor.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include "third_party/edge_impulse/sdk/classifier/ei_run_classifier.h"
#include "third_party/edge_impulse/model/tflite/tflite_learn_1101485_3.h"
#include <string>
#include "core/defect_report.hpp" 
#include <chrono>

extern std::vector<DefectReport> g_defect_reports;
extern std::mutex g_defect_mutex;

namespace minimind {

ROIInference::ROIInference(
    const std::string& model_path,
    const std::string& opencl_backend_path,
    float defect_threshold
)
    : defect_threshold_(defect_threshold)
{
    // The OpenCL backend is loaded explicitly because MNN discovers it through
    // the shared object at runtime on the target board.
    // Save original stderr file descriptor
    int stderr_backup = -1;
    int null_fd = -1;

    // If NOT in debug mode, suppress MNN output at file descriptor level
    if (!Config::getInstance().isDebugMode()) {
        stderr_backup = dup(STDERR_FILENO);
        null_fd = open("/dev/null", O_WRONLY);
        dup2(null_fd, STDERR_FILENO);
    }

    opencl_handle_ = dlopen(opencl_backend_path.c_str(), RTLD_NOW | RTLD_GLOBAL);

    if (!opencl_handle_) {
        throw std::runtime_error(
            "Failed to load OpenCL backend: " + std::string(dlerror())
        );
    }

    interpreter_ = MNN::Interpreter::createFromFile(model_path.c_str());

    if (!interpreter_) {
        dlclose(opencl_handle_);
        opencl_handle_ = nullptr;
        throw std::runtime_error("Failed to load MNN model: " + model_path);
    }

    MNN::ScheduleConfig config;
    config.type = MNN_FORWARD_OPENCL;
    config.numThread = 1;

    session_ = interpreter_->createSession(config);

    if (Config::getInstance().isDebugMode()) {
        std::cout << "MNN OpenCL session created: " << session_ << std::endl;
    }

    if (!session_) {
        delete interpreter_;
        interpreter_ = nullptr;
        dlclose(opencl_handle_);
        opencl_handle_ = nullptr;
        throw std::runtime_error("Failed to create MNN OpenCL session");
    }

    input_tensor_ = interpreter_->getSessionInput(session_, nullptr);

    if (!input_tensor_) {
        interpreter_->releaseSession(session_);
        session_ = nullptr;
        delete interpreter_;
        interpreter_ = nullptr;
        dlclose(opencl_handle_);
        opencl_handle_ = nullptr;
        throw std::runtime_error("Failed to get MNN input tensor");
    }

    if (Config::getInstance().isDebugMode()) {
        std::cout << "ROIInference initialized\n"
                  << "  Model: " << model_path << "\n"
                  << "  Backend: OpenCL\n"
                  << "  Input: 1x3x224x224 FP32\n"
                  << "  Defect threshold: " << defect_threshold_ << "\n";
    }

    // Warm up the CPU inference path (XNNPACK delegate, caches, etc.)
    warmup();

    // Restore stderr
    if (!Config::getInstance().isDebugMode()) {
        dup2(stderr_backup, STDERR_FILENO);
        close(null_fd);
        close(stderr_backup);
    }
    
}

ROIInference::~ROIInference()
{
    if (interpreter_ && session_) {
        interpreter_->releaseSession(session_);
        session_ = nullptr;
    }

    if (interpreter_) {
        delete interpreter_;
        interpreter_ = nullptr;
    }

    if (opencl_handle_) {
        dlclose(opencl_handle_);
        opencl_handle_ = nullptr;
    }
}

BenchmarkTimer& ROIInference::getBenchmarkTimer()
{
    return benchmark_;
}

void ROIInference::prepareInput(const cv::Mat& rgb, float* input)
{
    // The MNN model expects normalized, channel-first RGB tensors in 224x224
    // form. The CPU Edge Impulse path has a separate packed-RGB contract.
    if (rgb.empty()) {
        throw std::runtime_error("ROI image is empty");
    }

    if (rgb.channels() != 3) {
        throw std::runtime_error("ROI must have 3 channels");
    }

    constexpr float mean[3] = {0.485f, 0.456f, 0.406f};
    constexpr float stddev[3] = {0.229f, 0.224f, 0.225f};

    const int plane_size = INPUT_SIZE * INPUT_SIZE;

    for (int y = 0; y < INPUT_SIZE; ++y) {
        const cv::Vec3b* row = rgb.ptr<cv::Vec3b>(y);

        for (int x = 0; x < INPUT_SIZE; ++x) {
            const cv::Vec3b& pixel = row[x];

            for (int c = 0; c < 3; ++c) {
                const float value = static_cast<float>(pixel[c]) / 255.0f;
                input[c * plane_size + y * INPUT_SIZE + x] = (value - mean[c]) / stddev[c];
            }
        }
    }
}

float ROIInference::computeDefectProbability(const std::vector<float>& probabilities) const
{
    // Class order is cuts=0, hole=1, lint=2, normal=3, oil=4 in both model
    // exports. Only cuts, holes, and oil are production defects; lint and
    // normal remain valid classes but must not contribute to this sum.
    return probabilities[CUTS_IDX] + probabilities[HOLE_IDX] + probabilities[OIL_IDX];
}

void ROIInference::ensureBatchCapacity(int batch_size)
{
    int input_elements = batch_size * 3 * INPUT_SIZE * INPUT_SIZE;
    int output_elements = batch_size * NUM_CLASSES;

    bool need_realloc = (batch_size > current_max_batch_);

    if (need_realloc) {
        batch_input_buffer_.resize(input_elements);
        batch_output_buffer_.resize(output_elements);

        std::vector<int> input_shape = {batch_size, 3, INPUT_SIZE, INPUT_SIZE};
        std::vector<int> output_shape = {batch_size, NUM_CLASSES};

        batch_host_input_.reset();
        batch_host_output_.reset();

        batch_host_input_ = std::shared_ptr<MNN::Tensor>(
            MNN::Tensor::create<float>(input_shape, batch_input_buffer_.data(), MNN::Tensor::CAFFE)
        );

        batch_host_output_ = std::shared_ptr<MNN::Tensor>(
            MNN::Tensor::create<float>(output_shape, batch_output_buffer_.data(), MNN::Tensor::CAFFE)
        );

        current_max_batch_ = batch_size;

        if (Config::getInstance().isDebugMode()) {
            std::cout << "[DEBUG] Batch capacity allocated for " << batch_size << " ROIs" << std::endl;
            std::cout << "  Input buffer: " << input_elements << " floats ("
                      << (input_elements * sizeof(float)) / 1024.0 / 1024.0 << " MB)" << std::endl;
            std::cout << "  Output buffer: " << output_elements << " floats" << std::endl;
        }
    }
}

DefectImages ROIInference::saveDefectFrame(const ROIResult& roi, const ROIInferenceResult& result,
                                           const cv::Mat& original_frame, const std::string& source)
{
    DefectImages out;

    // ============================================================
    // 1. STORE DEFECT REPORT FOR CSV
    // ============================================================
    DefectReport report;
    report.frame_id = roi.frame_id;
    report.roi_id = roi.box.number;
    report.predicted_class = result.predicted_label;
    report.confidence = result.confidence;
    report.defect_probability = result.defect_probability;
    report.timestamp = std::chrono::system_clock::now();

    {
        std::lock_guard<std::mutex> lock(g_defect_mutex);
        g_defect_reports.push_back(report);
    }

    // ============================================================
    // 2. CHECK IF SAVING IS ENABLED
    // ============================================================
    if (!Config::getInstance().production.save_defects) {
        if (Config::getInstance().isDebugMode()) {
            std::cout << "[DEBUG] save_defects is FALSE, not producing images" << std::endl;
        }
        return out;   // empty DefectImages
    }

    // ============================================================
    // 3. SETUP OUTPUT DIRECTORIES
    // ============================================================
    std::string base_dir = "output/";
    std::string rois_dir = base_dir + "rois/";
    std::string frames_dir = base_dir + "frames/";
    std::string cmd = "mkdir -p " + rois_dir + " " + frames_dir;
    system(cmd.c_str());

    // ============================================================
    // 4. FORMAT FRAME NUMBER
    // ============================================================
    std::stringstream ss_frame;
    ss_frame << std::setw(6) << std::setfill('0') << roi.frame_id;
    std::string frame_str = ss_frame.str();

    // ============================================================
    // 5. BUILD ROI CROP
    // ============================================================
    out.roi_path = rois_dir + "frame_" + frame_str + "_roi_" + std::to_string(roi.box.number) + ".jpg";

    cv::cvtColor(roi.image, out.roi_display, cv::COLOR_RGB2BGR);

    std::string label = result.predicted_label + " " +
                        std::to_string((int)(result.confidence * 100)) + "%[" + source + "]";
    cv::putText(out.roi_display, label,
                cv::Point(10, 30),
                cv::FONT_HERSHEY_SIMPLEX, 0.6,
                cv::Scalar(0, 0, 255), 2);

    // ============================================================
    // 6. BUILD FULL FRAME WITH ROI BOX OVERLAY
    // ============================================================
    if (!original_frame.empty()) {
        out.full_path = frames_dir + "frame_" + frame_str + "_full.jpg";

        cv::cvtColor(original_frame, out.full_display, cv::COLOR_RGB2BGR);

        cv::rectangle(out.full_display,
                      cv::Rect(roi.box.x, roi.box.y, roi.box.width, roi.box.height),
                      cv::Scalar(0, 0, 255), 3);

        cv::putText(out.full_display, label,
                    cv::Point(roi.box.x, roi.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8,
                    cv::Scalar(0, 0, 255), 2);
    } else {
        if (Config::getInstance().isDebugMode()) {
            std::cout << "[DEBUG] original_frame is EMPTY, skipping full frame build" << std::endl;
        }
    }

    // ============================================================
    // 7. LOG (work has been produced, not yet written)
    // ============================================================
    std::cout << "[DEFECT SCHEDULED] Frame: " << roi.frame_id
              << " | ROI: " << roi.box.number
              << " | Class: " << result.predicted_label
              << " | Confidence: " << std::fixed << std::setprecision(1) << (result.confidence * 100) << "%"
              << " | ROI path: " << out.roi_path
              << (out.full_path.empty() ? "" : " | Full path: " + out.full_path)
              << std::endl;

    return out;
}

ROIInferenceResult ROIInference::infer(const ROIResult& roi)
{
    benchmark_.start("08b_tensor_alloc");
    MNN::Tensor host_input(input_tensor_, MNN::Tensor::CAFFE);
    benchmark_.stop("08b_tensor_alloc");

    float* input = host_input.host<float>();

    if (!input) {
        throw std::runtime_error("Failed to access MNN input tensor");
    }

    benchmark_.start("08c_preprocess");
    prepareInput(roi.image, input);
    benchmark_.stop("08c_preprocess");

    benchmark_.start("08d_copy_to_device");
    input_tensor_->copyFromHostTensor(&host_input);
    benchmark_.stop("08d_copy_to_device");

    benchmark_.start("08a_mnn_inference");
    interpreter_->runSession(session_);
    benchmark_.stop("08a_mnn_inference");

    benchmark_.start("08e_get_output");
    MNN::Tensor* output_tensor = interpreter_->getSessionOutput(session_, nullptr);

    if (!output_tensor) {
        throw std::runtime_error("Failed to get MNN output tensor");
    }
    benchmark_.stop("08e_get_output");

    benchmark_.start("08f_output_alloc");
    MNN::Tensor host_output(output_tensor, MNN::Tensor::CAFFE);
    benchmark_.stop("08f_output_alloc");

    benchmark_.start("08g_copy_to_host");
    output_tensor->copyToHostTensor(&host_output);
    benchmark_.stop("08g_copy_to_host");

    benchmark_.start("08h_postprocess");
    const float* logits = host_output.host<float>();

    if (!logits) {
        throw std::runtime_error("Failed to access MNN output");
    }

    float max_logit = logits[0];
    for (int i = 1; i < NUM_CLASSES; ++i) {
        max_logit = std::max(max_logit, logits[i]);
    }

    std::vector<float> probabilities(NUM_CLASSES);
    float sum = 0.0f;

    for (int i = 0; i < NUM_CLASSES; ++i) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }

    for (float& p : probabilities) {
        p /= sum;
    }

    const int predicted_class = static_cast<int>(
        std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin()
    );

    const float confidence = probabilities[predicted_class];
    const float defect_probability = computeDefectProbability(probabilities);
    const bool predicted_is_non_defect =
        (predicted_class == LINT_IDX) || (predicted_class == NORMAL_IDX);

    // The low production threshold favors recall. A high-confidence lint or
    // normal argmax is still treated as non-defect by the confidence guard.
    const bool defect = (defect_probability >= defect_threshold_) &&
                        (!predicted_is_non_defect || confidence < 0.80f);

    benchmark_.stop("08h_postprocess");

    ROIInferenceResult result = {
        roi.frame_id,
        roi.box,
        predicted_class,
        CLASS_NAMES[predicted_class],
        defect_probability,
        confidence,
        defect
    };

    if (defect) {
        // Single-ROI path: no async worker available. Write synchronously.
        DefectImages imgs = saveDefectFrame(roi, result, m_current_frame, "GPU");
        if (!imgs.roi_display.empty())  cv::imwrite(imgs.roi_path,  imgs.roi_display);
        if (!imgs.full_display.empty()) cv::imwrite(imgs.full_path, imgs.full_display);
    }

    return result;
}

std::vector<ROIInferenceResult> ROIInference::process(const std::vector<ROIResult>& rois)
{
    std::vector<ROIInferenceResult> results;

    if (rois.empty())
        return results;

    results.reserve(rois.size());

    benchmark_.start("08_roi_inference");

    int current_frame_id = rois.front().frame_id;
    bool frame_has_defect = false;

    benchmark_.start("08i_loop_overhead");

    for (const auto& roi : rois) {
        if (roi.frame_id != current_frame_id) {
            current_frame_id = roi.frame_id;
            frame_has_defect = false;
        }

        if (frame_has_defect)
            continue;

        ROIInferenceResult result = infer(roi);
        results.push_back(result);

        if (result.defect) {
            frame_has_defect = true;
        }
    }

    benchmark_.stop("08i_loop_overhead");
    benchmark_.stop("08_roi_inference");

    return results;
}

std::vector<ROIInferenceResult> ROIInference::processBatch(const std::vector<ROIResult>& rois, const cv::Mat& original_frame)
{
    std::vector<ROIInferenceResult> results;

    if (rois.empty())
        return results;

    int total_batch_size = static_cast<int>(rois.size());
    results.reserve(total_batch_size);

    int num_batches = (total_batch_size + MAX_BATCH_SIZE - 1) / MAX_BATCH_SIZE;
    bool frame_has_defect = false;

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx) {
        int start = batch_idx * MAX_BATCH_SIZE;
        int end = std::min(start + MAX_BATCH_SIZE, total_batch_size);

        std::vector<ROIResult> sub_rois(rois.begin() + start, rois.begin() + end);
        auto sub_results = processBatchInternal(sub_rois, original_frame, frame_has_defect);
        results.insert(results.end(), sub_results.begin(), sub_results.end());
    }

    return results;
}

std::vector<ROIInferenceResult> ROIInference::processBatchInternal(const std::vector<ROIResult>& rois, const cv::Mat& original_frame, bool& frame_has_defect)
{
    m_current_frame = original_frame.clone();
    std::vector<ROIInferenceResult> results;

    if (rois.empty())
        return results;

    results.reserve(rois.size());

    benchmark_.start("08_batch_inference");

    int batch_size = static_cast<int>(rois.size());

    benchmark_.start("08batch_get_input_tensor");
    MNN::Tensor* input_tensor = interpreter_->getSessionInput(session_, nullptr);
    if (!input_tensor) {
        throw std::runtime_error("Failed to get input tensor");
    }
    benchmark_.stop("08batch_get_input_tensor");

    benchmark_.start("08batch_ensure_capacity");
    ensureBatchCapacity(batch_size);
    benchmark_.stop("08batch_ensure_capacity");

    benchmark_.start("08batch_prepare_all");
    float* batch_data = batch_host_input_->host<float>();
    if (!batch_data) {
        throw std::runtime_error("Failed to access batch input buffer");
    }

    for (int i = 0; i < batch_size; ++i) {
        float* roi_input = batch_data + i * (3 * INPUT_SIZE * INPUT_SIZE);
        prepareInput(rois[i].image, roi_input);
    }
    benchmark_.stop("08batch_prepare_all");

    benchmark_.start("08batch_resize_tensor");
    std::vector<int> batch_shape = {batch_size, 3, INPUT_SIZE, INPUT_SIZE};
    interpreter_->resizeTensor(input_tensor, batch_shape);
    benchmark_.stop("08batch_resize_tensor");

    benchmark_.start("08batch_resize_session");
    interpreter_->resizeSession(session_);
    benchmark_.stop("08batch_resize_session");

    benchmark_.start("08batch_get_input_after_resize");
    input_tensor = interpreter_->getSessionInput(session_, nullptr);
    if (!input_tensor) {
        throw std::runtime_error("Failed to get input tensor after resize");
    }
    benchmark_.stop("08batch_get_input_after_resize");

    benchmark_.start("08batch_copy_to_device");
    bool copy_success = input_tensor->copyFromHostTensor(batch_host_input_.get());
    if (!copy_success) {
        throw std::runtime_error("Failed to copy batch input to device");
    }
    benchmark_.stop("08batch_copy_to_device");

    if (Config::getInstance().isDebugMode()) {
        std::cout << "\n[DEBUG] Before runSession()" << std::endl;
    }

    benchmark_.start("08batch_mnn_inference");
    interpreter_->runSession(session_);
    benchmark_.stop("08batch_mnn_inference");

    if (Config::getInstance().isDebugMode()) {
        std::cout << "[DEBUG] Immediately after runSession()" << std::endl;
    }

    benchmark_.start("08batch_get_output_before");
    MNN::Tensor* output_tensor = interpreter_->getSessionOutput(session_, nullptr);
    if (!output_tensor) {
        throw std::runtime_error("Failed to get output tensor");
    }
    benchmark_.stop("08batch_get_output_before");

    if (Config::getInstance().isDebugMode()) {
        std::cout << "\n[DEBUG] Output tensor:" << std::endl;
        std::cout << "  Dimensions: " << output_tensor->dimensions() << std::endl;
        std::cout << "  Shape: ";
        for (int i = 0; i < output_tensor->dimensions(); ++i) {
            std::cout << output_tensor->length(i) << " ";
        }
        std::cout << std::endl;
        std::cout << "\n[DEBUG] Starting copyToHostTensor() (preallocated)" << std::endl;
        std::cout << "[DEBUG] BEFORE copyToHostTensor()" << std::endl;
    }

    benchmark_.start("08batch_copy_to_host");
    bool copy_out_success = output_tensor->copyToHostTensor(batch_host_output_.get());
    if (!copy_out_success) {
        throw std::runtime_error("Failed to copy output from device");
    }
    benchmark_.stop("08batch_copy_to_host");

    benchmark_.start("08batch_postprocess");
    const float* logits = batch_host_output_->host<float>();
    if (!logits) {
        throw std::runtime_error("Failed to access output logits");
    }


    for (int i = 0; i < batch_size; ++i) {
        const auto& roi = rois[i];
        const float* roi_logits = &logits[i * NUM_CLASSES];


        if (frame_has_defect) {
            continue;
        }

        float max_logit = roi_logits[0];
        for (int j = 1; j < NUM_CLASSES; ++j) {
            max_logit = std::max(max_logit, roi_logits[j]);
        }

        std::vector<float> probabilities(NUM_CLASSES);
        float sum = 0.0f;

        for (int j = 0; j < NUM_CLASSES; ++j) {
            probabilities[j] = std::exp(roi_logits[j] - max_logit);
            sum += probabilities[j];
        }

        for (float& p : probabilities) {
            p /= sum;
        }

        const int predicted_class = static_cast<int>(
            std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin()
        );

        const float confidence = probabilities[predicted_class];
        const float defect_probability = computeDefectProbability(probabilities);
	const bool predicted_is_non_defect =
	    (predicted_class == LINT_IDX) || (predicted_class == NORMAL_IDX);

	const bool defect = (defect_probability >= defect_threshold_) &&
	                    (!predicted_is_non_defect || confidence < 0.80f);

        ROIInferenceResult result = {
            roi.frame_id,
            roi.box,
            predicted_class,
            CLASS_NAMES[predicted_class],
            defect_probability,
            confidence,
            defect
        };

        results.push_back(result);

        if (defect) {
            result.flag_time_ms = std::chrono::duration<double, std::milli>(
                std::chrono::high_resolution_clock::now().time_since_epoch()
            ).count();

            std::cout << "[DEFECT DETECTED] Frame: " << roi.frame_id
                      << " | ROI: " << roi.box.number
                      << " | Batch idx: " << i
                      << " | Class: " << CLASS_NAMES[predicted_class]
                      << " | Confidence: " << std::fixed << std::setprecision(1) << (confidence * 100) << "%"
                      << " | DefectProb: " << std::fixed << std::setprecision(6) << defect_probability
                      << " (thr " << defect_threshold_ << ")" << std::endl;

            std::cout << "  Logits:";
            for (int j = 0; j < NUM_CLASSES; ++j) {
                std::cout << " " << std::fixed << std::setprecision(3) << roi_logits[j];
            }
            std::cout << "  [cuts hole lint normal oil]" << std::endl;

            std::cout << "  Probs :";
            for (int j = 0; j < NUM_CLASSES; ++j) {
                std::cout << " " << std::fixed << std::setprecision(6) << probabilities[j];
            }
            std::cout << std::endl;

            frame_has_defect = true;
            // Batched GPU path: enqueue to async worker if available, else write inline.
            DefectImages imgs = saveDefectFrame(roi, result, m_current_frame, "GPU");
            if (io_worker_) {
                if (!imgs.roi_display.empty())  io_worker_->enqueue_write(std::move(imgs.roi_display),  imgs.roi_path);
                if (!imgs.full_display.empty()) io_worker_->enqueue_write(std::move(imgs.full_display), imgs.full_path);
            } else {
                if (!imgs.roi_display.empty())  cv::imwrite(imgs.roi_path,  imgs.roi_display);
                if (!imgs.full_display.empty()) cv::imwrite(imgs.full_path, imgs.full_display);
            }
        }         
    }

    benchmark_.stop("08batch_postprocess");
    benchmark_.stop("08_batch_inference");

    if (Config::getInstance().isDebugMode()) {
        std::cout << "\n========================================" << std::endl;
        std::cout << "BATCH INFERENCE COMPLETE (Preallocated)" << std::endl;
        std::cout << "========================================" << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;
        std::cout << "Results: " << results.size() << " ROIs processed" << std::endl;
        std::cout << "========================================" << std::endl;
    }

    return results;
}

ROIInferenceResult ROIInference::inferCPU(const ROIResult& roi) {
    // TIMER_START_BEGIN
    auto t_pack_0 = std::chrono::high_resolution_clock::now();
    // TIMER_START_END

    // Safety check
    if (roi.image.empty() || roi.image.rows != 224 || roi.image.cols != 224) {
        throw std::runtime_error("Invalid ROI image for CPU inference");
    }
    
    // Prepare input for Edge Impulse - Packed RGB (1 float per pixel)
    std::vector<float> raw_features;
    raw_features.reserve(224 * 224);  // 50,176 features
    
    // roi.image is already RGB (from ROIResult)
    for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
            cv::Vec3b pixel = roi.image.at<cv::Vec3b>(y, x);
            // Pack RGB into a single float: (R << 16) | (G << 8) | B
            uint32_t rgb_packed = (static_cast<uint32_t>(pixel[0]) << 16) | 
                                  (static_cast<uint32_t>(pixel[1]) << 8) | 
                                  static_cast<uint32_t>(pixel[2]);
            raw_features.push_back(static_cast<float>(rgb_packed));
        }
    }

    // TIMER_PACK_END
    auto t_pack_1 = std::chrono::high_resolution_clock::now();
    // TIMER_PACK_END
    
    // Verify size matches expected
    if (raw_features.size() != EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE) {
        std::cerr << "CPU inference: raw_features size " << raw_features.size() 
                  << " != " << EI_CLASSIFIER_DSP_INPUT_FRAME_SIZE << std::endl;
        throw std::runtime_error("Invalid raw_features size for CPU inference");
    }
    
    // Run Edge Impulse classifier
    signal_t signal;
    numpy::signal_from_buffer(raw_features.data(), raw_features.size(), &signal);

    // TIMER_SETUP_END
    auto t_setup_1 = std::chrono::high_resolution_clock::now();
    // TIMER_SETUP_END
    
    ei_impulse_result_t ei_result;
    EI_IMPULSE_ERROR res = run_classifier(&signal, &ei_result, false);
    if (res != EI_IMPULSE_OK) {
        throw std::runtime_error("Edge Impulse inference failed");
    }

    // TIMER_CLASSIFY_END
    auto t_classify_1 = std::chrono::high_resolution_clock::now();
    // TIMER_CLASSIFY_END
    
    // Get raw logits from Edge Impulse
    std::vector<float> logits(NUM_CLASSES);
    for (int i = 0; i < NUM_CLASSES && i < EI_CLASSIFIER_LABEL_COUNT; i++) {
        logits[i] = ei_result.classification[i].value;
    }
    
    // Apply softmax to convert logits to probabilities
    float max_logit = logits[0];
    for (int i = 1; i < NUM_CLASSES; i++) {
        max_logit = std::max(max_logit, logits[i]);
    }
    
    std::vector<float> probabilities(NUM_CLASSES);
    float sum = 0.0f;
    for (int i = 0; i < NUM_CLASSES; i++) {
        probabilities[i] = std::exp(logits[i] - max_logit);
        sum += probabilities[i];
    }
    for (float& p : probabilities) {
        p /= sum;
    }
    
    const int predicted_class = static_cast<int>(
        std::max_element(probabilities.begin(), probabilities.end()) - probabilities.begin()
    );
    const float confidence = probabilities[predicted_class];

    const float defect_probability = computeDefectProbability(probabilities);
    const bool predicted_is_non_defect =
	(predicted_class == LINT_IDX) || (predicted_class == NORMAL_IDX);

    const bool defect = (defect_probability >= defect_threshold_) &&
	                (!predicted_is_non_defect || confidence < 0.80f);

    // ============================================================
    // DEBUG: dump logits and probabilities for this ROI //////////////////////////////////////////////////////////////////////////////////////////////////////
    // ============================================================

    if (Config::getInstance().isDebugMode()) {
        static int roi_dump_counter = 0;
        std::string dump_dir = "output/debug/roi_crops/";
        static bool dir_created = false;
        if (!dir_created) { system(("mkdir -p " + dump_dir).c_str()); dir_created = true; }

        std::stringstream ss;
        ss << dump_dir << "frame_" << std::setw(4) << std::setfill('0') << roi.frame_id
           << "_roi_" << roi.box.number << "_" << roi_dump_counter++ << ".png";
        cv::Mat bgr_dump;
        cv::cvtColor(roi.image, bgr_dump, cv::COLOR_RGB2BGR);
        cv::imwrite(ss.str(), bgr_dump);

        std::cout << "[inferCPU] Dumped ROI crop: " << ss.str()
                  << " (roi.box: x=" << roi.box.x << " y=" << roi.box.y
                  << " w=" << roi.box.width << " h=" << roi.box.height << ")"
                  << std::endl;
    }

    if (Config::getInstance().isDebugMode()) {
        std::cout << "[inferCPU] Frame " << roi.frame_id
                  << " ROI " << roi.box.number
                  << " | Logits:";
        for (int i = 0; i < NUM_CLASSES; ++i) {
            std::cout << " " << std::fixed << std::setprecision(3) << logits[i];
        }
        std::cout << "  [cuts hole lint normal oil]" << std::endl;

        std::cout << "[inferCPU] Frame " << roi.frame_id
                  << " ROI " << roi.box.number
                  << " | Probs :";
        for (int i = 0; i < NUM_CLASSES; ++i) {
            std::cout << " " << std::fixed << std::setprecision(6) << probabilities[i];
        }
        std::cout << std::endl;

        std::cout << "[inferCPU] Frame " << roi.frame_id
                  << " ROI " << roi.box.number
                  << " | predicted=" << CLASS_NAMES[predicted_class]
                  << " confidence=" << std::fixed << std::setprecision(4) << confidence
                  << " defect_prob=" << std::fixed << std::setprecision(6) << defect_probability
                  << " threshold=" << defect_threshold_
                  << " defect=" << (defect ? "true" : "false")
                  << std::endl;
    }

    // ============================================================
    // DEBUG: dump logits and probabilities for this ROI //////////////////////////////////////////////////////////////////////////////////////////////////////
    // ============================================================

    ROIInferenceResult result = {
        roi.frame_id,
        roi.box,
        predicted_class,
        CLASS_NAMES[predicted_class],
        defect_probability,
        probabilities[predicted_class],
        defect
    };
    
    return result;
}

void ROIInference::warmup() {
    std::cout << "[ROIInference] Warming up CPU inference..." << std::endl;

    // Build a dummy 224x224 RGB image (gray)
    cv::Mat dummy(224, 224, CV_8UC3, cv::Scalar(128, 128, 128));

    ROIResult dummy_roi;
    dummy_roi.frame_id = -1;
    dummy_roi.image = dummy;

    auto t0 = std::chrono::high_resolution_clock::now();

    try {
        inferCPU(dummy_roi);
    } catch (const std::exception& e) {
        std::cerr << "[ROIInference] Warmup failed: " << e.what() << std::endl;
        return;
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    std::cout << "[ROIInference] Warmup complete: " << ms << " ms" << std::endl;
}

} // namespace minimind
