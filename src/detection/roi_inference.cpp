#include <iostream>
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

struct DefectReport {
    int frame_id;
    int roi_id;
    std::string predicted_class;
    float confidence;
    float defect_probability;
    std::chrono::system_clock::time_point timestamp;
}; 

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

void ROIInference::saveDefectFrame(const ROIResult& roi, const ROIInferenceResult& result, const cv::Mat& original_frame)
{
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
    
    // Push to global vector for CSV export
    {
        std::lock_guard<std::mutex> lock(g_defect_mutex);
        g_defect_reports.push_back(report);

    }
    
    // ============================================================
    // 2. CHECK IF SAVING IS ENABLED
    // ============================================================
    if (!Config::getInstance().production.save_defects) {
        std::cout << "[DEBUG] save_defects is FALSE, returning early" << std::endl;
        return;
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
    // FORMAT FRAME NUMBER (ADD THIS)
    // ============================================================
    std::stringstream ss_frame;
    ss_frame << std::setw(6) << std::setfill('0') << roi.frame_id;
    std::string frame_str = ss_frame.str();

    // ============================================================
    // 4. SAVE ROI CROP (only for defects)
    // ============================================================
    std::stringstream ss_roi;
    ss_roi << rois_dir << "frame_" << frame_str 
           << "_roi_" << roi.box.number << ".jpg";
    
    cv::Mat roi_display;
    cv::cvtColor(roi.image, roi_display, cv::COLOR_RGB2BGR);
    
    // Add label on ROI image
    std::string label = result.predicted_label + " " + 
                        std::to_string((int)(result.confidence * 100)) + "%";
    cv::putText(roi_display, label, 
                cv::Point(10, 30), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                cv::Scalar(0, 0, 255), 2);
    
    cv::imwrite(ss_roi.str(), roi_display);
    
    // ============================================================
    // 5. SAVE FULL FRAME WITH ROI BOX OVERLAY
    // ============================================================
    if (!original_frame.empty()) {
        std::stringstream ss_full;
        ss_full << frames_dir << "frame_" << frame_str << "_full.jpg";
        
        cv::Mat full_display;
        cv::cvtColor(original_frame, full_display, cv::COLOR_RGB2BGR);
        
        // Draw ROI box on full frame
        cv::rectangle(full_display, 
                      cv::Rect(roi.box.x, roi.box.y, roi.box.width, roi.box.height),
                      cv::Scalar(0, 0, 255), 3);
        
        // Draw label
        cv::putText(full_display, label, 
                    cv::Point(roi.box.x, roi.box.y - 10),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                    cv::Scalar(0, 0, 255), 2);
        
        
        cv::imwrite(ss_full.str(), full_display);
        std::cout << "[FRAME SAVED] " << ss_full.str() << std::endl;
    } else {
        std::cout << "[DEBUG] original_frame is EMPTY, skipping full frame save" << std::endl;
}
    
    // ============================================================
    // 6. LOG OUTPUT
    // ============================================================
    std::cout << "[DEFECT SAVED] Frame: " << roi.frame_id 
              << " | ROI: " << roi.box.number
              << " | Class: " << result.predicted_label
              << " | Confidence: " << std::fixed << std::setprecision(1) << (result.confidence * 100) << "%"
              << " | Saved to: " << rois_dir << std::endl;
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
    const bool defect = defect_probability >= defect_threshold_;

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

    // Save if defect detected
    if (defect) {
        saveDefectFrame(roi, result, m_current_frame);  // Need to store current frame
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

    if (Config::getInstance().isDebugMode()) {
        std::cout << "\n[DEBUG] BEFORE any operations:" << std::endl;
        std::cout << "  Input tensor dimensions: " << input_tensor->dimensions() << std::endl;
        std::cout << "  Input tensor shape: ";
        for (int i = 0; i < input_tensor->dimensions(); ++i) {
            std::cout << input_tensor->length(i) << " ";
        }
        std::cout << std::endl;
    }

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

    if (Config::getInstance().isDebugMode()) {
        std::cout << "\n[DEBUG] AFTER resizeTensor:" << std::endl;
        std::cout << "  Input tensor dimensions: " << input_tensor->dimensions() << std::endl;
        std::cout << "  Input tensor shape: ";
        for (int i = 0; i < input_tensor->dimensions(); ++i) {
            std::cout << input_tensor->length(i) << " ";
        }
        std::cout << std::endl;
    }

    benchmark_.start("08batch_resize_session");
    interpreter_->resizeSession(session_);
    benchmark_.stop("08batch_resize_session");

    if (Config::getInstance().isDebugMode()) {
        std::cout << "\n[DEBUG] AFTER resizeSession:" << std::endl;
        std::cout << "  Input tensor dimensions: " << input_tensor->dimensions() << std::endl;
        std::cout << "  Input tensor shape: ";
        for (int i = 0; i < input_tensor->dimensions(); ++i) {
            std::cout << input_tensor->length(i) << " ";
        }
        std::cout << std::endl;
    }

    benchmark_.start("08batch_get_input_after_resize");
    input_tensor = interpreter_->getSessionInput(session_, nullptr);
    if (!input_tensor) {
        throw std::runtime_error("Failed to get input tensor after resize");
    }
    benchmark_.stop("08batch_get_input_after_resize");

    if (Config::getInstance().isDebugMode()) {
        std::cout << "\n[DEBUG] FINAL input tensor after resize:" << std::endl;
        std::cout << "  Input tensor dimensions: " << input_tensor->dimensions() << std::endl;
        std::cout << "  Input tensor shape: ";
        for (int i = 0; i < input_tensor->dimensions(); ++i) {
            std::cout << input_tensor->length(i) << " ";
        }
        std::cout << std::endl;
    }

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

    if (Config::getInstance().isDebugMode()) {
        std::cout << "[DEBUG] AFTER copyToHostTensor()" << std::endl;
        std::cout << "  Host output dimensions: " << batch_host_output_->dimensions() << std::endl;
        std::cout << "  Host output shape: ";
        for (int i = 0; i < batch_host_output_->dimensions(); ++i) {
            std::cout << batch_host_output_->length(i) << " ";
        }
        std::cout << std::endl;
    }

    benchmark_.start("08batch_postprocess");
    const float* logits = batch_host_output_->host<float>();
    if (!logits) {
        throw std::runtime_error("Failed to access output logits");
    }

    if (Config::getInstance().isDebugMode()) {
        std::cout << "\n[DEBUG] First 5 logits: ";
        for (int i = 0; i < 5 && i < batch_size * NUM_CLASSES; ++i) {
            std::cout << logits[i] << " ";
        }
        std::cout << std::endl;
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
        const bool defect = defect_probability >= defect_threshold_;

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
            std::cout << "\033[32;0H" << std::flush;
            std::cout << "[DEFECT DETECTED] Frame: " << roi.frame_id 
                      << " | ROI: " << roi.box.number
                      << " | Class: " << CLASS_NAMES[predicted_class]
                      << " | Confidence: " << std::fixed << std::setprecision(1) << (confidence * 100) << "%" << std::endl;
            frame_has_defect = true;
            saveDefectFrame(roi, result, m_current_frame);
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

} // namespace minimind
