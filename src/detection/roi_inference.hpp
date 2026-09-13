#pragma once

#include <string>
#include <vector>
#include <memory>

#include "roi_processor.hpp"
#include "benchmark.hpp"
#include "core/config.hpp"

#include <MNN/Interpreter.hpp>

namespace minimind {

struct DefectImages {
    cv::Mat     roi_display;     // owns data; ready for imwrite
    std::string roi_path;
    cv::Mat     full_display;    // owns data; empty if original_frame was empty
    std::string full_path;
};

struct ROIInferenceResult {
    int frame_id;
    ROIBox box;

    int predicted_class;
    std::string predicted_label;

    float defect_probability;
    float confidence;

    bool defect;
    double flag_time_ms = 0.0;
};

class ROIInference {
public:
    ROIInference(
        const std::string& model_path,
        const std::string& opencl_backend_path,
        float defect_threshold = 0.14f
    );

    // Run one dummy inference to warm up XNNPACK and other lazy-initialized paths
    void warmup();
    void setIoWorker(class IoWorker* w) { io_worker_ = w; } 

    BenchmarkTimer& getBenchmarkTimer();

    ~ROIInference();

    // Processes ROIs in their existing order.
    // Once a defect is detected, remaining ROIs from that frame
    // are not inferred.
    std::vector<ROIInferenceResult> process(
        const std::vector<ROIResult>& rois
    );

    // Batch process all ROIs - copies all inputs first, runs all inferences,
    // then copies all outputs back in one batch for better performance.
    std::vector<ROIInferenceResult> processBatch(
        const std::vector<ROIResult>& rois, const cv::Mat& original_frame
    );
    
DefectImages saveDefectFrame(const ROIResult& roi, const ROIInferenceResult& result,
                             const cv::Mat& original_frame, const std::string& source);

    // CPU inference for single ROI (Edge Impulse)
    ROIInferenceResult inferCPU(const ROIResult& roi);

private:
    static constexpr int INPUT_SIZE = 224;
    static constexpr int NUM_CLASSES = 5;

    static constexpr int CUTS_IDX = 0;
    static constexpr int HOLE_IDX = 1;
    static constexpr int LINT_IDX = 2;
    static constexpr int NORMAL_IDX = 3;
    static constexpr int OIL_IDX = 4;

    // Tuned GPU batch ceiling; larger batches increase memory without reducing
    // the measured synchronization-dominated latency proportionally.
    static constexpr int MAX_BATCH_SIZE = 16;
    std::vector<ROIInferenceResult> processBatchInternal(const std::vector<ROIResult>& rois, const cv::Mat& original_frame, bool& frame_has_defect);
    static constexpr const char* CLASS_NAMES[NUM_CLASSES] = {
        "cuts",
        "hole",
        "lint",
        "normal",
        "oil"
    };

    class IoWorker* io_worker_ = nullptr;

    // Edge Impulse model data
    const unsigned char* ei_model_data_;
    unsigned int ei_model_data_len_;
    
    // TFLite Micro tensor arena
    std::unique_ptr<uint8_t[]> tensor_arena_;
    size_t tensor_arena_size_;
    
    // Edge Impulse initialization
    bool initializeEdgeImpulse();

 
    cv::Mat m_current_frame;

    void loadOpenCLBackend(const std::string& backend_path);

    void prepareInput(const cv::Mat& rgb, float* input);

    ROIInferenceResult infer(const ROIResult& roi);

    float computeDefectProbability(const std::vector<float>& probabilities) const;

    void ensureBatchCapacity(int batch_size);

    BenchmarkTimer benchmark_;

    void* opencl_handle_ = nullptr;

    MNN::Interpreter* interpreter_ = nullptr;
    MNN::Session* session_ = nullptr;
    MNN::Tensor* input_tensor_ = nullptr;

    float defect_threshold_;

    // Preallocated batch memory
    int current_max_batch_ = 0;
    std::vector<float> batch_input_buffer_;
    std::vector<float> batch_output_buffer_;
    std::shared_ptr<MNN::Tensor> batch_host_input_;
    std::shared_ptr<MNN::Tensor> batch_host_output_;
};

} // namespace minimind
