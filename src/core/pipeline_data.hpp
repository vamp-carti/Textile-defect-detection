#include <iostream>
#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <memory>
#include <vector>

#include "level1_detector.hpp"
#include "roi_processor.hpp"
#include "roi_inference.hpp"

namespace minimind {

// =====================================================================
// FRAME DATA CONTAINER
// =====================================================================

struct FrameData {
    int frame_id = 0;
    std::string filename;
    
    // Input images (released after processing to save memory)
    cv::Mat rgb_image;        // For ROI extraction (3-channel RGB)
    cv::Mat grayscale_image;  // For Level1 (1-channel gray)
    
    // Producer outputs (Level1 + ROIProcessor)
    std::vector<DefectComponent> components;
    std::vector<ROIResult> rois;
    
    // Consumer output (ROIInference)
    std::vector<ROIInferenceResult> inference_results;
    
    // Pipeline state
    bool level1_done = false;
    bool roi_done = false;
    bool inference_done = false;
    bool has_defect = false;
    bool processed = false;
    bool rgb_released = false;
 
    // Timing (for debugging)
    double level1_time_ms = 0.0;
    double roi_time_ms = 0.0;
    double inference_time_ms = 0.0;
    
    // Release RGB memory early (called after ROI generation)
    void releaseRGB() {
        rgb_image.release();
    }
    
    // Release grayscale memory early (called after Level1)
    void releaseGrayscale() {
        grayscale_image.release();
    }
};

// =====================================================================
// THREAD-SAFE PIPELINE QUEUE
// =====================================================================

class PipelineQueue {
public:
    explicit PipelineQueue(size_t max_size = 3)
        : max_size_(max_size)
        , stop_(false)
        , active_count_(0)
    {}
    
    // Push frame data (producer side)
    bool push(std::shared_ptr<FrameData> data) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        cv_.wait(lock, [this]() {
            return queue_.size() < max_size_ || stop_;
        });
        
        if (stop_) {
            return false;
        }
        
        queue_.push(data);
        active_count_++;
        lock.unlock();
        cv_.notify_one();
        return true;
    }
    
    // Pop frame data (consumer side)
    std::shared_ptr<FrameData> pop() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        cv_.wait(lock, [this]() {
            return !queue_.empty() || stop_;
        });
        
        if (stop_ && queue_.empty()) {
            return nullptr;
        }
        
        auto data = queue_.front();
        queue_.pop();
        lock.unlock();
        cv_.notify_one();
        return data;
    }
    
    bool shouldStop() const {
        return stop_;
    }
    
    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stop_ = true;
        cv_.notify_all();
    }
    
    void waitForCompletion() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this]() {
            return queue_.empty() && active_count_ == 0;
        });
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    size_t activeCount() const {
        return active_count_;
    }
    
    void markCompleted() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_count_ > 0) {
            active_count_--;
        }
        cv_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<std::shared_ptr<FrameData>> queue_;
    size_t max_size_;
    bool stop_;
    size_t active_count_;
};

// =====================================================================
// PIPELINE STATISTICS (for monitoring)
// =====================================================================

struct PipelineStats {
    std::atomic<size_t> frames_processed{0};
    std::atomic<size_t> frames_with_defects{0};
    std::atomic<size_t> total_components{0};
    std::atomic<size_t> total_rois{0};
    std::atomic<size_t> total_inferences{0};
    
    std::atomic<float> total_level1_time{0.0};
    std::atomic<float> total_roi_time{0.0};
    std::atomic<float> total_inference_time{0.0};
    
    void reset() {
        frames_processed = 0;
        frames_with_defects = 0;
        total_components = 0;
        total_rois = 0;
        total_inferences = 0;
        total_level1_time = 0.0;
        total_roi_time = 0.0;
        total_inference_time = 0.0;
    }
    
    void printSummary() const {
        size_t processed = frames_processed.load();
        if (processed == 0) {
            std::cout << "No frames processed.\n";
            return;
        }
        
        std::cout << "\n============================================================\n";
        std::cout << "PIPELINE SUMMARY\n";
        std::cout << "============================================================\n";
        std::cout << "Frames processed:        " << processed << "\n";
        std::cout << "Frames with defects:     " << frames_with_defects.load() << "\n";
        std::cout << "Total components:        " << total_components.load() << "\n";
        std::cout << "Total ROIs:              " << total_rois.load() << "\n";
        std::cout << "Total inferences:        " << total_inferences.load() << "\n";
        std::cout << "------------------------------------------------------------\n";
        std::cout << "Avg Level1 time:         " << (total_level1_time.load() / static_cast<float>(processed)) << " ms\n";
        std::cout << "Avg ROI time:            " << (total_roi_time.load() / static_cast<float>(processed)) << " ms\n";
        std::cout << "Avg Inference time:      " << (total_inference_time.load() / static_cast<float>(processed)) << " ms\n";
        std::cout << "============================================================\n";
    }
};

} // namespace minimind
