#pragma once

#include <queue>
#include <mutex>
#include <condition_variable>
#include <vector>
#include <atomic>
#include "roi_processor.hpp"

namespace minimind {

struct QueuedROI {
    // A GPU batch may contain ROIs from more than one frame.
    ROIResult roi;
    int frame_id;
};

class GPUQueue {
public:
    // The default batch size matches the MNN inference path's tuned maximum;
    // partial batches are released at frame boundaries.
    GPUQueue(size_t batch_size = 16) 
        : batch_size_(batch_size), 
          current_batch_count_(0),
          new_frame_started_(false),
          should_stop_(false) {}

    // Queue an ROI; the consumer wakes when the batch fills or a frame ends.
    bool push(const ROIResult& roi) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        if (should_stop_) return false;
        
        pending_rois_.push_back({roi, roi.frame_id});
        current_batch_count_++;
        
        // Trigger inference only when batch is full
        if (current_batch_count_ >= batch_size_) {
            cv_.notify_one();
        }
        
        return true;
    }

    // Release a partial batch from the previous frame before processing the new one.
    void markFrameStarted(int frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        new_frame_started_ = true;
        
        // If there are pending ROIs from previous frames, trigger processing
        if (!pending_rois_.empty()) {
            cv_.notify_one();
        }
    }

    // Pop the next batch, including a partial batch at a frame boundary.
    std::vector<ROIResult> popBatch() {
        std::unique_lock<std::mutex> lock(mutex_);
        
        cv_.wait(lock, [this] {
            // Trigger when: batch full OR (we have ROIs AND a new frame started)
            return (current_batch_count_ >= batch_size_) ||
                   (!pending_rois_.empty() && new_frame_started_) ||
                   should_stop_;
        });
        
        if (should_stop_ && pending_rois_.empty()) {
            return {};
        }
        
        // Determine batch size
        size_t batch_size = std::min(current_batch_count_, batch_size_);
        
        std::vector<ROIResult> batch;
        batch.reserve(batch_size);
        
        // Pop from front
        for (size_t i = 0; i < batch_size && !pending_rois_.empty(); i++) {
            batch.push_back(pending_rois_.front().roi);
            pending_rois_.erase(pending_rois_.begin());
        }
        
        current_batch_count_ -= batch.size();
        new_frame_started_ = false;
        
        return batch;
    }

    // Force any pending ROIs to become eligible for popBatch, without
    // waiting for a new frame to start. Used during stop-the-world operations
    // (e.g. recalibration) to drain in-flight ROIs quickly.
    void drain() {
        std::lock_guard<std::mutex> lock(mutex_);
        new_frame_started_ = true;
        cv_.notify_one();
    }

    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        should_stop_ = true;
        cv_.notify_one();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pending_rois_.size();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    
    std::vector<QueuedROI> pending_rois_;
    
    const size_t batch_size_;
    size_t current_batch_count_;
    bool new_frame_started_;
    bool should_stop_;
};

} // namespace minimind
