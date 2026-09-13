#pragma once

#include <opencv2/opencv.hpp>
#include <map>
#include <mutex>
#include <chrono>

namespace minimind {

class FrameBuffer {
public:
    FrameBuffer(int max_frames = 5) : max_frames_(max_frames) {}

    void store(int frame_id, const cv::Mat& frame, double producer_start_ms = 0.0) {
        std::lock_guard<std::mutex> lock(mutex_);

        buffer_[frame_id] = frame.clone();
        start_times_[frame_id] = producer_start_ms;

        // Keep delayed GPU results bounded by evicting the oldest frame first.
        while (buffer_.size() > max_frames_) {
            auto it = buffer_.begin();
            start_times_.erase(it->first);
            buffer_.erase(it);
        }
    }

    cv::Mat retrieve(int frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        // Returning a clone lets the consumer use the image after releasing
        // the buffer lock and makes eviction independent of the caller's data.
        auto it = buffer_.find(frame_id);
        if (it != buffer_.end()) {
            return it->second.clone();
        }
        return cv::Mat();
    }

    // Returns producer_start_ms for the given frame id, or 0.0 if not present.
    double getProducerStartMs(int frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = start_times_.find(frame_id);
        return (it != start_times_.end()) ? it->second : 0.0;
    }

    void remove(int frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        buffer_.erase(frame_id);
        start_times_.erase(frame_id);
    }

    bool contains(int frame_id) {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.find(frame_id) != buffer_.end();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return buffer_.size();
    }

private:
    mutable std::mutex mutex_;
    std::map<int, cv::Mat> buffer_;
    std::map<int, double>  start_times_;
    int max_frames_;
};

} // namespace minimind
