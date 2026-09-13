#include "pipeline/io_worker.hpp"

#include <iostream>
#include <opencv2/imgcodecs.hpp>

namespace minimind {

IoWorker::IoWorker() {
    worker_ = std::thread(&IoWorker::run, this);
}

IoWorker::~IoWorker() {
    // If the caller never called flush(), do it now. Idempotent.
    flush();
}

void IoWorker::request_decode(const std::string& path) {
    std::unique_lock<std::mutex> lk(m_);

    // Already decoded and waiting for the producer?
    if (registry_.count(path)) return;

    // Already queued?
    if (queued_.count(path)) return;

    // Block until a decode slot frees (or we're stopping).
    cv_slot_.wait(lk, [&] {
        return stop_.load() || decode_q_.size() < kDecodeCap;
    });
    if (stop_.load()) return;

    decode_q_.push_back(path);
    queued_[path] = true;
    cv_task_.notify_one();
}

cv::Mat IoWorker::wait_for_frame(const std::string& path) {
    std::unique_lock<std::mutex> lk(m_);

    cv_frame_.wait(lk, [&] {
        return stop_.load() || registry_.count(path);
    });

    if (stop_.load() && !registry_.count(path)) {
        return {};
    }

    cv::Mat out = std::move(registry_[path]);
    registry_.erase(path);
    // queued_ entry was already erased by the worker when it produced the mat.
    cv_slot_.notify_all();
    return out;
}

void IoWorker::enqueue_write(cv::Mat img, const std::string& path) {
    std::unique_lock<std::mutex> lk(m_);

    cv_slot_.wait(lk, [&] {
        return stop_.load() || write_q_.size() < kWriteCap;
    });
    if (stop_.load()) return;

    write_q_.push_back({std::move(img), path});
    cv_task_.notify_one();
}

void IoWorker::run() {
    // One worker serializes disk access while keeping decode ahead of writes;
    // this prevents defect-image output from delaying the next input frame.
    while (true) {
        std::unique_lock<std::mutex> lk(m_);

        cv_task_.wait(lk, [&] {
            return stop_.load() || !decode_q_.empty() || !write_q_.empty();
        });

        // Exit condition: stopping AND both queues drained.
        if (stop_.load() && decode_q_.empty() && write_q_.empty()) {
            return;
        }

        // Decode has priority because producer progress depends on the next frame.
        if (!decode_q_.empty()) {
            std::string path = decode_q_.front();
            decode_q_.pop_front();
            // Slot is free once popped; wake any producer blocked on request_decode.
            cv_slot_.notify_all();
            lk.unlock();

            cv::Mat m = cv::imread(path, cv::IMREAD_COLOR);

            lk.lock();
            queued_.erase(path);
            if (m.empty()) {
                // Decode failed. Wake the producer anyway so it doesn't block
                // forever waiting for a frame that will never arrive.
                // We insert an empty mat; wait_for_frame will return it.
                registry_[path] = cv::Mat();
            } else {
                registry_[path] = std::move(m);
            }
            cv_frame_.notify_all();
            continue;
        }

        // With no pending decode, drain one deferred defect-image write.
        WriteTask t = std::move(write_q_.front());
        write_q_.pop_front();
        // Slot is free once popped; wake any consumer blocked on enqueue_write.
        cv_slot_.notify_all();
        lk.unlock();

        try {
            cv::imwrite(t.path, t.img);
        } catch (const std::exception& e) {
            std::cerr << "[IoWorker] imwrite failed for " << t.path
                      << ": " << e.what() << std::endl;
        }
    }
}

void IoWorker::flush() {
    bool expected = false;
    if (!joined_.compare_exchange_strong(expected, true)) {
        // Someone else already flushed; nothing to do.
        return;
    }

    {
        std::lock_guard<std::mutex> lk(m_);
        stop_.store(true);
    }
    cv_task_.notify_all();
    cv_slot_.notify_all();
    cv_frame_.notify_all();

    if (worker_.joinable()) {
        worker_.join();
    }
}

} // namespace minimind
