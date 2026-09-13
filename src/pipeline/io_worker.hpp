#pragma once

#include <condition_variable>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <atomic>

#include <opencv2/core.hpp>

namespace minimind {

class IoWorker {
public:
    IoWorker();
    ~IoWorker();

    IoWorker(const IoWorker&) = delete;
    IoWorker& operator=(const IoWorker&) = delete;

    // -------- Producer side --------

    // Enqueue a decode request for `path`. Blocks if the decode queue is at capacity.
    // No-op if the frame is already decoded (in registry_) or currently queued.
    void request_decode(const std::string& path);

    // Block until the decoded frame for `path` is available; return it.
    // The returned cv::Mat owns its pixel data. Consumed exactly once.
    cv::Mat wait_for_frame(const std::string& path);

    // -------- Consumer side --------

    // Enqueue a write task. Blocks if the write queue is at capacity.
    // `img` must own its data (e.g. produced by cvtColor / clone / imread).
    // Returns immediately once the task is queued — does not wait for disk.
    void enqueue_write(cv::Mat img, const std::string& path);

    // -------- Lifecycle --------

    // Signal the worker to stop accepting new tasks and exit once queues are
    // drained. Blocks until the worker thread has joined. Safe to call once.
    void flush();

private:
    struct WriteTask {
        cv::Mat     img;
        std::string path;
    };

    void run();

    std::thread              worker_;
    std::mutex               m_;
    std::condition_variable  cv_task_;    // worker wakes when a task is queued
    std::condition_variable  cv_frame_;   // producer wakes when a frame is ready
    std::condition_variable  cv_slot_;    // enqueuers wake when a slot frees

    std::deque<std::string>                       decode_q_;
    std::deque<WriteTask>                         write_q_;
    std::unordered_map<std::string, cv::Mat>      registry_;   // decoded-but-unconsumed
    std::unordered_map<std::string, bool>         queued_;     // decode-in-flight dedupe

    static constexpr size_t kDecodeCap = 2;
    static constexpr size_t kWriteCap  = 4;

    std::atomic<bool> stop_{false};
    std::atomic<bool> joined_{false};
};

} // namespace minimind
