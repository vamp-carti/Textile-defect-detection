#ifndef DATA_SENDER_HPP
#define DATA_SENDER_HPP

#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <cstring>
#include <iostream>
#include <vector>

// Forward declare socket headers
struct sockaddr_in;

namespace minimind {

struct UIData {
    // === Status ===
    std::string status = "WAITING";        // WAITING, RUNNING, PAUSED, COMPLETE, STOPPED
    std::string gpu_status = "ONLINE";     // ONLINE, OFFLINE
    
    // === Pipeline stats ===
    size_t frames_processed = 0;
    size_t queue_size = 0;
    size_t active_count = 0;
    size_t total_components = 0;
    size_t total_rois = 0;
    size_t total_inferences = 0;
    float fps = 0.0f;
    float avg_time_ms = 0.0f;
    
    // === System stats ===
    float cpu_usage = 0.0f;
    float cpu_temp = 0.0f;
    float gpu_temp = 0.0f;
    float memory_usage_mb = 0.0f;
    
    // === ROI overshoot alert ===
    bool roi_overshoot = false;
    int roi_overshoot_frame_id = 0;
    
    // === Defect-triggered frame (only for defect frames) ===
    bool has_defect_frame = false;
    int defect_frame_id = 0;
    std::string defect_image_base64;       // JPEG image with ROI boxes
    int defect_image_width = 0;
    int defect_image_height = 0;

    // === Calibration state ===
    std::string calibration_state = "idle";       // idle | running | done | failed
    float       calibration_progress = 0.0f;      // 0..100
    std::string calibration_source_path;          // path to source frame image
    
    // === Single defect (for backward compatibility / quick display) ===
    bool has_defect = false;
    struct DefectInfo {
        int frame_id = 0;
        int roi_id = 0;
        std::string predicted_class;
        float confidence = 0.0f;
        float defect_probability = 0.0f;
    } last_defect;
    
    // === Recent defects (last 50) ===
    std::vector<DefectInfo> recent_defects;
    // Total defects seen since pipeline start (unbounded). Used by the UI
    // for the "Defects:" counter. recent_defects remains capped at 50 and
    // is only for the preview list.
    size_t total_defects = 0;
    
    // Helper: add a defect to the rolling list
void addDefect(int frame_id, int roi_id, const std::string& cls, 
               float conf, float prob) {
    DefectInfo d;
    d.frame_id = frame_id;
    d.roi_id = roi_id;
    d.predicted_class = cls;
    d.confidence = conf;
    d.defect_probability = prob;
    
    recent_defects.push_back(d);
    if (recent_defects.size() > 10) {
        recent_defects.erase(recent_defects.begin());
    }
    total_defects++;
    }
};

class DataSender {
public:
    DataSender(int port = 9999);
    ~DataSender();
    
    void start();
    void stop();
    void updateData(const UIData& data);
    
private:
    void serverLoop();
    void handleClient(int client_fd);
    void sendData(int client_fd);
    
    int m_port;
    bool m_running;
    std::thread m_server_thread;
    UIData m_current_data;
    std::mutex m_data_mutex;
};

} // namespace minimind

#endif // DATA_SENDER_HPP
