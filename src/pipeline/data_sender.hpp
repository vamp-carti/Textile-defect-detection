#ifndef DATA_SENDER_HPP
#define DATA_SENDER_HPP

#include <string>
#include <thread>
#include <mutex>
#include <queue>
#include <cstring>
#include <iostream>

// Forward declare socket headers
struct sockaddr_in;

namespace minimind {

struct UIData {
    std::string gpu_status = "ONLINE";
    float cpu_usage = 0.0f;
    float gpu_temp = 0.0f;
    float memory_usage_mb = 0.0f;
    size_t frames_processed = 0;
    float fps = 0.0f;
    float avg_time_ms = 0.0f;
    size_t total_components = 0;
    size_t total_rois = 0;
    size_t total_inferences = 0;
    size_t queue_size = 0;
    size_t active_count = 0;
    std::string status = "IDLE";
    bool has_defect = false;
    
    struct DefectInfo {
        int frame_id = 0;
        int roi_id = 0;
        std::string predicted_class;
        float confidence = 0.0f;
        float defect_probability = 0.0f;
    } last_defect;
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
