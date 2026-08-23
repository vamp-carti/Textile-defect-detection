#include "data_sender.hpp"

#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <fcntl.h>

namespace minimind {

DataSender::DataSender(int port) 
    : m_port(port), m_running(false) {}

DataSender::~DataSender() {
    stop();
}

void DataSender::start() {
    if (m_running) return;
    m_running = true;
    m_server_thread = std::thread(&DataSender::serverLoop, this);
}

void DataSender::stop() {
    m_running = false;
    if (m_server_thread.joinable()) {
        m_server_thread.join();
    }
}

void DataSender::updateData(const UIData& data) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    m_current_data = data;
}

void DataSender::serverLoop() {
    int server_fd, client_fd;
    struct sockaddr_in address;
    int opt = 1;
    int addrlen = sizeof(address);
    
    // Create socket
    if ((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) {
        std::cerr << "[DataSender] Socket creation failed" << std::endl;
        return;
    }
    
    // Set socket option
    if (setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt))) {
        std::cerr << "[DataSender] Setsockopt failed" << std::endl;
        close(server_fd);
        return;
    }
    
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = INADDR_ANY;
    address.sin_port = htons(m_port);
    
    // Bind
    if (bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        std::cerr << "[DataSender] Bind failed on port " << m_port << std::endl;
        close(server_fd);
        return;
    }
    
    // Listen
    if (listen(server_fd, 3) < 0) {
        std::cerr << "[DataSender] Listen failed" << std::endl;
        close(server_fd);
        return;
    }
    
    std::cout << "[DataSender] Listening on port " << m_port << std::endl;
    
    while (m_running) {
        // Accept connection with timeout
        fd_set readfds;
        FD_ZERO(&readfds);
        FD_SET(server_fd, &readfds);
        
        struct timeval tv;
        tv.tv_sec = 1;
        tv.tv_usec = 0;
        
        int activity = select(server_fd + 1, &readfds, NULL, NULL, &tv);
        
        if (activity < 0) {
            if (m_running) {
                std::cerr << "[DataSender] Select error" << std::endl;
            }
            continue;
        }
        
        if (activity == 0) {
            // Timeout, check if we should continue
            continue;
        }
        
        if ((client_fd = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen)) < 0) {
            if (m_running) {
                std::cerr << "[DataSender] Accept failed" << std::endl;
            }
            continue;
        }
        
        // Handle client in separate thread
        std::thread client_thread(&DataSender::handleClient, this, client_fd);
        client_thread.detach();
    }
    
    close(server_fd);
    std::cout << "[DataSender] Server stopped" << std::endl;
}

void DataSender::handleClient(int client_fd) {
    std::cout << "[DataSender] Client connected" << std::endl;
    
    // Send initial data
    sendData(client_fd);
    
    char buffer[1024];
    while (m_running) {
        int bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read <= 0) {
            break;
        }
        buffer[bytes_read] = '\0';
        
        // Check for "GET_DATA" command
        std::string cmd(buffer);
        if (cmd.find("GET_DATA") != std::string::npos) {
            sendData(client_fd);
        }
    }
    
    close(client_fd);
    std::cout << "[DataSender] Client disconnected" << std::endl;
}

void DataSender::sendData(int client_fd) {
    std::lock_guard<std::mutex> lock(m_data_mutex);
    
    // Build JSON string
    std::string json = "{";
    json += "\"gpu_status\":\"" + m_current_data.gpu_status + "\",";
    json += "\"cpu_usage\":" + std::to_string(m_current_data.cpu_usage) + ",";
    json += "\"gpu_temp\":" + std::to_string(m_current_data.gpu_temp) + ",";
    json += "\"memory_usage_mb\":" + std::to_string(m_current_data.memory_usage_mb) + ",";
    json += "\"frames_processed\":" + std::to_string(m_current_data.frames_processed) + ",";
    json += "\"fps\":" + std::to_string(m_current_data.fps) + ",";
    json += "\"avg_time_ms\":" + std::to_string(m_current_data.avg_time_ms) + ",";
    json += "\"total_components\":" + std::to_string(m_current_data.total_components) + ",";
    json += "\"total_rois\":" + std::to_string(m_current_data.total_rois) + ",";
    json += "\"total_inferences\":" + std::to_string(m_current_data.total_inferences) + ",";
    json += "\"queue_size\":" + std::to_string(m_current_data.queue_size) + ",";
    json += "\"active_count\":" + std::to_string(m_current_data.active_count) + ",";
    json += "\"status\":\"" + m_current_data.status + "\",";
    json += "\"has_defect\":" + std::string(m_current_data.has_defect ? "true" : "false");
    
    if (m_current_data.has_defect) {
        json += ",\"last_defect\":{";
        json += "\"frame_id\":" + std::to_string(m_current_data.last_defect.frame_id) + ",";
        json += "\"roi_id\":" + std::to_string(m_current_data.last_defect.roi_id) + ",";
        json += "\"predicted_class\":\"" + m_current_data.last_defect.predicted_class + "\",";
        json += "\"confidence\":" + std::to_string(m_current_data.last_defect.confidence) + ",";
        json += "\"defect_probability\":" + std::to_string(m_current_data.last_defect.defect_probability);
        json += "}";
    }
    
    json += "}\n";
    
    send(client_fd, json.c_str(), json.length(), 0);
}

} // namespace minimind
