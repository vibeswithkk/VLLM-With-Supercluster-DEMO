#pragma once

#include <nccl.h>
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <atomic>
#include <stdexcept>
#include <iostream>
#include <string>
#include <thread>
#include <chrono>

// Error checking macros for NCCL operations
#define NCCL_CHECK(call) do { \
    ncclResult_t result = call; \
    if (result != ncclSuccess) { \
        throw std::runtime_error(std::string("NCCL error: ") + ncclGetErrorString(result)); \
    } \
} while(0)

#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(error)); \
    } \
} while(0)

/**
 * @brief Enterprise-grade NCCL environment manager for distributed computing
 * 
 * This class provides a comprehensive interface for managing NCCL communication
 * environments in distributed GPU computing scenarios. It handles initialization,
 * resource management, error handling, and security considerations.
 */
class NCCLEnvironment {
private:
    // Singleton instance
    static std::unique_ptr<NCCLEnvironment> instance_;
    static std::mutex instance_mutex_;
    
    // NCCL communicator and related resources
    ncclComm_t communicator_;
    std::vector<int> device_ids_;
    int world_size_;
    int rank_;
    bool initialized_;
    
    // Thread safety
    mutable std::mutex env_mutex_;
    
    // Performance monitoring
    std::atomic<uint64_t> bytes_transferred_;
    std::atomic<uint64_t> operations_count_;
    
    // Resource tracking
    std::atomic<size_t> memory_allocated_;
    
    // Security and validation
    std::string session_id_;
    std::atomic<bool> secure_mode_;
    
    /**
     * @brief Private constructor for singleton pattern
     */
    NCCLEnvironment();
    
    /**
     * @brief Generate secure session identifier
     * @return Secure session ID string
     */
    std::string generate_session_id();
    
    /**
     * @brief Validate device configuration
     */
    void validate_device_configuration();
    
    /**
     * @brief Initialize NCCL communicator
     */
    void initialize_communicator();
    
public:
    /**
     * @brief Get singleton instance
     * @return Reference to NCCLEnvironment instance
     */
    static NCCLEnvironment& getInstance();
    
    /**
     * @brief Destructor
     */
    ~NCCLEnvironment();
    
    /**
     * @brief Initialize NCCL environment with specified parameters
     * @param device_ids Vector of GPU device IDs to use
     * @param world_size Number of processes in the communicator
     * @param rank Rank of current process
     * @param secure_mode Enable security features
     */
    void initialize(const std::vector<int>& device_ids, 
                   int world_size, 
                   int rank, 
                   bool secure_mode = true);
    
    /**
     * @brief Check if environment is initialized
     * @return True if initialized
     */
    bool isInitialized() const;
    
    /**
     * @brief Get NCCL communicator
     * @return NCCL communicator handle
     */
    ncclComm_t getCommunicator() const;
    
    /**
     * @brief Get world size
     * @return Number of processes in communicator
     */
    int getWorldSize() const;
    
    /**
     * @brief Get process rank
     * @return Rank of current process
     */
    int getRank() const;
    
    /**
     * @brief Get device IDs
     * @return Vector of device IDs
     */
    const std::vector<int>& getDeviceIds() const;
    
    /**
     * @brief Synchronize all GPUs in communicator
     * @param stream CUDA stream to use (default: 0)
     */
    void synchronize(cudaStream_t stream = 0);
    
    /**
     * @brief Get performance statistics
     * @param bytes_transferred Reference to store bytes transferred
     * @param operations_count Reference to store operations count
     */
    void getPerformanceStats(uint64_t& bytes_transferred, uint64_t& operations_count) const;
    
    /**
     * @brief Reset performance statistics
     */
    void resetPerformanceStats();
    
    /**
     * @brief Get memory allocation statistics
     * @return Total memory allocated in bytes
     */
    size_t getMemoryUsage() const;
    
    /**
     * @brief Get session ID
     * @return Session identifier string
     */
    const std::string& getSessionId() const;
    
    /**
     * @brief Check if secure mode is enabled
     * @return True if secure mode is enabled
     */
    bool isSecureMode() const;
    
    /**
     * @brief Set error handling mode
     * @param enable_async Enable asynchronous error handling
     */
    void setErrorHandlingMode(bool enable_async);
    
    /**
     * @brief Cleanup and shutdown NCCL environment
     */
    void shutdown();
};

// Implementation of inline methods

inline bool NCCLEnvironment::isInitialized() const {
    std::lock_guard<std::mutex> lock(env_mutex_);
    return initialized_;
}

inline ncclComm_t NCCLEnvironment::getCommunicator() const {
    std::lock_guard<std::mutex> lock(env_mutex_);
    return communicator_;
}

inline int NCCLEnvironment::getWorldSize() const {
    std::lock_guard<std::mutex> lock(env_mutex_);
    return world_size_;
}

inline int NCCLEnvironment::getRank() const {
    std::lock_guard<std::mutex> lock(env_mutex_);
    return rank_;
}

inline const std::vector<int>& NCCLEnvironment::getDeviceIds() const {
    std::lock_guard<std::mutex> lock(env_mutex_);
    return device_ids_;
}

inline void NCCLEnvironment::getPerformanceStats(uint64_t& bytes_transferred, uint64_t& operations_count) const {
    bytes_transferred = bytes_transferred_.load();
    operations_count = operations_count_.load();
}

inline void NCCLEnvironment::resetPerformanceStats() {
    bytes_transferred_.store(0);
    operations_count_.store(0);
}

inline size_t NCCLEnvironment::getMemoryUsage() const {
    return memory_allocated_.load();
}

inline const std::string& NCCLEnvironment::getSessionId() const {
    return session_id_;
}

inline bool NCCLEnvironment::isSecureMode() const {
    return secure_mode_.load();
}