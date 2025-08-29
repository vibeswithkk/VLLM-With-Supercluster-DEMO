#include "nccl_env.hpp"
#include <random>
#include <sstream>
#include <iomanip>
#include <openssl/sha.h>
#include <cstring>

// Static member definitions
std::unique_ptr<NCCLEnvironment> NCCLEnvironment::instance_ = nullptr;
std::mutex NCCLEnvironment::instance_mutex_;

/**
 * @brief Private constructor for singleton pattern
 */
NCCLEnvironment::NCCLEnvironment() 
    : communicator_(nullptr), 
      world_size_(0), 
      rank_(-1), 
      initialized_(false),
      bytes_transferred_(0),
      operations_count_(0),
      memory_allocated_(0),
      secure_mode_(false) {
    // Generate unique session ID
    session_id_ = generate_session_id();
}

/**
 * @brief Generate secure session identifier
 * @return Secure session ID string
 */
std::string NCCLEnvironment::generate_session_id() {
    // Create a more secure session ID using timestamp and random values
    auto now = std::chrono::high_resolution_clock::now();
    auto duration = now.time_since_epoch();
    auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
    
    // Generate random component
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(100000, 999999);
    
    // Combine timestamp and random value
    std::stringstream ss;
    ss << "nccl_session_" << nanoseconds << "_" << dis(gen);
    
    // If in secure mode, add hash
    if (secure_mode_.load()) {
        // Simple hash for demonstration - in production, use proper cryptographic hash
        std::string data = ss.str();
        unsigned char hash[SHA256_DIGEST_LENGTH];
        SHA256_CTX sha256;
        SHA256_Init(&sha256);
        SHA256_Update(&sha256, data.c_str(), data.size());
        SHA256_Final(hash, &sha256);
        
        // Convert to hex string
        std::stringstream hash_ss;
        for(int i = 0; i < SHA256_DIGEST_LENGTH; i++) {
            hash_ss << std::hex << std::setw(2) << std::setfill('0') << (int)hash[i];
        }
        
        return hash_ss.str().substr(0, 32); // Return first 32 characters
    }
    
    return ss.str();
}

/**
 * @brief Validate device configuration
 */
void NCCLEnvironment::validate_device_configuration() {
    if (device_ids_.empty()) {
        throw std::invalid_argument("Device IDs vector cannot be empty");
    }
    
    if (world_size_ <= 0) {
        throw std::invalid_argument("World size must be positive");
    }
    
    if (rank_ < 0 || rank_ >= world_size_) {
        throw std::invalid_argument("Rank must be between 0 and world_size-1");
    }
    
    // Validate device IDs
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    
    for (const auto& device_id : device_ids_) {
        if (device_id < 0 || device_id >= device_count) {
            throw std::invalid_argument("Invalid device ID: " + std::to_string(device_id));
        }
    }
}

/**
 * @brief Initialize NCCL communicator
 */
void NCCLEnvironment::initialize_communicator() {
    // Set the first device as the current device
    if (!device_ids_.empty()) {
        CUDA_CHECK(cudaSetDevice(device_ids_[0]));
    }
    
    // Create NCCL communicator
    NCCL_CHECK(ncclCommInitRank(&communicator_, world_size_, ncclUniqueId(), rank_));
    
    // Update memory allocation tracking
    memory_allocated_.fetch_add(sizeof(ncclComm_t));
}

/**
 * @brief Get singleton instance
 * @return Reference to NCCLEnvironment instance
 */
NCCLEnvironment& NCCLEnvironment::getInstance() {
    std::lock_guard<std::mutex> lock(instance_mutex_);
    if (!instance_) {
        instance_ = std::unique_ptr<NCCLEnvironment>(new NCCLEnvironment());
    }
    return *instance_;
}

/**
 * @brief Destructor
 */
NCCLEnvironment::~NCCLEnvironment() {
    shutdown();
}

/**
 * @brief Initialize NCCL environment with specified parameters
 * @param device_ids Vector of GPU device IDs to use
 * @param world_size Number of processes in the communicator
 * @param rank Rank of current process
 * @param secure_mode Enable security features
 */
void NCCLEnvironment::initialize(const std::vector<int>& device_ids, 
                                int world_size, 
                                int rank, 
                                bool secure_mode) {
    std::lock_guard<std::mutex> lock(env_mutex_);
    
    if (initialized_) {
        throw std::runtime_error("NCCL environment already initialized");
    }
    
    // Store parameters
    device_ids_ = device_ids;
    world_size_ = world_size;
    rank_ = rank;
    secure_mode_.store(secure_mode);
    
    // Validate configuration
    validate_device_configuration();
    
    // Initialize communicator
    initialize_communicator();
    
    // Mark as initialized
    initialized_ = true;
    
    // Log initialization
    if (secure_mode) {
        std::cout << "NCCL Environment initialized in secure mode. Session ID: " << session_id_ << std::endl;
    } else {
        std::cout << "NCCL Environment initialized. Session ID: " << session_id_ << std::endl;
    }
}

/**
 * @brief Synchronize all GPUs in communicator
 * @param stream CUDA stream to use (default: 0)
 */
void NCCLEnvironment::synchronize(cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(env_mutex_);
    
    if (!initialized_) {
        throw std::runtime_error("NCCL environment not initialized");
    }
    
    NCCL_CHECK(ncclCommSynchronize(communicator_, stream));
    operations_count_.fetch_add(1);
}

/**
 * @brief Set error handling mode
 * @param enable_async Enable asynchronous error handling
 */
void NCCLEnvironment::setErrorHandlingMode(bool enable_async) {
    std::lock_guard<std::mutex> lock(env_mutex_);
    
    if (!initialized_) {
        throw std::runtime_error("NCCL environment not initialized");
    }
    
    // In a real implementation, this would configure NCCL error handling
    // For now, we'll just log the setting
    std::cout << "NCCL error handling mode set to: " << (enable_async ? "async" : "sync") << std::endl;
}

/**
 * @brief Cleanup and shutdown NCCL environment
 */
void NCCLEnvironment::shutdown() {
    std::lock_guard<std::mutex> lock(env_mutex_);
    
    if (initialized_ && communicator_ != nullptr) {
        ncclCommDestroy(communicator_);
        communicator_ = nullptr;
        initialized_ = false;
        
        // Update memory allocation tracking
        memory_allocated_.fetch_sub(sizeof(ncclComm_t));
        
        std::cout << "NCCL Environment shutdown completed. Session: " << session_id_ << std::endl;
    }
}