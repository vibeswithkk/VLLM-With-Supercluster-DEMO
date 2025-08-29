#pragma once

#include "nccl_env.hpp"
#include <cuda_runtime.h>
#include <nccl.h>
#include <memory>
#include <vector>
#include <mutex>
#include <atomic>
#include <stdexcept>

/**
 * @brief Enterprise-grade AllReduce implementation for distributed inference
 * 
 * This class provides optimized AllReduce operations for distributed GPU computing
 * with support for various data types, error handling, performance monitoring,
 * and security features.
 */
class InferenceAllReduce {
private:
    // Reference to NCCL environment
    NCCLEnvironment& nccl_env_;
    
    // Thread safety
    mutable std::mutex allreduce_mutex_;
    
    // Performance tracking
    std::atomic<uint64_t> total_elements_reduced_;
    std::atomic<uint64_t> total_operations_;
    
    // Buffer management
    void* temp_buffer_;
    size_t buffer_size_;
    
    /**
     * @brief Ensure buffer is large enough
     * @param required_size Required buffer size in bytes
     */
    void ensure_buffer_size(size_t required_size);
    
    /**
     * @brief Get NCCL data type from template parameter
     * @tparam T Data type
     * @return NCCL data type
     */
    template<typename T>
    ncclDataType_t get_nccl_datatype();
    
public:
    /**
     * @brief Constructor
     */
    InferenceAllReduce();
    
    /**
     * @brief Destructor
     */
    ~InferenceAllReduce();
    
    /**
     * @brief Perform AllReduce operation
     * @tparam T Data type
     * @param sendbuff Source buffer
     * @param recvbuff Destination buffer
     * @param count Number of elements
     * @param op Reduction operation
     * @param stream CUDA stream
     */
    template<typename T>
    void allreduce(const T* sendbuff, 
                   T* recvbuff, 
                   size_t count, 
                   ncclRedOp_t op = ncclSum, 
                   cudaStream_t stream = 0);
    
    /**
     * @brief Perform AllReduce operation with custom reduction function
     * @tparam T Data type
     * @param sendbuff Source buffer
     * @param recvbuff Destination buffer
     * @param count Number of elements
     * @param op Custom reduction operation
     * @param stream CUDA stream
     */
    template<typename T>
    void allreduce_with_custom_op(const T* sendbuff, 
                                  T* recvbuff, 
                                  size_t count, 
                                  ncclRedOp_t op, 
                                  cudaStream_t stream = 0);
    
    /**
     * @brief Get performance statistics
     * @param total_elements Reference to store total elements reduced
     * @param total_operations Reference to store total operations
     */
    void getPerformanceStats(uint64_t& total_elements, uint64_t& total_operations) const;
    
    /**
     * @brief Reset performance statistics
     */
    void resetPerformanceStats();
    
    /**
     * @brief Get buffer usage statistics
     * @return Current buffer size in bytes
     */
    size_t getBufferSize() const;
};

// Template specializations for NCCL data types
template<>
inline ncclDataType_t InferenceAllReduce::get_nccl_datatype<float>() {
    return ncclFloat32;
}

template<>
inline ncclDataType_t InferenceAllReduce::get_nccl_datatype<double>() {
    return ncclFloat64;
}

template<>
inline ncclDataType_t InferenceAllReduce::get_nccl_datatype<int>() {
    return ncclInt32;
}

template<>
inline ncclDataType_t InferenceAllReduce::get_nccl_datatype<unsigned int>() {
    return ncclUint32;
}

template<>
inline ncclDataType_t InferenceAllReduce::get_nccl_datatype<long long>() {
    return ncclInt64;
}

template<>
inline ncclDataType_t InferenceAllReduce::get_nccl_datatype<unsigned long long>() {
    return ncclUint64;
}

template<>
inline ncclDataType_t InferenceAllReduce::get_nccl_datatype<__half>() {
    return ncclFloat16;
}

// Template implementation for allreduce
template<typename T>
void InferenceAllReduce::allreduce(const T* sendbuff, 
                                   T* recvbuff, 
                                   size_t count, 
                                   ncclRedOp_t op, 
                                   cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(allreduce_mutex_);
    
    // Validate environment
    if (!nccl_env_.isInitialized()) {
        throw std::runtime_error("NCCL environment not initialized");
    }
    
    // Validate parameters
    if (sendbuff == nullptr || recvbuff == nullptr) {
        throw std::invalid_argument("Buffer pointers cannot be null");
    }
    
    if (count == 0) {
        return; // Nothing to do
    }
    
    // Ensure buffer is large enough
    size_t required_size = count * sizeof(T);
    ensure_buffer_size(required_size);
    
    // Perform AllReduce operation
    ncclDataType_t datatype = get_nccl_datatype<T>();
    NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, datatype, op, 
                             nccl_env_.getCommunicator(), stream));
    
    // Update performance statistics
    total_elements_reduced_.fetch_add(count);
    total_operations_.fetch_add(1);
}

// Inline method implementations
inline void InferenceAllReduce::getPerformanceStats(uint64_t& total_elements, uint64_t& total_operations) const {
    total_elements = total_elements_reduced_.load();
    total_operations = total_operations_.load();
}

inline void InferenceAllReduce::resetPerformanceStats() {
    total_elements_reduced_.store(0);
    total_operations_.store(0);
}

inline size_t InferenceAllReduce::getBufferSize() const {
    return buffer_size_;
}