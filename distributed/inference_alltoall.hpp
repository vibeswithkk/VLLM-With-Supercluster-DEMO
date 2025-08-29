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
 * @brief Enterprise-grade AllToAll implementation for distributed inference
 * 
 * This class provides optimized AllToAll operations for distributed GPU computing
 * with support for various data types, error handling, performance monitoring,
 * and security features.
 */
class InferenceAllToAll {
private:
    // Reference to NCCL environment
    NCCLEnvironment& nccl_env_;
    
    // Thread safety
    mutable std::mutex alltoall_mutex_;
    
    // Performance tracking
    std::atomic<uint64_t> total_elements_transferred_;
    std::atomic<uint64_t> total_operations_;
    
    // Buffer management
    void* send_buffer_;
    void* recv_buffer_;
    size_t buffer_size_;
    
    /**
     * @brief Ensure buffers are large enough
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
    InferenceAllToAll();
    
    /**
     * @brief Destructor
     */
    ~InferenceAllToAll();
    
    /**
     * @brief Perform AllToAll operation
     * @tparam T Data type
     * @param sendbuff Source buffer
     * @param recvbuff Destination buffer
     * @param count Number of elements per rank
     * @param stream CUDA stream
     */
    template<typename T>
    void alltoall(const T* sendbuff, 
                  T* recvbuff, 
                  size_t count, 
                  cudaStream_t stream = 0);
    
    /**
     * @brief Perform AllToAllv operation with variable counts
     * @tparam T Data type
     * @param sendbuff Source buffer
     * @param sendcounts Number of elements to send to each rank
     * @param sdispls Displacements for send buffer
     * @param recvbuff Destination buffer
     * @param recvcounts Number of elements to receive from each rank
     * @param rdispls Displacements for receive buffer
     * @param stream CUDA stream
     */
    template<typename T>
    void alltoallv(const T* sendbuff,
                   const size_t* sendcounts,
                   const size_t* sdispls,
                   T* recvbuff,
                   const size_t* recvcounts,
                   const size_t* rdispls,
                   cudaStream_t stream = 0);
    
    /**
     * @brief Get performance statistics
     * @param total_elements Reference to store total elements transferred
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
inline ncclDataType_t InferenceAllToAll::get_nccl_datatype<float>() {
    return ncclFloat32;
}

template<>
inline ncclDataType_t InferenceAllToAll::get_nccl_datatype<double>() {
    return ncclFloat64;
}

template<>
inline ncclDataType_t InferenceAllToAll::get_nccl_datatype<int>() {
    return ncclInt32;
}

template<>
inline ncclDataType_t InferenceAllToAll::get_nccl_datatype<unsigned int>() {
    return ncclUint32;
}

template<>
inline ncclDataType_t InferenceAllToAll::get_nccl_datatype<long long>() {
    return ncclInt64;
}

template<>
inline ncclDataType_t InferenceAllToAll::get_nccl_datatype<unsigned long long>() {
    return ncclUint64;
}

template<>
inline ncclDataType_t InferenceAllToAll::get_nccl_datatype<__half>() {
    return ncclFloat16;
}

// Template implementation for alltoall
template<typename T>
void InferenceAllToAll::alltoall(const T* sendbuff, 
                                 T* recvbuff, 
                                 size_t count, 
                                 cudaStream_t stream) {
    std::lock_guard<std::mutex> lock(alltoall_mutex_);
    
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
    
    int world_size = nccl_env_.getWorldSize();
    size_t total_count = count * world_size;
    
    // Ensure buffer is large enough
    size_t required_size = total_count * sizeof(T);
    ensure_buffer_size(required_size);
    
    // Perform AllToAll operation
    ncclDataType_t datatype = get_nccl_datatype<T>();
    NCCL_CHECK(ncclAllToAll(sendbuff, recvbuff, count, datatype, 
                            nccl_env_.getCommunicator(), stream));
    
    // Update performance statistics
    total_elements_transferred_.fetch_add(total_count);
    total_operations_.fetch_add(1);
}

// Inline method implementations
inline void InferenceAllToAll::getPerformanceStats(uint64_t& total_elements, uint64_t& total_operations) const {
    total_elements = total_elements_transferred_.load();
    total_operations = total_operations_.load();
}

inline void InferenceAllToAll::resetPerformanceStats() {
    total_elements_transferred_.store(0);
    total_operations_.store(0);
}

inline size_t InferenceAllToAll::getBufferSize() const {
    return buffer_size_;
}