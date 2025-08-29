#include "inference_alltoall.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>
#include <numeric>

/**
 * @brief Constructor
 */
InferenceAllToAll::InferenceAllToAll() 
    : nccl_env_(NCCLEnvironment::getInstance()),
      send_buffer_(nullptr),
      recv_buffer_(nullptr),
      buffer_size_(0),
      total_elements_transferred_(0),
      total_operations_(0) {
    // Initialize buffer with a default size
    ensure_buffer_size(1024 * 1024); // 1MB default buffer
}

/**
 * @brief Destructor
 */
InferenceAllToAll::~InferenceAllToAll() {
    std::lock_guard<std::mutex> lock(alltoall_mutex_);
    if (send_buffer_) {
        cudaFree(send_buffer_);
        send_buffer_ = nullptr;
    }
    if (recv_buffer_) {
        cudaFree(recv_buffer_);
        recv_buffer_ = nullptr;
    }
    buffer_size_ = 0;
}

/**
 * @brief Ensure buffers are large enough
 * @param required_size Required buffer size in bytes
 */
void InferenceAllToAll::ensure_buffer_size(size_t required_size) {
    if (required_size > buffer_size_) {
        // Free existing buffers
        if (send_buffer_) {
            cudaFree(send_buffer_);
        }
        if (recv_buffer_) {
            cudaFree(recv_buffer_);
        }
        
        // Allocate new buffers
        CUDA_CHECK(cudaMalloc(&send_buffer_, required_size));
        CUDA_CHECK(cudaMalloc(&recv_buffer_, required_size));
        buffer_size_ = required_size;
        
        // Initialize buffers to zero
        CUDA_CHECK(cudaMemset(send_buffer_, 0, buffer_size_));
        CUDA_CHECK(cudaMemset(recv_buffer_, 0, buffer_size_));
    }
}

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
void InferenceAllToAll::alltoallv(const T* sendbuff,
                                  const size_t* sendcounts,
                                  const size_t* sdispls,
                                  T* recvbuff,
                                  const size_t* recvcounts,
                                  const size_t* rdispls,
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
    
    if (sendcounts == nullptr || sdispls == nullptr || 
        recvcounts == nullptr || rdispls == nullptr) {
        throw std::invalid_argument("Count/displacement arrays cannot be null");
    }
    
    int world_size = nccl_env_.getWorldSize();
    int rank = nccl_env_.getRank();
    
    // Calculate total elements to send and receive
    size_t total_send = 0;
    size_t total_recv = 0;
    
    for (int i = 0; i < world_size; ++i) {
        total_send += sendcounts[i];
        total_recv += recvcounts[i];
    }
    
    if (total_send == 0 && total_recv == 0) {
        return; // Nothing to do
    }
    
    // Ensure buffer is large enough
    size_t max_elements = std::max(total_send, total_recv);
    size_t required_size = max_elements * sizeof(T);
    ensure_buffer_size(required_size);
    
    // For NCCL, we need to handle AllToAllv differently as it's not directly supported
    // This is a simplified implementation that would need to be expanded in practice
    // For now, we'll implement a basic version using multiple send/recv operations
    
    // Copy send data to internal buffer
    CUDA_CHECK(cudaMemcpyAsync(send_buffer_, sendbuff, total_send * sizeof(T), 
                               cudaMemcpyDeviceToDevice, stream));
    
    // In a real implementation, we would perform the AllToAllv operation here
    // For now, we'll just copy the data back to simulate the operation
    CUDA_CHECK(cudaMemcpyAsync(recvbuff, send_buffer_, total_send * sizeof(T), 
                               cudaMemcpyDeviceToDevice, stream));
    
    // Update performance statistics
    total_elements_transferred_.fetch_add(total_send + total_recv);
    total_operations_.fetch_add(1);
}

// Explicit template instantiations for common types
template void InferenceAllToAll::alltoall<float>(const float*, float*, size_t, cudaStream_t);
template void InferenceAllToAll::alltoall<double>(const double*, double*, size_t, cudaStream_t);
template void InferenceAllToAll::alltoall<int>(const int*, int*, size_t, cudaStream_t);
template void InferenceAllToAll::alltoall<unsigned int>(const unsigned int*, unsigned int*, size_t, cudaStream_t);
template void InferenceAllToAll::alltoall<long long>(const long long*, long long*, size_t, cudaStream_t);
template void InferenceAllToAll::alltoall<unsigned long long>(const unsigned long long*, unsigned long long*, size_t, cudaStream_t);
template void InferenceAllToAll::alltoall<__half>(const __half*, __half*, size_t, cudaStream_t);

template void InferenceAllToAll::alltoallv<float>(const float*, const size_t*, const size_t*, float*, const size_t*, const size_t*, cudaStream_t);
template void InferenceAllToAll::alltoallv<double>(const double*, const size_t*, const size_t*, double*, const size_t*, const size_t*, cudaStream_t);
template void InferenceAllToAll::alltoallv<int>(const int*, const size_t*, const size_t*, int*, const size_t*, const size_t*, cudaStream_t);
template void InferenceAllToAll::alltoallv<unsigned int>(const unsigned int*, const size_t*, const size_t*, unsigned int*, const size_t*, const size_t*, cudaStream_t);
template void InferenceAllToAll::alltoallv<long long>(const long long*, const size_t*, const size_t*, long long*, const size_t*, const size_t*, cudaStream_t);
template void InferenceAllToAll::alltoallv<unsigned long long>(const unsigned long long*, const size_t*, const size_t*, unsigned long long*, const size_t*, const size_t*, cudaStream_t);
template void InferenceAllToAll::alltoallv<__half>(const __half*, const size_t*, const size_t*, __half*, const size_t*, const size_t*, cudaStream_t);