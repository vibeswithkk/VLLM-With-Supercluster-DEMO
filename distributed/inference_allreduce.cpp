#include "inference_allreduce.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <iostream>

/**
 * @brief Constructor
 */
InferenceAllReduce::InferenceAllReduce() 
    : nccl_env_(NCCLEnvironment::getInstance()),
      temp_buffer_(nullptr),
      buffer_size_(0),
      total_elements_reduced_(0),
      total_operations_(0) {
    // Initialize buffer with a default size
    ensure_buffer_size(1024 * 1024); // 1MB default buffer
}

/**
 * @brief Destructor
 */
InferenceAllReduce::~InferenceAllReduce() {
    std::lock_guard<std::mutex> lock(allreduce_mutex_);
    if (temp_buffer_) {
        cudaFree(temp_buffer_);
        temp_buffer_ = nullptr;
        buffer_size_ = 0;
    }
}

/**
 * @brief Ensure buffer is large enough
 * @param required_size Required buffer size in bytes
 */
void InferenceAllReduce::ensure_buffer_size(size_t required_size) {
    if (required_size > buffer_size_) {
        // Free existing buffer
        if (temp_buffer_) {
            cudaFree(temp_buffer_);
        }
        
        // Allocate new buffer
        CUDA_CHECK(cudaMalloc(&temp_buffer_, required_size));
        buffer_size_ = required_size;
        
        // Initialize buffer to zero
        CUDA_CHECK(cudaMemset(temp_buffer_, 0, buffer_size_));
    }
}

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
void InferenceAllReduce::allreduce_with_custom_op(const T* sendbuff, 
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
    
    // For custom operations, we might need to use the temporary buffer
    // This is a simplified implementation - in practice, custom operations
    // might require more sophisticated handling
    
    // Perform AllReduce operation
    ncclDataType_t datatype = get_nccl_datatype<T>();
    NCCL_CHECK(ncclAllReduce(sendbuff, recvbuff, count, datatype, op, 
                             nccl_env_.getCommunicator(), stream));
    
    // Update performance statistics
    total_elements_reduced_.fetch_add(count);
    total_operations_.fetch_add(1);
}

// Explicit template instantiations for common types
template void InferenceAllReduce::allreduce<float>(const float*, float*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce<double>(const double*, double*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce<int>(const int*, int*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce<unsigned int>(const unsigned int*, unsigned int*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce<long long>(const long long*, long long*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce<unsigned long long>(const unsigned long long*, unsigned long long*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce<__half>(const __half*, __half*, size_t, ncclRedOp_t, cudaStream_t);

template void InferenceAllReduce::allreduce_with_custom_op<float>(const float*, float*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce_with_custom_op<double>(const double*, double*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce_with_custom_op<int>(const int*, int*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce_with_custom_op<unsigned int>(const unsigned int*, unsigned int*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce_with_custom_op<long long>(const long long*, long long*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce_with_custom_op<unsigned long long>(const unsigned long long*, unsigned long long*, size_t, ncclRedOp_t, cudaStream_t);
template void InferenceAllReduce::allreduce_with_custom_op<__half>(const __half*, __half*, size_t, ncclRedOp_t, cudaStream_t);