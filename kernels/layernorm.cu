#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cmath>
#include <cstdint>
#include <cassert>

// Enterprise-grade error handling macros
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

#define CUBLAS_SAFE_CALL(call) do { \
    cublasStatus_t err = call; \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS error at %s:%d - %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

// Constants for numerical stability and performance
constexpr float kEpsilon = 1e-5f;
constexpr int kWarpSize = 32;
constexpr int kMaxThreadsPerBlock = 1024;

// Optimized layer normalization kernel with warp-level primitives
__global__ void layernorm_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int64_t rows,
    const int64_t cols,
    const float epsilon
) {
    // Calculate thread and block indices
    const int64_t row = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t block_size = blockDim.x;
    
    // Bounds checking
    if (row >= rows) return;
    
    // Calculate offsets
    const int64_t row_offset = row * cols;
    const float* x = input + row_offset;
    float* y = output + row_offset;
    
    // Compute mean using warp-level reductions
    float sum = 0.0f;
    for (int64_t i = tid; i < cols; i += block_size) {
        sum += x[i];
    }
    
    // Warp-level reduction for sum
    for (int64_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    // Broadcast mean to all threads in warp
    const float mean = sum / static_cast<float>(cols);
    
    // Compute variance using warp-level reductions
    float sum_sq = 0.0f;
    for (int64_t i = tid; i < cols; i += block_size) {
        const float diff = x[i] - mean;
        sum_sq += diff * diff;
    }
    
    // Warp-level reduction for sum of squares
    for (int64_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    
    // Broadcast variance to all threads in warp
    const float variance = sum_sq / static_cast<float>(cols);
    const float inv_std = rsqrtf(variance + epsilon);
    
    // Normalize and apply weight/bias
    for (int64_t i = tid; i < cols; i += block_size) {
        const float normalized = (x[i] - mean) * inv_std;
        y[i] = normalized * weight[i] + bias[i];
    }
}

// Specialized kernel for small dimensions
__global__ void layernorm_kernel_small(
    const float* __restrict__ input,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ output,
    const int64_t rows,
    const int64_t cols,
    const float epsilon
) {
    const int64_t row = blockIdx.x;
    const int64_t tid = threadIdx.x;
    
    if (row >= rows) return;
    
    // Shared memory for reduction
    extern __shared__ float shared_mem[];
    float* shared_data = shared_mem;
    float* shared_sum = shared_mem;
    float* shared_sum_sq = shared_mem + kWarpSize;
    
    const int64_t row_offset = row * cols;
    const float* x = input + row_offset;
    float* y = output + row_offset;
    
    // Initialize shared memory
    if (tid < kWarpSize) {
        shared_sum[tid] = 0.0f;
        shared_sum_sq[tid] = 0.0f;
    }
    __syncthreads();
    
    // Compute partial sums
    float local_sum = 0.0f;
    float local_sum_sq = 0.0f;
    
    for (int64_t i = tid; i < cols; i += blockDim.x) {
        local_sum += x[i];
        local_sum_sq += x[i] * x[i];
    }
    
    // Reduction within warp
    for (int64_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
        local_sum_sq += __shfl_down_sync(0xFFFFFFFF, local_sum_sq, offset);
    }
    
    // Store partial results in shared memory
    const int64_t warp_id = tid / kWarpSize;
    const int64_t lane_id = tid % kWarpSize;
    
    if (lane_id == 0) {
        shared_sum[warp_id] = local_sum;
        shared_sum_sq[warp_id] = local_sum_sq;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        float sum = (lane_id < (blockDim.x + kWarpSize - 1) / kWarpSize) ? shared_sum[lane_id] : 0.0f;
        float sum_sq = (lane_id < (blockDim.x + kWarpSize - 1) / kWarpSize) ? shared_sum_sq[lane_id] : 0.0f;
        
        for (int64_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
            sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
            sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
        }
        
        if (lane_id == 0) {
            shared_sum[0] = sum;
            shared_sum_sq[0] = sum_sq;
        }
    }
    __syncthreads();
    
    // Broadcast results to all threads
    const float mean = shared_sum[0] / static_cast<float>(cols);
    const float variance = shared_sum_sq[0] / static_cast<float>(cols) - mean * mean;
    const float inv_std = rsqrtf(variance + epsilon);
    
    // Normalize and apply weight/bias
    for (int64_t i = tid; i < cols; i += blockDim.x) {
        const float normalized = (x[i] - mean) * inv_std;
        y[i] = normalized * weight[i] + bias[i];
    }
}

// Host function with enhanced error checking and performance optimization
cudaError_t launch_layernorm(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    const int64_t rows,
    const int64_t cols,
    const float epsilon,
    cudaStream_t stream
) {
    // Input validation
    if (!input || !weight || !bias || !output) {
        return cudaErrorInvalidValue;
    }
    
    if (rows <= 0 || cols <= 0) {
        return cudaErrorInvalidValue;
    }
    
    // Choose optimal kernel based on dimension size
    if (cols <= 1024) {
        // Use specialized kernel for small dimensions
        const int64_t block_size = min(kMaxThreadsPerBlock, max(kWarpSize, (cols + 31) / 32 * 32));
        const dim3 block_dim(block_size);
        const dim3 grid_dim(rows);
        
        // Calculate shared memory size
        const int64_t warps_per_block = (block_size + kWarpSize - 1) / kWarpSize;
        const size_t shared_mem_size = 2 * warps_per_block * sizeof(float);
        
        layernorm_kernel_small<<<grid_dim, block_dim, shared_mem_size, stream>>>(
            input, weight, bias, output, rows, cols, epsilon
        );
    } else {
        // Use general kernel for large dimensions
        const int64_t block_size = min(kMaxThreadsPerBlock, max(kWarpSize, (cols + 31) / 32 * 32));
        const dim3 block_dim(block_size);
        const dim3 grid_dim(rows);
        
        layernorm_kernel<<<grid_dim, block_dim, 0, stream>>>(
            input, weight, bias, output, rows, cols, epsilon
        );
    }
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return err;
    }
    
    return cudaSuccess;
}

// Template version for different data types
template<typename T>
__global__ void layernorm_kernel_template(
    const T* __restrict__ input,
    const T* __restrict__ weight,
    const T* __restrict__ bias,
    T* __restrict__ output,
    const int64_t rows,
    const int64_t cols,
    const T epsilon
) {
    const int64_t row = blockIdx.x;
    const int64_t tid = threadIdx.x;
    const int64_t block_size = blockDim.x;
    
    if (row >= rows) return;
    
    const int64_t row_offset = row * cols;
    const T* x = input + row_offset;
    T* y = output + row_offset;
    
    // Compute mean
    T sum = T(0);
    for (int64_t i = tid; i < cols; i += block_size) {
        sum += x[i];
    }
    
    // Reduction
    for (int64_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
        sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
    }
    
    const T mean = sum / static_cast<T>(cols);
    
    // Compute variance
    T sum_sq = T(0);
    for (int64_t i = tid; i < cols; i += block_size) {
        const T diff = x[i] - mean;
        sum_sq += diff * diff;
    }
    
    // Reduction
    for (int64_t offset = kWarpSize / 2; offset > 0; offset /= 2) {
        sum_sq += __shfl_down_sync(0xFFFFFFFF, sum_sq, offset);
    }
    
    const T variance = sum_sq / static_cast<T>(cols);
    const T inv_std = rsqrt(variance + epsilon);
    
    // Normalize and apply weight/bias
    for (int64_t i = tid; i < cols; i += block_size) {
        const T normalized = (x[i] - mean) * inv_std;
        y[i] = normalized * weight[i] + bias[i];
    }
}

// Explicit template instantiations
template __global__ void layernorm_kernel_template<float>(
    const float*, const float*, const float*, float*, const int64_t, const int64_t, const float);
template __global__ void layernorm_kernel_template<double>(
    const double*, const double*, const double*, double*, const int64_t, const int64_t, const double);